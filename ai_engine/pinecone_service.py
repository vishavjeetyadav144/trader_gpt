import logging
import hashlib
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from django.conf import settings

from ai_engine.models import LongTermMemory, AIDecision
from openai import OpenAI

logger = logging.getLogger(__name__)


class PineconeMemoryService:
    """Pinecone integration for AI long-term memory"""
    
    def __init__(self):
        self.api_key = settings.AI_CONFIG['PINECONE']['API_KEY']
        self.environment = settings.AI_CONFIG['PINECONE']['ENVIRONMENT']
        self.index_name = settings.AI_CONFIG['PINECONE']['INDEX_NAME']
        self.openai_api_key = settings.AI_CONFIG['OPENAI']['API_KEY'] 
        
        if not self.api_key:
            raise ValueError("Pinecone API key not configured")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not configured")

        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        self.openai_client = OpenAI(api_key=self.openai_api_key)


        self.embedding_model = "text-embedding-3-large"
        self.dimension = 3072  # ✅ dimension for text-embedding-3-large

        if self.index_name not in [i.name for i in self.pc.list_indexes()]:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        self.index = self.pc.Index(self.index_name)

    
    def store_decision_memory(self, ai_decision: AIDecision, outcome_data: Optional[Dict] = None):
        """Store AI decision in long-term memory"""
        try:
            if ai_decision.decision_type.lower() == "hold":
                logger.info(f"Skipping HOLD decision for {ai_decision.symbol}")
                return None
            # Create memory content
            memory_content = self._create_memory_content(ai_decision, outcome_data)
            searchable_text = memory_content['searchable_text']

            # ✅ Generate embedding via OpenAI
            embedding_response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=searchable_text
            )
            embedding = embedding_response.data[0].embedding

            # Create unique ID
            memory_id = f"decision_{ai_decision.decision_id}_{int(datetime.now(timezone.utc).timestamp())}"
            
            # Store in Pinecone
            metadata = {
                "decision_type": ai_decision.decision_type,
                "symbol": ai_decision.symbol,
                "confidence": float(ai_decision.confidence_score),
                "timestamp": ai_decision.created_at.isoformat(),
                "was_profitable": outcome_data.get('was_profitable') if outcome_data else False,
                "actual_outcome_pct": float(outcome_data.get('actual_outcome_pct', 0)) if outcome_data else 0.0,
                "is_closed": False,
                "content": memory_content['full_content'][:2000]  # Truncate for metadata
            }
            
            self.index.upsert([
                (memory_id, embedding, metadata)
            ])
            
            # Also store in MongoDB for detailed retrieval
            # long_term_memory = LongTermMemory(
            #     memory_id=memory_id,
            #     memory_type='ai_decision',
            #     title=f"{ai_decision.decision_type.upper()} {ai_decision.symbol} - {ai_decision.confidence_score:.1%} confidence",
            #     description=f"AI trading decision with reasoning and market context",
            #     content=memory_content['full_content'],
            #     pinecone_vector_id=memory_id,
            #     symbols_related=[ai_decision.symbol],
            #     time_period=ai_decision.created_at,
            #     importance_score=ai_decision.confidence_score
            # )
            
            # long_term_memory.save()
            # logger.info(f"Stored decision memory: {memory_id}")
            
            return memory_id
            
        except Exception as e:
            logger.error(f"Error storing decision memory: {e}")
            return None
    
    def find_similar_decisions(self, current_reasoning: str, symbol: str, market_context: Optional[Dict] = None, risk_factors: Optional[List[str]] = None, top_k: int = 3, min_similarity: float = 0.8) -> List[Dict]:
        """Find similar past decisions using semantic search"""
        try:
            # Create search query embedding
            query_parts = [f"Symbol: {symbol}", f"Reasoning: {current_reasoning}"]
            if risk_factors:
                query_parts.append(f"Risk factors: {', '.join(risk_factors)}")

            if market_context:
                market_summary = self.summarize_market_context(market_context)
                query_parts.append(f"Market context summary: {market_summary}")

            query_text = " ".join(query_parts)
            embedding_response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=query_text
            )
            query_embedding = embedding_response.data[0].embedding

            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                filter={
                    "symbol": {"$eq": symbol},
                    "is_closed": {"$eq": True}
                },
                top_k=top_k,
                include_metadata=True
            )
            
            similar_memories = []
            
            for match in results.get('matches', []):
                logger.error(f'min_similarity { match["score"]}')
                if match["score"] < min_similarity:
                    continue

                memory_data = {
                    "similarity_score": match['score'],
                    "decision_type": match['metadata'].get('decision_type'),
                    "confidence": match['metadata'].get('confidence'),
                    "timestamp": match['metadata'].get('timestamp'),
                    "was_profitable": match['metadata'].get('was_profitable'),
                    "actual_outcome_pct": match['metadata'].get('actual_outcome_pct'),
                    "reasoning_summary": match['metadata'].get('content', '')[:500]
                }
                
                # Get full content from MongoDB if needed
                full_memory = LongTermMemory.objects(
                    pinecone_vector_id=match['id']
                ).first()
                
                if full_memory:
                    memory_data['full_reasoning'] = full_memory.content
                    memory_data['importance_score'] = float(full_memory.importance_score or 0)
                    full_memory.mark_accessed()
                
                similar_memories.append(memory_data)
            
            logger.info(f"Found {len(similar_memories)} similar decisions for {symbol}")
            return similar_memories
            
        except Exception as e:
            logger.error(f"Error finding similar decisions: {e}")
            return []
    
    def get_memory_summary(self, similar_memories: List[Dict]) -> str:
        """Generate a summary of similar past decisions"""
        if not similar_memories:
            return "No similar past decisions found."
        
        summary_parts = []
        
        for i, memory in enumerate(similar_memories, 1):
            outcome_text = "Unknown outcome"
            if memory.get('was_profitable') is not None:
                if memory['was_profitable']:
                    outcome_text = f"Profitable (+{memory.get('actual_outcome_pct', 0):.2f}%)"
                else:
                    outcome_text = f"Loss ({memory.get('actual_outcome_pct', 0):.2f}%)"
            
            summary_parts.append(f"""
            SIMILAR DECISION #{i} (Similarity: {memory['similarity_score']:.2f}):
            - Decision: {memory['decision_type'].upper()}
            - Confidence: {memory['confidence']:.1%}
            - Outcome: {outcome_text}
            - Date: {memory['timestamp'][:10]}
            - Key Reasoning: {memory['reasoning_summary']}
            """)
        
        return "PAST SIMILAR DECISIONS FOR CONTEXT:\n" + "\n".join(summary_parts)
    
    def summarize_market_context(self, market_ctx: Dict) -> str:
        """Summarize intraday and higher timeframe indicator snapshots (no series)."""
        if not market_ctx:
            return "No market context available."

        try:
            # --- Basic Market Snapshot ---
            funding_rate = float(market_ctx.get("funding_rate", 0))
            mark_change_24h = float(market_ctx.get("mark_change_24h", 0))
            open_interest = market_ctx.get("open_interest", "N/A")

            base_summary = (
                f"24h Change: {mark_change_24h:.2f}%, "
                f"Funding Rate: {funding_rate:.4f}, Open Interest: {open_interest}."
            )

            # --- Intraday (1m) Snapshot ---
            intraday = market_ctx.get("intraday series (1m interval)", {})
            rsi_1m = intraday.get("rsi_14", [])
            macd_1m = intraday.get("macd", [])
            ema_1m = intraday.get("ema20", [])
            intraday_summary = ""
            if rsi_1m and macd_1m and ema_1m:
                intraday_summary = (
                    f" 1m → RSI {rsi_1m[-1]:.1f}, MACD {macd_1m[-1]:.1f}, EMA20 {ema_1m[-1]:.1f}."
                )

            # --- Higher Timeframes (use only 'current' section) ---
            higher_summaries = []
            for key, tf_data in market_ctx.items():
                if not key.startswith("higher timeframe"):
                    continue

                tf_name = key.split("(")[-1].replace(")", "").strip()
                current = tf_data.get("current", {})

                if not current:
                    continue

                rsi = current.get("rsi_14")
                ema20 = current.get("ema20")
                ema50 = current.get("ema50")
                atr14 = current.get("atr_14")
                atr3 = current.get("atr_3")
                vol = current.get("volume")
                avg_vol = current.get("avg_volume")

                parts = [f"{tf_name.upper()} →"]
                if rsi is not None:
                    parts.append(f"RSI {rsi:.1f}")
                if ema20 is not None and ema50 is not None:
                    parts.append(f"EMA20 {ema20:.1f}, EMA50 {ema50:.1f}")
                if atr14 is not None:
                    parts.append(f"ATR14 {atr14:.1f}")
                if atr3 is not None:
                    parts.append(f"ATR3 {atr3:.1f}")
                if vol is not None and avg_vol is not None:
                    parts.append(f"Volume {vol}, AvgVol {avg_vol:.1f}")

                higher_summaries.append(" ".join(parts))

            higher_summary = " | ".join(higher_summaries) if higher_summaries else ""
            return f"{base_summary}{intraday_summary} {higher_summary}".strip()

        except Exception as e:
            logger.error(f"Error summarizing market context: {e}")
            return "Market context unavailable due to parsing error."

    def _create_memory_content(self, ai_decision: AIDecision, outcome_data: Optional[Dict]) -> Dict[str, str]:
        """Create searchable and full content for memory storage"""
        
        # Create searchable text (for embeddings)
        searchable_parts = [
            f"Trading decision: {ai_decision.decision_type}",
            f"Symbol: {ai_decision.symbol}",
            f"Reasoning: {ai_decision.reasoning}",
        ]
        
        if ai_decision.risk_factors:
            searchable_parts.append(f"Key factors: {', '.join(ai_decision.risk_factors)}")
        
        if ai_decision.market_context:
            market_summary = self.summarize_market_context(ai_decision.market_context)
            searchable_parts.append(f"Market context summary: {market_summary}")

        # Create full content (for detailed storage)
        full_content_parts = [
            f"TRADING DECISION RECORD",
            f"Symbol: {ai_decision.symbol}",
            f"Decision: {ai_decision.decision_type.upper()}",
            f"REASONING:",
            f"{ai_decision.reasoning}",
            f"",
            f"KEY FACTORS:",
            f"{', '.join(ai_decision.risk_factors) if ai_decision.risk_factors else 'None'}",
        ]

        if ai_decision.market_context:
            full_content_parts.extend([
                "",
                "MARKET CONTEXT SUMMARY:",
                self.summarize_market_context(ai_decision.market_context),
            ])
            
        if outcome_data:
            full_content_parts.extend([
                f"",
                f"ACTUAL OUTCOME:",
                f"Profitable: {'Yes' if outcome_data.get('was_profitable') else 'No'}",
                f"Actual Return: {outcome_data.get('actual_outcome_pct', 0):.2f}%",
                f"Outcome Date: {outcome_data.get('outcome_date', 'N/A')}"
            ])
        
        return {
            "searchable_text": " ".join(searchable_parts),
            "full_content": "\n".join(full_content_parts)
        }
    
    def update_decision_performance(self, pinecone_id: str, was_profitable: bool, actual_pnl: float, price_change_pct: float, exit_price: float):
        """Update decision performance in Pinecone memory"""
        try:
            # Update Pinecone metadata
            fetch_response = self.index.fetch(ids=[pinecone_id])
            vectors = fetch_response.vectors

            if pinecone_id in vectors:
                vector_data = vectors[pinecone_id]
                current_metadata = dict(vector_data.metadata or {})

                # Update metadata
                current_metadata.update({
                    'was_profitable': was_profitable,
                    'actual_outcome_pct': price_change_pct,
                    'actual_pnl_usd': actual_pnl,
                    'exit_price': exit_price,
                    'outcome_updated_at': datetime.now(timezone.utc).isoformat(),
                    'is_closed': True
                })

                # Re-upsert with updated metadata
                self.index.upsert(vectors=[
                    {
                        "id": pinecone_id,
                        "values": vector_data.values,
                        "metadata": current_metadata
                    }
                ])

                logger.info(f"✅ Updated Pinecone decision performance for {pinecone_id}: Profitable={was_profitable}, PnL=${actual_pnl}")
            else:
                logger.warning(f"⚠️ Pinecone vector {pinecone_id} not found for performance update")

        except Exception as e:
            logger.error(f"Error updating decision performance in Pinecone: {e}")

    def update_memory_outcome(self, memory_id: str, outcome_data: Dict):
        """Update stored memory with actual trading outcome"""
        try:
            # Update Pinecone metadata
            current_data = self.index.fetch([memory_id])
            if memory_id in current_data.get('vectors', {}):
                current_metadata = current_data['vectors'][memory_id]['metadata']
                current_metadata.update({
                    'was_profitable': outcome_data.get('was_profitable'),
                    'actual_outcome_pct': float(outcome_data.get('actual_outcome_pct', 0))
                })
                
                # Re-upsert with updated metadata
                vector = current_data['vectors'][memory_id]['values']
                self.index.upsert([(memory_id, vector, current_metadata)])
            
            # Update MongoDB record
            # memory_record = LongTermMemory.objects(pinecone_vector_id=memory_id).first()
            # if memory_record:
            #     outcome_text = f"\n\nACTUAL OUTCOME:\nProfitable: {'Yes' if outcome_data.get('was_profitable') else 'No'}\nActual Return: {outcome_data.get('actual_outcome_pct', 0):.2f}%"
            #     memory_record.content += outcome_text
            #     memory_record.save()
            
            logger.error(f"Updated memory outcome for {memory_id}")
            
        except Exception as e:
            logger.error(f"Error updating memory outcome: {e}")
