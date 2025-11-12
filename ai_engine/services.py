import json
import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import requests
from openai import OpenAI
from django.conf import settings
from exchanges.services import DeltaExchangeClient, TechnicalAnalysisService, PortfolioSyncService
from portfolio.models import Portfolio, Position
from ai_engine.models import AIDecision, AIModel
from accounts.models import TradingUser

logger = logging.getLogger(__name__)


class LLMPromptGenerator:
    """Generate comprehensive trading prompts for LLM analysis"""
    
    def __init__(self):
        self.delta_client = DeltaExchangeClient()
        self.ta_service = TechnicalAnalysisService()
    
    def generate_trading_prompt(self, symbol="MARK:BTCUSD", timeframes=["1m", "15m", "1h", "4h"]):
        """Generate comprehensive trading analysis prompt"""
        try:
            tf_minutes = {
                "1m": 1,
                "5m": 5,
                "15m": 15,
                "1h": 60,
                "2h": 120,
                "4h": 240
            }
            
            portfolio_data = self._get_portfolio_state()
            prompt_data = {}
                        
            for tf in timeframes:
                minutes_per_candle = tf_minutes.get(tf, 60) 
                lookback_minutes = max(2000 * minutes_per_candle, 2000)

                data = self._get_timeframe_data(symbol, tf, lookback_minutes)
                if tf == "1m":
                    prompt_data[f"intraday series ({tf} interval)"] = self._extract_intraday_series(data, last_n=15)
                else:
                    prompt_data[f"higher timeframe ({tf})"] = self._extract_higher_timeframe(data)

            # Add funding rate and market context
            market_data = self._get_market_context(symbol)
            return self._format_trading_prompt(market_data | prompt_data, portfolio_data)
            
        except Exception as e:
            logger.error(f"Error generating trading prompt: {e}")
            return None
    
    def _get_portfolio_state(self):
        """Get current portfolio state"""
        try:
            portfolio = Portfolio.get_primary_portfolio()
            if not portfolio:
                return {"error": "No primary portfolio found"}
            
            # Get open positions
            positions = Position.get_open_positions()
            position_data = []
            
            for pos in positions:
                position_data.append({
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "quantity": float(pos.quantity),
                    "entry_price": float(pos.entry_price),
                    "current_price": float(pos.current_price),
                    "unrealized_pnl": float(pos.unrealized_pnl_usd),
                    "pnl_pct": float((pos.current_price - pos.entry_price) / pos.entry_price * 100)
                })
            
            return {
                "total_value": float(portfolio.total_value_usd),
                "available_balance": float(portfolio.available_balance_usd),
                "unrealized_pnl": float(portfolio.unrealized_pnl_usd),
                "open_positions": position_data,
                "positions_count": len(position_data),
                "daily_pnl_pct": float(portfolio.daily_return_pct or 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio state: {e}")
            return {"error": str(e)}
    
    def _get_timeframe_data(self, symbol, timeframe, lookback_minutes):
        """Get OHLCV data with indicators for a timeframe"""
        try:
            # Calculate time range
            now = int(datetime.now(timezone.utc).timestamp())
            start = now - lookback_minutes * 60
            
            # Fetch and process data
            df = self.delta_client.fetch_candles(symbol, timeframe, start, now)
            if df.empty:
                return None
            df = self.ta_service.compute_indicators(df)
            return df
            
        except Exception as e:
            logger.error(f"Error getting {timeframe} data: {e}")
            return None
    
    def _extract_current_snapshot(self, df):
        """Extract current market snapshot"""
        latest = df.iloc[-1]
        
        return {
            "price": round(float(latest["close"]), 2),
            "ema20": round(float(latest.get("ema20", 0)), 3),
            "ema50": round(float(latest.get("ema50", 0)), 3),
            "macd": round(float(latest.get("macd", 0)), 3),
            "macd_signal": round(float(latest.get("macd_signal", 0)), 3),
            "macd_hist": round(float(latest.get("macd_hist", 0)), 3),
            "rsi_7": round(float(latest.get("rsi_7", 0)), 3),
            "rsi_14": round(float(latest.get("rsi_14", 0)), 3),
            "atr_14": round(float(latest.get("atr_14", 0)), 3),
            "volume": round(float(latest["volume"]), 3),
            "avg_volume": round(float(latest.get("avg_volume", 0)), 3),
            "bb_upper": round(float(latest.get("bb_upper", 0)), 3),
            "bb_lower": round(float(latest.get("bb_lower", 0)), 3)
        }
    
    def _extract_intraday_series(self, df, last_n=50):
        """Extract intraday time series data"""
        # Get last N candles
        recent_df = df.tail(last_n)
        
        return {
            "prices": [round(float(x), 2) for x in recent_df["close"]],
            "ema20": [round(float(x), 3) for x in recent_df.get("ema20", [0]*len(recent_df))],
            "macd": [round(float(x), 3) for x in recent_df.get("macd", [0]*len(recent_df))],
            "macd_hist": [round(float(x), 3) for x in recent_df.get("macd_hist", [0]*len(recent_df))],
            # "rsi_7": [round(float(x), 3) for x in recent_df.get("rsi_7", [0]*len(recent_df))],
            "rsi_14": [round(float(x), 3) for x in recent_df.get("rsi_14", [0]*len(recent_df))],
            "volume": [round(float(x), 3) for x in recent_df["volume"]]
        }
    
    def _extract_higher_timeframe(self, df):
        """Extract higher timeframe analysis"""
        latest = df.iloc[-1]
        
        # Get recent series for trend analysis
        recent_df = df.tail(5)
        
        return {
            "current": {
                "ema20": round(float(latest.get("ema20", 0)), 3),
                "ema50": round(float(latest.get("ema50", 0)), 3),
                "atr_3": round(float(latest.get("atr_3", 0)), 3),
                "atr_14": round(float(latest.get("atr_14", 0)), 3),
                "volume": round(float(latest["volume"]), 3),
                "avg_volume": round(float(latest.get("avg_volume", 0)), 3),
                "rsi_14": round(float(latest.get("rsi_14", 0)), 3)
            },
            "recent_series": {
                "macd": [round(float(x), 3) for x in recent_df.get("macd", [0]*len(recent_df))],
                "rsi_14": [round(float(x), 3) for x in recent_df.get("rsi_14", [0]*len(recent_df))],
                "prices": [round(float(x), 2) for x in recent_df["close"]]
            }
        }
    
    def _get_market_context(self, symbol):
        """Get additional market context"""
        try:
            # This would typically fetch from multiple sources
            # For now, returning basic context
            data = self.delta_client.get_ticker(symbol)
            return {
                "symbol": data.get('symbol'),
                "mark_price": data.get('mark_price'),
                "funding_rate": data.get('funding_rate'),
                "open_interest": data.get('oi'),  
                "mark_change_24h": data.get('mark_change_24h')
            }
        except:
            return {}
    
    def _format_trading_prompt(self, market_data, portfolio_data):
        """Format the comprehensive trading prompt for LLM"""
        
        # Create the structured prompt exactly as requested
        
        # last_trade_str = market_data.get("last_trade_time")
        # last_trade_time = datetime.fromisoformat(last_trade_str.replace("Z", "+00:00")) if last_trade_str else datetime.utcnow()

        # minutes_since_last_trade = int((datetime.utcnow() - last_trade_time).total_seconds() / 60)
        # It has been 17804 minutes since trading started. Current time: {datetime.utcnow()}.
        # You have been invoked 6635 times.

        # Add the trading context prompt
        trading_prompt = f"""
        You are an expert crypto swing trader analyzing {market_data['symbol']}.
        Below is a complete snapshot of the environment, including market data, predictive signals, and your portfolio state.
        ---
        ⚙️ Notes:
        - All price or signal data is ordered chronologically: OLDEST → NEWEST.
        - Unless stated otherwise, intraday data is sampled at 3-minute intervals.

        ### MARKET DATA
        {json.dumps(market_data, indent=2)}
        
        ---
        ### PORTFOLIO DATA
        {json.dumps(portfolio_data, indent=2)}

        ---
        ### YOUR TASK
        Based on the above data, perform a deep trading analysis and generate a structured recommendation.

        Your analysis **must** cover:
        1. **Market Structure** — trend, key support/resistance levels.
        2. **Technical Indicators** — RSI, MACD, EMA/SMA crossovers.
        3. **Volume Analysis** — compare current volume vs average.
        4. **Higher Timeframe Context** — alignment with broader trend.
        5. **Risk Assessment** — volatility, stop-loss discipline, reward/risk ratio, and exposure size.

        ---
        ### POSITION & RISK RULES
        To ensure consistent profitability over time, strictly follow these constraints:
        - The profit target must be at least +1% from the entry price
        - Maintain a **minimum Reward-to-Risk (RRR)** of **at least 1.5:1** (e.g. aiming for +1.5% gain if risking −1%).
        - The stop-loss and take-profit levels should be chosen such that over many trades, the strategy remains **positively skewed** (expected value > 0).
        - Favor setups where technical confluence and higher-timeframe trend agree with direction.

        ---
        ### REQUIRED OUTPUT (STRICT JSON ONLY)
        Respond **ONLY** in JSON, no text outside the JSON.

        The JSON **must** strictly follow this structure:

        {{
            "trading_signal": "BUY" | "SELL" | "HOLD",
            "confidence_level": float,            # Between 0 and 100
            "entry_price": float,                 # Entry recommendation
            "stop_loss": float,
            "take_profit": [float, float],        # One or multiple take profit levels
            "position_size": int,                 # % of portfolio (integer)
            "reasoning": str,                     # Detailed explanation (max 400 chars)
            "risk_factors": [str]                # List of risk factors
        }}

        Rules:
        - Do not include any markdown, commentary, or text outside JSON.
        - **Important:** If the portfolio does **not** currently hold {market_data['symbol']}, `"HOLD"` is **not allowed** as a trading signal.
        - **Important:** If the portfolio currently holds a position in {market_data['symbol']}, allowed signals are only 'HOLD' or 'SELL' is position is 'BUY' else 'BUY' if position is 'SELL'."
        """
        
        return {
            "prompt": trading_prompt,
            "market_data": market_data,
            "portfolio_data": portfolio_data
        }


class DeepSeekLLMService:
    """DeepSeek LLM integration for trading decisions"""
    
    def __init__(self):
        self.api_key = settings.AI_CONFIG['DEEPSEEK']['API_KEY']
        self.base_url = settings.AI_CONFIG['DEEPSEEK']['BASE_URL']
        self.model = settings.AI_CONFIG['DEEPSEEK']['MODEL']
        
        if not self.api_key:
            raise ValueError("DeepSeek API key not configured")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def get_trading_decision(self, prompt_data):
        """Get trading decision from DeepSeek LLM"""
        try:
            print(f"Using model: {self.model}")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert crypto trader with deep knowledge of "
                            "technical analysis, risk management, and market psychology."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt_data["prompt"]
                    }
                ],
                temperature=1.0,
                timeout=5*60
            )

            message = response.choices[0].message.content

            return {
                "response": message,
                "usage": getattr(response, "usage", {}),
                "model": self.model
            }

        except Exception as e:
            logger.error(f"DeepSeek API error: {e}", exc_info=True)
            return None

    def parse_trading_decision(self, llm_response):
        """Parse LLM response into structured trading decision"""
        try:
            content = llm_response["response"]
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                decision_data = json.loads(json_str)

                # Handle take_profit flexibly
                take_profit_raw = decision_data.get("take_profit", [])
                if isinstance(take_profit_raw, list):
                    # You can decide how to handle: average or first
                    take_profit_value = take_profit_raw[0] #sum(take_profit_raw) / len(take_profit_raw) if take_profit_raw else 0
                else:
                    take_profit_value = float(take_profit_raw)

                return {
                    "signal": decision_data.get("trading_signal", decision_data.get("signal", "HOLD")).upper(),
                    "confidence": float(decision_data.get("confidence_level", decision_data.get("confidence", 0))),
                    "entry_price": float(decision_data.get("entry_price", 0)),
                    "stop_loss": float(decision_data.get("stop_loss", 0)),
                    "take_profit": take_profit_value,
                    "position_size_pct": float(decision_data.get("position_size", 0)),
                    "reasoning": decision_data.get("reasoning", ""),
                    "risk_factors": decision_data.get("risk_factors", []),
                    "raw_response": content
                }

            return self._fallback_parse(content)

        except Exception as e:
            logger.error(f"Error parsing LLM decision: {e}", exc_info=True)
            return None


    def _fallback_parse(self, content):
        """Fallback parsing for non-JSON responses"""
        signal = "HOLD"
        if "BUY" in content.upper():
            signal = "BUY"
        elif "SELL" in content.upper():
            signal = "SELL"

        return {
            "signal": signal,
            "confidence": 50,
            "entry_price": 0,
            "stop_loss": 0,
            "take_profit": 0,
            "position_size_pct": 0.02,
            "reasoning": content,
            "risk_factors": [],
            "raw_response": content
        }

class TradingDecisionService:
    """Orchestrate the complete trading decision process with memory integration"""
    
    def __init__(self):
        self.prompt_generator = LLMPromptGenerator()
        self.llm_service = DeepSeekLLMService()
        self.portfolio_sync = PortfolioSyncService()
        
        # Initialize Pinecone memory service
        try:
            from ai_engine.pinecone_service import PineconeMemoryService
            self.memory_service = PineconeMemoryService()
        except Exception as e:
            logger.warning(f"Pinecone service not available: {e}")
            self.memory_service = None
    
    def make_trading_decision(self, symbol="BTCUSD"):
        """Complete memory-enhanced trading decision pipeline"""
        try:
            logger.info(f"Making memory-enhanced trading decision for {symbol}")
            
            # 1. Generate comprehensive prompt (no auto-sync)
            prompt_data = self.prompt_generator.generate_trading_prompt(symbol) 
            
            if not prompt_data:
                logger.error("Failed to generate trading prompt")
                return None
            
            # 3. Get initial LLM decision
            logger.info("Getting initial LLM decision...")
            initial_llm_response = self.llm_service.get_trading_decision(prompt_data)
            if not initial_llm_response:
                logger.error("Failed to get initial LLM response")
                return None
        
            # 4. Parse initial decision
            initial_decision = self.llm_service.parse_trading_decision(initial_llm_response)
            if not initial_decision:
                logger.error("Failed to parse initial LLM decision")
                return None
            
            logger.info(f"Initial decision: {initial_decision['signal']} with {initial_decision['confidence']}% confidence")
            
            # 5. Check similarity in Pinecone memory (if available)
            similar_memories = []
            memory_summary = ""
            
            if self.memory_service and initial_decision['reasoning']:
                logger.info("Searching for similar past decisions...")
                similar_memories = self.memory_service.find_similar_decisions(
                    current_reasoning=initial_decision['reasoning'],
                    symbol=symbol,
                    market_context=prompt_data.get("market_data"),
                    risk_factors=initial_decision['risk_factors'],
                    top_k=50
                )
                memory_summary = self.memory_service.get_memory_summary(similar_memories)
                print("similar_memories", similar_memories)
                logger.info(f"Found {len(similar_memories)} similar past decisions")
            
            final_decision = initial_decision

            if len(similar_memories) > 0: 

                # 6. Create enhanced prompt with memory context
                enhanced_prompt_data = self._create_enhanced_prompt(
                    prompt_data, initial_decision, memory_summary
                )
                
                # 7. Get final memory-enhanced decision
                logger.info("Getting memory-enhanced final decision...")
                final_llm_response = self.llm_service.get_trading_decision(enhanced_prompt_data)
                if not final_llm_response:
                    logger.error("Failed to get enhanced LLM response")
                    # Fallback to initial decision
                    final_decision = initial_decision
                else:
                    final_decision = self.llm_service.parse_trading_decision(final_llm_response)
                    if not final_decision:
                        logger.warning("Failed to parse enhanced decision, using initial")
                        final_decision = initial_decision
                
            logger.info(f"Final decision: {final_decision['signal']} with {final_decision['confidence']}% confidence")
            ai_decision = None
            memory_id = None
            order_result = None
            if final_decision['confidence'] >= 70:
                # 8. Save decision to database
                ai_decision = self._save_decision(symbol, final_decision, prompt_data.get("market_data"))
                
                # 9. Store in memory for future reference
                if self.memory_service and ai_decision:
                    memory_id = self.memory_service.store_decision_memory(ai_decision)
                    # Store the Pinecone ID in the AI decision for later performance updates
                    if memory_id:
                        ai_decision.pinecone_id = memory_id
                        ai_decision.save()
                
                # 10. Execute trade if confidence > 80%
                if final_decision['signal'] in ['BUY', 'SELL']:
                    logger.info(f"High confidence ({final_decision['confidence']}%) - executing trade")
                    order_result = self._execute_high_confidence_trade(symbol, final_decision, ai_decision)
            else:
                logger.info(f"Confidence too low ({final_decision['confidence']}%)")

                
            return {
                "initial_decision": initial_decision,
                "final_decision": final_decision,
                "similar_memories": similar_memories,
                "memory_summary": memory_summary,
                "prompt_data": prompt_data["prompt"],
                "ai_decision_id": str(ai_decision.id) if ai_decision else None,
                "memory_id": memory_id,
                "order_result": order_result
            }
            
        except Exception as e:
            logger.error(f"Error in memory-enhanced trading decision process: {e}")
            return None
    
    def _create_enhanced_prompt(self, original_prompt_data, initial_decision, memory_summary):
        """Create enhanced prompt with memory context"""
        enhanced_prompt = f"""
        You are an expert **crypto swing trader** with access to your **trading memory** and **historical performance**.  
        You previously made an initial trading decision, and now you have access to similar past trades and their outcomes.  
        Use this context to refine your current position intelligently — adjusting your bias, confidence, and risk parameters if needed.

        ---
        ⚙️ Notes:
        - All price or signal data is ordered chronologically: OLDEST → NEWEST.
        - Unless stated otherwise, intraday data is sampled at 3-minute intervals.

        ### ORIGINAL MARKET DATA
        {json.dumps(original_prompt_data['market_data'], indent=2)}

        ### ORIGINAL PORTFOLIO DATA
        {json.dumps(original_prompt_data['portfolio_data'], indent=2)}

        ---

        ### YOUR INITIAL DECISION
        - Signal: {initial_decision['signal']}
        - Confidence: {initial_decision['confidence']}%
        - Entry Price: ${initial_decision.get('entry_price', 0)}
        - Stop Loss: ${initial_decision.get('stop_loss', 0)}
        - Take Profit: ${initial_decision.get('take_profit', 0)}
        - Reasoning: {initial_decision['reasoning']}
        - Risk factors: {initial_decision['risk_factors']}

        ---
        ### SIMILAR PAST DECISIONS (FROM MEMORY)
        {memory_summary}
        ---

        ### OBJECTIVE
        Re-evaluate your previous trade using the outcomes of these past similar decisions.
        Incorporate *pattern recognition*, *risk bias adjustments*, and *historical success rates* to refine your new decision.

        Your analysis **must** answer:
        1. Do you maintain the same trading signal or change it (BUY/SELL/HOLD)?
        2. Should your confidence level be adjusted based on past trade outcomes?
        3. Are there common failure or success patterns in similar setups that influence this case?
        4. Should entry, stop loss, or take profit be fine-tuned?
        5. What specific lessons from prior trades directly apply here?

        ---
        ### POSITION & RISK RULES
        To ensure consistent profitability over time, strictly follow these constraints:
        - The profit target must be at least +1% from the entry price
        - Maintain a **minimum Reward-to-Risk (RRR)** of **at least 1.5:1** (e.g. aiming for +1.5% gain if risking −1%).
        - The stop-loss and take-profit levels should be chosen such that over many trades, the strategy remains **positively skewed** (expected value > 0).
        - Favor setups where technical confluence and higher-timeframe trend agree with direction.

        ---
        ### OUTPUT FORMAT (STRICT JSON ONLY)
        Respond **ONLY** in valid JSON — no markdown, commentary, or text outside JSON.  
        All numeric values must be floats or integers (no strings).  

        Expected schema:

        {{
            "trading_signal": "BUY" | "SELL" | "HOLD",
            "confidence_level": float,          # 0–100, updated after memory review
            "entry_price": float,               # refined entry price
            "stop_loss": float,                 # adjusted stop loss
            "take_profit": [float, float],      # one or multiple target levels
            "position_size": int,               # position size as % of portfolio (integer)
            "reasoning": str,                   # reasoning combining memory insights (≤500 chars)
            "risk_factors": [str],              # key insights from similar past decisions
        }}

        ---
        ### EXECUTION RULES
        - Adjust `entry_price`, `stop_loss`, and `take_profit` if your refined logic suggests better risk/reward.
        - Incorporate lessons such as *early reversal detection*, *trend continuation bias*, or *mean reversion traps* into reasoning.
        - Be concise but insightful — your reasoning is crucial for audit and explainability.
        ---

        ### CONTEXTUAL NOTE
        Current UTC time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}.
        Your goal is to **refine the previous decision** using experience from similar trades and improve long-term performance consistency.
        """

        return {
            "prompt": enhanced_prompt,
            "market_data": original_prompt_data['market_data'],
            "portfolio_data": original_prompt_data['portfolio_data']
        }
    
    def _execute_high_confidence_trade(self, symbol, decision, ai_decision):
        """Execute trade for high confidence decisions (>80%)"""
        try:
            logger.info(f"Executing high confidence trade: {decision['signal']} {symbol}")
            
            from exchanges.services import DeltaExchangeClient
            client = DeltaExchangeClient()
            
            # Get current portfolio to calculate position size
            portfolio = Portfolio.get_primary_portfolio()
            if not portfolio:
                logger.error("No primary portfolio found")
                return {"success": False, "error": "No portfolio"}
            
            leverage = 10 #decision.get('leverage', 10)
            # Calculate position size in USD with validation
            available_balance = float(portfolio.available_balance_usd)
            
            existing_position = Position.get_position_by_symbol(symbol)

            if existing_position:
                existing_side = existing_position.side.lower()
                new_side = decision['signal'].lower()

                if existing_side != new_side:
                    client.close_position(existing_position)
                    logger.info(f"Opposite direction detected — closing {existing_side.upper()} position before {new_side.upper()} entry")
                    exit_price = float(decision.get('entry_price', existing_position.current_price))
                    existing_position.close_position(exit_price)                            
                    available_balance += float(existing_position.current_value_usd)
                else:
                    logger.info(f"Same direction detected")

                    return {
                    "success": False,
                    "is_postion_opened": False,
                    "order_result": None,
                    "message": "Position exist on this side, quantity add is not yet implemented"
                    }
                    

            # Ensure we have minimum balance
            if available_balance < 10:  # Minimum $10 available
                logger.warning(f"Insufficient balance: ${available_balance}")
                return {"success": False, "error": f"Insufficient balance: ${available_balance}"}
            
            # Use available balance for position size
            position_size_usd = available_balance
            notional_value_usd = position_size_usd * leverage
            
            # Calculate quantity based on entry price
            entry_price = decision.get('entry_price', 0)
            if entry_price <= 0:
                logger.warning("No valid entry price provided")
                return {"success": False, "error": "No entry price"}
            
            btc_quantity = notional_value_usd / entry_price
            # Convert BTC quantity to lot size (1 lot = 0.001 BTC)
            lot_size = 0.001
            quantity_in_lots = btc_quantity / lot_size

            # Round to nearest whole lot (or exchange precision)
            quantity_in_lots = int(quantity_in_lots)
            if quantity_in_lots <= 0:
                logger.warning(f"Quantity too small: {quantity_in_lots} lots")
                return {"success": False, "error": "Quantity too small"}

            # Calculate required margin for validation
            required_margin = (quantity_in_lots * lot_size * entry_price) / leverage
            if required_margin > available_balance:
                logger.warning(f"Required margin ${required_margin} exceeds available balance ${available_balance}")
                return {"success": False, "error": f"Insufficient margin: need ${required_margin}"}

            # Place the order
            order_result = client.place_order(quantity_in_lots, ai_decision)
    
            if order_result:
      
                # Create trade record
                # trade_record = self._create_trade_record(
                #     symbol, decision, quantity_in_lots, entry_price, ai_decision
                # )
                
                # Create/Update position in portfolio
                position_result = self._create_or_update_position(
                    symbol, decision, entry_price, ai_decision, order_result
                )
            
                # Sync portfolio after trade execution
                portfolio_sync_result = self._sync_portfolio_after_trade()
                
                # Mark AI decision as executed
                if ai_decision:
                    ai_decision.mark_executed()
                
                logger.info(f"Trade executed successfully: {order_result}")
                logger.info(f"Position created/updated: {position_result}")
                logger.info(f"Portfolio synced: {portfolio_sync_result}")
                
                return {
                    "success": True,
                    "order_result": order_result,
                    # "trade_record_id": str(trade_record.id) if trade_record else None,
                    "position_result": position_result,
                    "portfolio_sync": portfolio_sync_result,
                    "quantity": quantity_in_lots,
                    "position_size_usd": position_size_usd
                }
            else:
                logger.error("Order placement failed")
                return {"success": False, "error": "Order placement failed"}
                
        except Exception as e:
            logger.error(f"Error executing high confidence trade: {e}")
            return {"success": False, "error": str(e)}
    
    def _create_trade_record(self, symbol, decision, quantity, entry_price, ai_decision):
        """Create trade record in database"""
        try:
            from trading.models import Trade, TradingStrategy
            
            user = TradingUser.get_primary_trader()
            strategy = TradingStrategy.get_active_strategy()
            
            if not user:
                logger.warning("No primary user found")
                return None
            
            trade = Trade(
                trade_id=f"trade_{int(datetime.now(timezone.utc).timestamp())}",
                user=user,
                strategy=strategy,
                symbol=symbol,
                side=decision['signal'].lower(),
                quantity=Decimal(str(quantity)),
                entry_price=Decimal(str(entry_price)),
                stop_loss=Decimal(str(decision.get('stop_loss', 0))),
                take_profit=Decimal(str(decision.get('take_profit', 0))),
                exchange='delta',
                status='pending'
            )
            
            trade.save()
            logger.info(f"Trade record created: {trade.trade_id}")
            return trade
            
        except Exception as e:
            logger.error(f"Error creating trade record: {e}")
            return None

    def _create_or_update_position(self, symbol, decision, entry_price, ai_decision, order_result):
        """Create or update position in portfolio"""
        try:
            from portfolio.models import Portfolio
            
            portfolio = Portfolio.get_primary_portfolio()
            if not portfolio:
                logger.error("No primary portfolio found")
                return {"success": False, "error": "No portfolio found"}
            
            # Convert lot size back to BTC quantity
            # lot_size = 0.001
            # btc_quantity = Decimal(str(quantity_in_lots * lot_size))
            entry_price_decimal = Decimal(str(entry_price))
            
            new_position = self._create_new_position(
                    portfolio, symbol, decision, entry_price_decimal, order_result, ai_decision
                )
            logger.info(f"Created new position: {new_position}")
            return {"success": True, "action": "created_new", "position_id": str(new_position.id)}
            
        except Exception as e:
            logger.error(f"Error creating/updating position: {e}")
            return {"success": False, "error": str(e)}
    
    def _create_new_position(self, portfolio, symbol, decision, entry_price, order_result, ai_decision):
        """Create a new position in the portfolio"""
        from portfolio.models import Position
        
        position = Position(
            portfolio=portfolio,
            ai_decision=ai_decision,
            symbol=symbol,
            exchange='delta',
            product_id=str(order_result.get('product_id')),  # BTCUSD product ID
            order_id=str(order_result.get('id')),  # Store the Delta Exchange order ID
            side=decision['signal'].lower(),
            quantity=order_result.get('size'),
            entry_price=entry_price,
            current_price=entry_price,
            stop_loss=Decimal(str(decision.get('stop_loss', 0))),
            take_profit=Decimal(str(decision.get('take_profit', 0))),
        )
        
        position.save()
        
        return position
    
    def _sync_portfolio_after_trade(self):
        """Sync portfolio balances and positions after trade execution"""
        try:
            # Sync balances and positions from exchange
            self.portfolio_sync.sync_balances()
            self.portfolio_sync.sync_positions()
            
            # Update portfolio metrics
            portfolio = Portfolio.get_primary_portfolio()
            if portfolio:
                portfolio.update_metrics()
                logger.info(f"Portfolio metrics updated - Total Value: ${portfolio.total_value_usd}")
                return {
                    "success": True,
                    "total_value": float(portfolio.total_value_usd),
                    "available_balance": float(portfolio.available_balance_usd),
                    "unrealized_pnl": float(portfolio.unrealized_pnl_usd),
                    "positions_count": portfolio.current_positions_count
                }
            else:
                return {"success": False, "error": "No portfolio found"}
                
        except Exception as e:
            logger.error(f"Error syncing portfolio after trade: {e}")
            return {"success": False, "error": str(e)}

    def update_ai_decision_performance(self, ai_decision, exit_price, realized_pnl_usd):
        """Update AI decision performance in both local DB and Pinecone memory"""
        try:
            if not ai_decision:
                logger.warning("No AI decision provided for performance update")
                return False

            # Calculate performance metrics
            entry_price = ai_decision.recommended_entry
            if not entry_price or entry_price == 0:
                logger.warning("No entry price available for performance calculation")
                return False

            # Calculate price change percentage
            price_change_pct = (exit_price - entry_price) / entry_price
            
            # Determine if profitable based on decision type
            if ai_decision.decision_type == 'buy':
                is_profitable = price_change_pct > 0
            elif ai_decision.decision_type == 'sell':
                is_profitable = price_change_pct < 0
            else:
                is_profitable = realized_pnl_usd > 0

            # Update local AI decision
            ai_decision.was_profitable = is_profitable
            ai_decision.actual_outcome = Decimal(str(price_change_pct))
            ai_decision.save()

            # Update model performance
            if ai_decision.model:
                ai_decision.model.update_performance(is_profitable)

            # Update Pinecone memory if available
            if self.memory_service and ai_decision.pinecone_id:
                try:
                    self.memory_service.update_decision_performance(
                        pinecone_id=ai_decision.pinecone_id,
                        was_profitable=is_profitable,
                        actual_pnl=float(realized_pnl_usd),
                        price_change_pct=float(price_change_pct),
                        exit_price=float(exit_price)
                    )
                    logger.info(f"Updated Pinecone memory {ai_decision.pinecone_id} with performance data")
                except Exception as pinecone_error:
                    logger.error(f"Failed to update Pinecone memory: {pinecone_error}")

            logger.info(f"Updated AI decision performance: Profitable={is_profitable}, PnL=${realized_pnl_usd}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating AI decision performance: {e}")
            return False

    def _save_decision(self, symbol, decision, market_data):
        """Save AI decision to database"""

        try:
            ai_model = AIModel.get_active_model('deepseek')
            if not ai_model:
                logger.warning("No active DeepSeek model found")
                return None
            
            ai_decision = AIDecision(
                decision_id=f"decision_{int(datetime.now(timezone.utc).timestamp())}",
                model=ai_model,
                symbol=symbol,
                decision_type=decision['signal'].lower(),
                confidence_score=Decimal(str(decision['confidence'] / 100)),
                reasoning=decision['reasoning'],
                risk_factors=decision.get('risk_factors', []),
                market_context=market_data,
                price_at_decision=Decimal(str(market_data.get('current', {}).get('price', 0))),
                recommended_entry=Decimal(str(decision.get('entry_price', 0))),
                recommended_stop_loss=Decimal(str(decision.get('stop_loss', 0))),
                recommended_take_profit=Decimal(str(decision.get('take_profit', 0))),
                position_size_pct=Decimal(str(decision.get('position_size_pct', 0)))/ Decimal("100"),
                expires_at=datetime.now(timezone.utc) + timedelta(minutes=15)  # 15-minute expiry
            )
            
            ai_decision.save()
            return ai_decision
            
        except Exception as e:
            logger.error(f"Error saving AI decision: {e}")
            return None
