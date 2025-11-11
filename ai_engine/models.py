from mongoengine import Document, EmbeddedDocument, fields
from datetime import datetime, timedelta
from decimal import Decimal
from accounts.models import TradingUser


class AIModel(Document):
    """AI model configurations and versions"""
    model_name = fields.StringField(max_length=100, required=True)
    model_version = fields.StringField(max_length=50, required=True)
    model_type = fields.StringField(
        max_length=30,
        choices=[
            ('deepseek', 'DeepSeek LLM'),
            ('custom', 'Custom Model'),
            ('ensemble', 'Ensemble Model')
        ],
        required=True
    )
    
    # Model configuration
    api_endpoint = fields.StringField(max_length=200)
    model_parameters = fields.DictField()
    context_window = fields.IntField(default=4000)
    
    # Performance tracking
    is_active = fields.BooleanField(default=True)
    accuracy_score = fields.DecimalField(min_value=0, max_value=1, precision=4)
    total_predictions = fields.IntField(default=0)
    correct_predictions = fields.IntField(default=0)
    
    created_at = fields.DateTimeField(default=datetime.utcnow)
    updated_at = fields.DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'ai_models',
        'indexes': ['model_type', 'is_active', 'model_name']
    }

    @classmethod
    def get_active_model(cls, model_type='deepseek'):
        """Get the active AI model"""
        return cls.objects(model_type=model_type, is_active=True).first()

    def update_performance(self, was_correct):
        """Update model performance metrics"""
        self.total_predictions += 1
        if was_correct:
            self.correct_predictions += 1
        
        if self.total_predictions > 0:
            self.accuracy_score = Decimal(self.correct_predictions / self.total_predictions)
        
        self.updated_at = datetime.utcnow()
        self.save()

    def __str__(self):
        return f"{self.model_name} v{self.model_version}"


class AIDecision(Document):
    """AI-generated trading decisions and analysis"""
    decision_id = fields.StringField(unique=True, required=True)
    model = fields.ReferenceField(AIModel, required=True)
    symbol = fields.StringField(max_length=20, required=True)
    
    # Decision details
    decision_type = fields.StringField(
        max_length=20,
        choices=[
            ('buy', 'Buy Signal'),
            ('sell', 'Sell Signal'),
            ('hold', 'Hold Position'),
            ('close', 'Close Position'),
            ('risk_alert', 'Risk Alert')
        ],
        required=True
    )
    
    # AI confidence and reasoning
    confidence_score = fields.DecimalField(min_value=0, max_value=1, precision=4, required=True)
    reasoning = fields.StringField(max_length=3000)
    risk_factors = fields.ListField(fields.StringField(max_length=200))
    
    # Market context used in decision
    market_context = fields.DictField()  # Technical indicators, sentiment, etc.
    price_at_decision = fields.DecimalField(min_value=0, precision=8)
    
    # Recommended actions
    recommended_entry = fields.DecimalField(min_value=0, precision=8)
    recommended_stop_loss = fields.DecimalField(min_value=0, precision=8)
    recommended_take_profit = fields.DecimalField(min_value=0, precision=8)
    position_size_pct = fields.DecimalField(min_value=0, max_value=1, precision=4)
    
    # Status tracking
    status = fields.StringField(
        max_length=20,
        choices=[
            ('pending', 'Pending'),
            ('executed', 'Executed'),
            ('ignored', 'Ignored'),
            ('expired', 'Expired')
        ],
        default='pending'
    )
    
    # Pinecone memory tracking
    pinecone_id = fields.StringField(max_length=100)  # ID in Pinecone memory for performance updates
    
    # Outcome tracking
    was_profitable = fields.BooleanField()
    actual_outcome = fields.DecimalField(precision=4)  # Actual price change percentage
    
    created_at = fields.DateTimeField(default=datetime.utcnow)
    expires_at = fields.DateTimeField()

    meta = {
        'collection': 'ai_decisions',
        'indexes': [
            'symbol', 'decision_type', 'status', 'created_at',
            ('symbol', 'status'), 'model', 'confidence_score'
        ]
    }

    @classmethod
    def get_recent_decisions(cls, symbol=None, limit=20):
        """Get recent AI decisions"""
        query = cls.objects
        if symbol:
            query = query.filter(symbol=symbol)
        return query.order_by('-created_at').limit(limit)

    @classmethod
    def get_pending_decisions(cls):
        """Get pending decisions that haven't been executed"""
        return cls.objects(
            status='pending',
            expires_at__gte=datetime.utcnow()
        ).order_by('-confidence_score')

    def mark_executed(self):
        """Mark decision as executed"""
        self.status = 'executed'
        self.save()

    def evaluate_outcome(self, current_price, realized_pnl_usd=None):
        """Evaluate the outcome of the decision and update both local DB and Pinecone memory"""
        if self.recommended_entry and current_price:
            price_change_pct = (current_price - self.recommended_entry) / self.recommended_entry
            
            if self.decision_type == 'buy':
                self.was_profitable = price_change_pct > 0
            elif self.decision_type == 'sell':
                self.was_profitable = price_change_pct < 0
            else:
                # For other decision types, use realized PnL if available
                self.was_profitable = realized_pnl_usd > 0 if realized_pnl_usd is not None else price_change_pct > 0
            
            self.actual_outcome = Decimal(str(price_change_pct))
            self.save()
            
            # Update model performance
            if self.model:
                self.model.update_performance(self.was_profitable)
            
            # Update Pinecone memory if available
            if self.pinecone_id:
                try:
                    from ai_engine.pinecone_service import PineconeMemoryService
                    memory_service = PineconeMemoryService()
                    
                    memory_service.update_decision_performance(
                        pinecone_id=self.pinecone_id,
                        was_profitable=self.was_profitable,
                        actual_pnl=float(realized_pnl_usd) if realized_pnl_usd is not None else 0.0,
                        price_change_pct=float(price_change_pct),
                        exit_price=float(current_price)
                    )
                    
                    from django.utils import timezone
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"Updated Pinecone memory {self.pinecone_id} with performance data")
                    
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Failed to update Pinecone memory for decision {self.decision_id}: {e}")

    def __str__(self):
        return f"AI {self.decision_type.upper()} {self.symbol} (Conf: {self.confidence_score})"


class ConversationHistory(Document):
    """Conversation history with DeepSeek for context"""
    session_id = fields.StringField(max_length=100, required=True)
    user = fields.ReferenceField(TradingUser)
    
    # Message details
    role = fields.StringField(
        max_length=20,
        choices=[('user', 'User'), ('assistant', 'Assistant'), ('system', 'System')],
        required=True
    )
    message = fields.StringField(required=True)
    
    # Context
    trading_context = fields.DictField()  # Current market conditions, portfolio state, etc.
    token_count = fields.IntField(default=0)
    
    timestamp = fields.DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'conversation_history',
        'indexes': ['session_id', 'timestamp', 'user']
    }

    @classmethod
    def get_recent_conversation(cls, session_id, limit=10):
        """Get recent conversation for context"""
        return cls.objects(session_id=session_id).order_by('-timestamp').limit(limit)

    @classmethod
    def cleanup_old_conversations(cls, days_old=7):
        """Clean up old conversations to manage storage"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        cls.objects(timestamp__lt=cutoff_date).delete()

    def __str__(self):
        return f"{self.role}: {self.message[:50]}..."


class LongTermMemory(Document):
    """Long-term memory storage using Pinecone for AI context"""
    memory_id = fields.StringField(unique=True, required=True)
    memory_type = fields.StringField(
        max_length=30,
        choices=[
            ('market_pattern', 'Market Pattern'),
            ('trading_rule', 'Trading Rule'),
            ('successful_trade', 'Successful Trade'),
            ('failed_trade', 'Failed Trade'),
            ('market_event', 'Market Event'),
            ('user_preference', 'User Preference')
        ],
        required=True
    )
    
    # Memory content
    title = fields.StringField(max_length=200, required=True)
    description = fields.StringField(max_length=1000)
    content = fields.StringField(required=True)
    
    # Embeddings and search
    pinecone_vector_id = fields.StringField(max_length=100)  # ID in Pinecone
    embedding_model = fields.StringField(max_length=50, default='text-embedding-ada-002')
    
    # Context and metadata
    symbols_related = fields.ListField(fields.StringField(max_length=20))
    time_period = fields.DateTimeField()
    importance_score = fields.DecimalField(min_value=0, max_value=1, precision=4)
    
    # Usage tracking
    access_count = fields.IntField(default=0)
    last_accessed = fields.DateTimeField()
    
    created_at = fields.DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'long_term_memory',
        'indexes': [
            'memory_type', 'symbols_related', 'importance_score',
            'last_accessed', 'created_at'
        ]
    }

    @classmethod
    def get_relevant_memories(cls, memory_type=None, symbols=None, limit=5):
        """Get relevant memories based on context"""
        query = cls.objects
        
        if memory_type:
            query = query.filter(memory_type=memory_type)
        
        if symbols:
            query = query.filter(symbols_related__in=symbols)
        
        return query.order_by('-importance_score', '-last_accessed').limit(limit)

    def mark_accessed(self):
        """Mark memory as accessed"""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()
        self.save()

    def __str__(self):
        return f"{self.memory_type}: {self.title}"


class AIPerformanceMetrics(Document):
    """Track AI performance over time"""
    date = fields.DateTimeField(required=True)
    model = fields.ReferenceField(AIModel, required=True)
    
    # Daily metrics
    total_decisions = fields.IntField(default=0)
    executed_decisions = fields.IntField(default=0)
    profitable_decisions = fields.IntField(default=0)
    
    # Performance scores
    accuracy_rate = fields.DecimalField(min_value=0, max_value=1, precision=4)
    precision_score = fields.DecimalField(min_value=0, max_value=1, precision=4)
    recall_score = fields.DecimalField(min_value=0, max_value=1, precision=4)
    f1_score = fields.DecimalField(min_value=0, max_value=1, precision=4)
    
    # Financial performance
    avg_return_per_decision = fields.DecimalField(precision=6)
    total_pnl_generated = fields.DecimalField(precision=8)
    sharpe_ratio = fields.DecimalField(precision=4)
    
    # Risk metrics
    max_drawdown = fields.DecimalField(precision=4)
    volatility = fields.DecimalField(precision=4)
    
    created_at = fields.DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'ai_performance_metrics',
        'indexes': ['date', 'model', ('model', 'date')]
    }

    @classmethod
    def get_performance_trend(cls, model, days=30):
        """Get performance trend for a model"""
        start_date = datetime.utcnow() - timedelta(days=days)
        return cls.objects(
            model=model,
            date__gte=start_date
        ).order_by('date')

    def __str__(self):
        return f"AI Performance {self.date.date()} - Accuracy: {self.accuracy_rate}"
