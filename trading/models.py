from mongoengine import Document, EmbeddedDocument, fields
from datetime import datetime
from decimal import Decimal
from accounts.models import TradingUser


class TradingStrategy(Document):
    """AI-driven trading strategies"""
    name = fields.StringField(max_length=100, required=True)
    description = fields.StringField(max_length=500)
    strategy_type = fields.StringField(
        max_length=30,
        choices=[
            ('swing', 'Swing Trading'),
            ('scalping', 'Scalping'),
            ('momentum', 'Momentum'),
            ('mean_reversion', 'Mean Reversion')
        ],
        default='swing'
    )
    is_active = fields.BooleanField(default=True)
    ai_model_version = fields.StringField(max_length=50)
    success_rate = fields.DecimalField(min_value=0, max_value=1, precision=4)
    parameters = fields.DictField()  # Strategy-specific parameters
    created_at = fields.DateTimeField(default=datetime.utcnow)
    updated_at = fields.DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'trading_strategies',
        'indexes': ['strategy_type', 'is_active']
    }

    @classmethod
    def get_active_strategy(cls):
        """Get the currently active strategy for primary user"""
        return cls.objects(is_active=True).first()

    def __str__(self):
        return f"Strategy: {self.name}"


class Trade(Document):
    """Individual trade records"""
    trade_id = fields.StringField(unique=True, required=True)
    user = fields.ReferenceField(TradingUser, required=True)
    strategy = fields.ReferenceField(TradingStrategy)
    
    # Trade basics
    symbol = fields.StringField(max_length=20, required=True)  # e.g., 'BTCUSD'
    side = fields.StringField(
        max_length=10,
        choices=[('buy', 'Buy'), ('sell', 'Sell')],
        required=True
    )
    trade_type = fields.StringField(
        max_length=20,
        choices=[
            ('market', 'Market'),
            ('limit', 'Limit'),
            ('stop_market', 'Stop Market'),
            ('stop_limit', 'Stop Limit')
        ],
        default='market'
    )
    
    # Quantities and prices
    quantity = fields.DecimalField(min_value=0, precision=8, required=True)
    entry_price = fields.DecimalField(min_value=0, precision=8)
    exit_price = fields.DecimalField(min_value=0, precision=8)
    stop_loss = fields.DecimalField(min_value=0, precision=8)
    take_profit = fields.DecimalField(min_value=0, precision=8)
    
    # Status and timing
    status = fields.StringField(
        max_length=20,
        choices=[
            ('pending', 'Pending'),
            ('open', 'Open'),
            ('filled', 'Filled'),
            ('partial', 'Partial'),
            ('cancelled', 'Cancelled'),
            ('closed', 'Closed')
        ],
        default='pending'
    )
    
    # Financial metrics
    pnl_usd = fields.DecimalField(default=Decimal('0'), precision=8)
    fee_usd = fields.DecimalField(default=Decimal('0'), precision=8)
    
    # AI decision context
    ai_confidence = fields.DecimalField(min_value=0, max_value=1, precision=4)
    ai_reasoning = fields.StringField(max_length=1000)
    market_conditions = fields.DictField()  # Technical indicators at time of trade
    
    # Exchange info
    exchange = fields.StringField(max_length=50, required=True)
    exchange_order_id = fields.StringField(max_length=100)
    
    # Timestamps
    created_at = fields.DateTimeField(default=datetime.utcnow)
    filled_at = fields.DateTimeField()
    closed_at = fields.DateTimeField()

    meta = {
        'collection': 'trades',
        'indexes': [
            'user', 'symbol', 'status', 'created_at',
            ('user', 'status'), ('symbol', 'created_at')
        ]
    }

    @classmethod
    def get_open_trades(cls):
        """Get all open trades for primary user"""
        primary_user = TradingUser.get_primary_trader()
        if primary_user:
            return cls.objects(user=primary_user, status__in=['open', 'partial'])
        return cls.objects.none()

    @classmethod
    def get_recent_trades(cls, limit=50):
        """Get recent trades for primary user"""
        primary_user = TradingUser.get_primary_trader()
        if primary_user:
            return cls.objects(user=primary_user).order_by('-created_at').limit(limit)
        return cls.objects.none()

    def calculate_pnl(self):
        """Calculate P&L for the trade"""
        if self.entry_price and self.exit_price:
            price_diff = self.exit_price - self.entry_price
            if self.side == 'sell':  # Short position
                price_diff = -price_diff
            self.pnl_usd = price_diff * self.quantity - self.fee_usd
            return self.pnl_usd
        return Decimal('0')

    def __str__(self):
        return f"{self.side.upper()} {self.quantity} {self.symbol} @ {self.entry_price}"


class TradingSignal(Document):
    """AI-generated trading signals"""
    signal_id = fields.StringField(unique=True, required=True)
    symbol = fields.StringField(max_length=20, required=True)
    
    # Signal details
    signal_type = fields.StringField(
        max_length=10,
        choices=[('buy', 'Buy'), ('sell', 'Sell'), ('hold', 'Hold')],
        required=True
    )
    strength = fields.DecimalField(min_value=0, max_value=1, precision=4)  # 0-1 confidence
    suggested_entry = fields.DecimalField(min_value=0, precision=8)
    suggested_stop_loss = fields.DecimalField(min_value=0, precision=8)
    suggested_take_profit = fields.DecimalField(min_value=0, precision=8)
    
    # AI analysis
    ai_reasoning = fields.StringField(max_length=2000)
    technical_indicators = fields.DictField()
    market_sentiment = fields.StringField(max_length=50)
    
    # Status
    status = fields.StringField(
        max_length=20,
        choices=[
            ('active', 'Active'),
            ('executed', 'Executed'),
            ('expired', 'Expired'),
            ('ignored', 'Ignored')
        ],
        default='active'
    )
    
    # References
    executed_trade = fields.ReferenceField(Trade)
    created_at = fields.DateTimeField(default=datetime.utcnow)
    expires_at = fields.DateTimeField()

    meta = {
        'collection': 'trading_signals',
        'indexes': ['symbol', 'status', 'created_at', 'strength']
    }

    @classmethod
    def get_active_signals(cls):
        """Get currently active signals"""
        return cls.objects(status='active', expires_at__gte=datetime.utcnow())

    def __str__(self):
        return f"{self.signal_type.upper()} {self.symbol} (Strength: {self.strength})"


class RiskManagement(Document):
    """Risk management settings and tracking - single user focused"""
    user = fields.ReferenceField(TradingUser, required=True)
    
    # Current risk metrics
    current_exposure_usd = fields.DecimalField(default=Decimal('0'), precision=8)
    daily_pnl_usd = fields.DecimalField(default=Decimal('0'), precision=8)
    weekly_pnl_usd = fields.DecimalField(default=Decimal('0'), precision=8)
    
    # Risk limits
    max_position_size_usd = fields.DecimalField(min_value=0, precision=8)
    daily_loss_limit_usd = fields.DecimalField(min_value=0, precision=8)
    
    # Risk flags
    is_trading_halted = fields.BooleanField(default=False)
    halt_reason = fields.StringField(max_length=200)
    
    # Timestamps
    date = fields.DateTimeField(default=datetime.utcnow)
    updated_at = fields.DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'risk_management',
        'indexes': ['user', 'date']
    }

    @classmethod
    def get_current_risk_status(cls):
        """Get current risk status for primary user"""
        primary_user = TradingUser.get_primary_trader()
        if primary_user:
            today = datetime.utcnow().date()
            return cls.objects(
                user=primary_user,
                date__gte=datetime.combine(today, datetime.min.time())
            ).first()
        return None

    def check_risk_limits(self):
        """Check if any risk limits are breached"""
        breaches = []
        
        if self.daily_loss_limit_usd and abs(self.daily_pnl_usd) > self.daily_loss_limit_usd:
            breaches.append('daily_loss_limit')
        
        if self.max_position_size_usd and self.current_exposure_usd > self.max_position_size_usd:
            breaches.append('position_size_limit')
        
        return breaches

    def __str__(self):
        return f"Risk Status - Exposure: ${self.current_exposure_usd}, Daily P&L: ${self.daily_pnl_usd}"
