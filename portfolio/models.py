from mongoengine import Document, EmbeddedDocument, fields
from datetime import datetime, timedelta
from decimal import Decimal
from accounts.models import TradingUser
from ai_engine.models import AIDecision

class Portfolio(Document):
    """Main portfolio for the user - single user focused but scalable"""
    user = fields.ReferenceField(TradingUser, required=True)
    portfolio_name = fields.StringField(max_length=100, default='Main Portfolio')
    
    # Portfolio metrics
    total_value_usd = fields.DecimalField(min_value=0, precision=8, default=Decimal('0'))
    available_balance_usd = fields.DecimalField(min_value=0, precision=8, default=Decimal('0'))
    invested_balance_usd = fields.DecimalField(min_value=0, precision=8, default=Decimal('0'))
    unrealized_pnl_usd = fields.DecimalField(precision=8, default=Decimal('0'))
    realized_pnl_usd = fields.DecimalField(precision=8, default=Decimal('0'))
    
    # Portfolio performance
    total_return_pct = fields.DecimalField(precision=4, default=Decimal('0'))
    daily_return_pct = fields.DecimalField(precision=4, default=Decimal('0'))
    max_drawdown_pct = fields.DecimalField(precision=4, default=Decimal('0'))
    
    # Risk metrics
    current_positions_count = fields.IntField(default=0)
    total_exposure_usd = fields.DecimalField(min_value=0, precision=8, default=Decimal('0'))
    
    # Status
    is_active = fields.BooleanField(default=True)
    is_primary = fields.BooleanField(default=True)  # For single user setup
    
    created_at = fields.DateTimeField(default=datetime.utcnow)
    updated_at = fields.DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'portfolios',
        'indexes': ['user', 'is_primary', 'is_active']
    }

    @classmethod
    def get_primary_portfolio(cls):
        """Get the primary portfolio for single-user setup"""
        primary_user = TradingUser.get_primary_trader()
        if primary_user:
            return cls.objects(user=primary_user, is_primary=True, is_active=True).first()
        return None

    def update_metrics(self):
        """Update portfolio metrics based on current positions and balances"""
        # This would be called after trades or balance updates
        positions = Position.objects(portfolio=self, is_open=True)
        
        self.current_positions_count = positions.count()
        self.total_exposure_usd = sum(pos.current_value_usd for pos in positions)
        self.unrealized_pnl_usd = sum(pos.unrealized_pnl_usd for pos in positions)
        self.total_value_usd = self.available_balance_usd + self.invested_balance_usd + self.unrealized_pnl_usd
        
        self.updated_at = datetime.utcnow()
        self.save()

    def __str__(self):
        return f"{self.portfolio_name} - ${self.total_value_usd}"


class Position(Document):
    """Individual positions in the portfolio"""
    portfolio = fields.ReferenceField(Portfolio, required=True)
    ai_decision = fields.ReferenceField(AIDecision, required=True)
    symbol = fields.StringField(max_length=20, required=True)
    exchange = fields.StringField(max_length=50, required=True)
    product_id = fields.StringField(max_length=50, required=True)
    order_id = fields.StringField(max_length=50)  # Delta Exchange order ID for tracking
    
    # Position details
    side = fields.StringField(
        max_length=10,
        choices=[('buy', 'Buy'), ('sell', 'Sell')],
        required=True
    )
    quantity = fields.DecimalField(min_value=0, precision=8, required=True)
    entry_price = fields.DecimalField(min_value=0, precision=8, required=True)
    current_price = fields.DecimalField(min_value=0, precision=8, required=True)
    
    # Financial metrics
    entry_value_usd = fields.DecimalField(min_value=0, precision=8, default=Decimal('0'))
    current_value_usd = fields.DecimalField(min_value=0, precision=8, default=Decimal('0'))
    unrealized_pnl_usd = fields.DecimalField(precision=8, default=Decimal('0'))
    realized_pnl_usd = fields.DecimalField(precision=8, default=Decimal('0'))
    
    # Risk management
    stop_loss = fields.DecimalField(min_value=0, precision=8)
    take_profit = fields.DecimalField(min_value=0, precision=8)
    
    # Status
    is_open = fields.BooleanField(default=True)
    
    # Timestamps
    opened_at = fields.DateTimeField(default=datetime.utcnow)
    closed_at = fields.DateTimeField()
    updated_at = fields.DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'positions',
        'indexes': [
            'portfolio', 'symbol', 'is_open', 'exchange',
            ('portfolio', 'is_open'), ('symbol', 'is_open')
        ]
    }

    @classmethod
    def get_open_positions(cls):
        """Get all open positions for primary portfolio"""
        primary_portfolio = Portfolio.get_primary_portfolio()
        if primary_portfolio:
            return cls.objects(portfolio=primary_portfolio, is_open=True)
        return cls.objects.none()

    @classmethod
    def get_position_by_symbol(cls, symbol):
        """Get open position for a specific symbol"""
        primary_portfolio = Portfolio.get_primary_portfolio()
        if primary_portfolio:
            return cls.objects(
                portfolio=primary_portfolio, 
                symbol=symbol, 
                is_open=True
            ).first()
        return None

    def update_current_price(self, new_price):
        """Update current price and recalculate P&L"""
        self.current_price = new_price
        self.current_value_usd = self.quantity * new_price
        
        # Calculate unrealized P&L
        price_diff = new_price - self.entry_price
        if self.side == 'sell':
            price_diff = -price_diff
        
        self.unrealized_pnl_usd = price_diff * self.quantity
        self.updated_at = datetime.utcnow()
        self.save()

    def close_position(self, exit_price, exit_quantity=None):
        """Close the position (partially or fully)"""
        if exit_quantity is None:
            exit_quantity = self.quantity
            
        contract_value = Decimal(str(0.001)) 
        # Calculate realized P&L
        price_diff = Decimal(str(exit_price)) - self.entry_price
        if self.side == 'sell':
            price_diff = -price_diff
        
        # contract_value = Decimal(0.001)
        # realized_pnl_usd = price_diff * contract_value * exit_quantity
        self.realized_pnl_usd = self.unrealized_pnl_usd
        self.unrealized_pnl_usd = 0
        
        # Calculate margin to release (10x leverage)
        margin_to_release = self.current_value_usd #closed_notional_value / leverage
        
        # Update portfolio balances safely to prevent negative values
        self.portfolio.invested_balance_usd = max(Decimal('0'), self.portfolio.invested_balance_usd - margin_to_release)
        self.portfolio.available_balance_usd = max(Decimal('0'), self.portfolio.available_balance_usd + margin_to_release)
        self.portfolio.realized_pnl_usd += self.realized_pnl_usd
        self.portfolio.save()
        
        # Update position
        if exit_quantity >= self.quantity:
            # Full close
            self.is_open = False
            self.closed_at = datetime.utcnow()
            self.quantity = Decimal('0')
        else:
            # Partial close
            self.quantity -= exit_quantity
            # Update entry value for remaining position
            self.entry_value_usd = self.quantity * self.entry_price
            self.current_value_usd = self.quantity * self.current_price
            
        self.updated_at = datetime.utcnow()
        self.save()
        
        # Update AI decision performance with realized PnL
        if self.ai_decision:
            self.ai_decision.evaluate_outcome(
                current_price=Decimal(str(exit_price)),
                realized_pnl_usd=self.realized_pnl_usd,
                entry_value_usd=self.entry_value_usd 
            )
        
        return self.realized_pnl_usd

    def __str__(self):
        return f"{self.side.upper()} {self.quantity} {self.symbol} @ {self.entry_price}"


class Balance(Document):
    """Asset balances in the portfolio"""
    portfolio = fields.ReferenceField(Portfolio, required=True)
    asset = fields.StringField(max_length=10, required=True)  # e.g., 'BTC', 'USD', 'ETH'
    exchange = fields.StringField(max_length=50, required=True)
    
    # Balance amounts
    total_balance = fields.DecimalField(min_value=0, precision=8, default=Decimal('0'))
    available_balance = fields.DecimalField(min_value=0, precision=8, default=Decimal('0'))
    locked_balance = fields.DecimalField(min_value=0, precision=8, default=Decimal('0'))
    
    # USD values (for reporting)
    usd_value = fields.DecimalField(min_value=0, precision=8, default=Decimal('0'))
    last_price_usd = fields.DecimalField(min_value=0, precision=8)
    
    # Tracking
    last_updated = fields.DateTimeField(default=datetime.utcnow)
    created_at = fields.DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'balances',
        'indexes': [
            'portfolio', 'asset', 'exchange',
            ('portfolio', 'asset'), ('portfolio', 'exchange')
        ]
    }

    @classmethod
    def get_portfolio_balances(cls):
        """Get all balances for primary portfolio"""
        primary_portfolio = Portfolio.get_primary_portfolio()
        if primary_portfolio:
            return cls.objects(portfolio=primary_portfolio)
        return cls.objects.none()

    @classmethod
    def get_asset_balance(cls, asset, exchange=None):
        """Get balance for a specific asset"""
        primary_portfolio = Portfolio.get_primary_portfolio()
        if primary_portfolio:
            query = cls.objects(portfolio=primary_portfolio, asset=asset)
            if exchange:
                query = query.filter(exchange=exchange)
            return query.first()
        return None

    def update_balance(self, total, available, locked, usd_price=None):
        """Update balance amounts"""
        self.total_balance = total
        self.available_balance = available
        self.locked_balance = locked
        
        if usd_price:
            self.last_price_usd = usd_price
            self.usd_value = self.total_balance * usd_price
        
        self.last_updated = datetime.utcnow()
        self.save()

    def __str__(self):
        return f"{self.asset}: {self.total_balance} (${self.usd_value})"


class PortfolioHistory(Document):
    """Historical snapshots of portfolio performance"""
    portfolio = fields.ReferenceField(Portfolio, required=True)
    
    # Snapshot data
    timestamp = fields.DateTimeField(default=datetime.utcnow)
    total_value_usd = fields.DecimalField(min_value=0, precision=8)
    available_balance_usd = fields.DecimalField(min_value=0, precision=8)
    invested_balance_usd = fields.DecimalField(min_value=0, precision=8)
    unrealized_pnl_usd = fields.DecimalField(precision=8)
    realized_pnl_usd = fields.DecimalField(precision=8)
    
    # Performance metrics
    daily_return_pct = fields.DecimalField(precision=4)
    cumulative_return_pct = fields.DecimalField(precision=4)
    
    # Risk metrics
    positions_count = fields.IntField()
    total_exposure_usd = fields.DecimalField(min_value=0, precision=8)
    
    # Market conditions
    btc_price_usd = fields.DecimalField(min_value=0, precision=8)
    market_sentiment = fields.StringField(max_length=20)

    meta = {
        'collection': 'portfolio_history',
        'indexes': [
            'portfolio', 'timestamp',
            ('portfolio', 'timestamp')
        ]
    }

    @classmethod
    def create_snapshot(cls, portfolio):
        """Create a portfolio snapshot"""
        # Get current BTC price for market reference
        from market_data.models import MarketData
        btc_data = MarketData.get_latest_price('BTCUSD')
        
        return cls.objects.create(
            portfolio=portfolio,
            total_value_usd=portfolio.total_value_usd,
            available_balance_usd=portfolio.available_balance_usd,
            invested_balance_usd=portfolio.invested_balance_usd,
            unrealized_pnl_usd=portfolio.unrealized_pnl_usd,
            realized_pnl_usd=portfolio.realized_pnl_usd,
            daily_return_pct=portfolio.daily_return_pct,
            cumulative_return_pct=portfolio.total_return_pct,
            positions_count=portfolio.current_positions_count,
            total_exposure_usd=portfolio.total_exposure_usd,
            btc_price_usd=btc_data if btc_data else Decimal('0')
        )

    @classmethod
    def get_performance_history(cls, days=30):
        """Get performance history for primary portfolio"""
        primary_portfolio = Portfolio.get_primary_portfolio()
        if primary_portfolio:
            start_date = datetime.utcnow() - timedelta(days=days)
            return cls.objects(
                portfolio=primary_portfolio,
                timestamp__gte=start_date
            ).order_by('timestamp')
        return cls.objects.none()

    def __str__(self):
        return f"Portfolio Snapshot {self.timestamp.date()} - ${self.total_value_usd}"


class PerformanceAnalytics(Document):
    """Portfolio performance analytics and metrics"""
    portfolio = fields.ReferenceField(Portfolio, required=True)
    analysis_date = fields.DateTimeField(default=datetime.utcnow)
    period_days = fields.IntField(required=True)  # Analysis period
    
    # Return metrics
    total_return = fields.DecimalField(precision=6)
    annualized_return = fields.DecimalField(precision=6)
    sharpe_ratio = fields.DecimalField(precision=4)
    sortino_ratio = fields.DecimalField(precision=4)
    
    # Risk metrics
    volatility = fields.DecimalField(precision=6)
    max_drawdown = fields.DecimalField(precision=6)
    var_95 = fields.DecimalField(precision=6)  # Value at Risk 95%
    
    # Trading metrics
    total_trades = fields.IntField()
    winning_trades = fields.IntField()
    win_rate = fields.DecimalField(precision=4)
    profit_factor = fields.DecimalField(precision=4)
    avg_win = fields.DecimalField(precision=6)
    avg_loss = fields.DecimalField(precision=6)
    
    # Benchmark comparison
    benchmark_return = fields.DecimalField(precision=6)  # e.g., BTC buy and hold
    alpha = fields.DecimalField(precision=6)
    beta = fields.DecimalField(precision=4)

    meta = {
        'collection': 'performance_analytics',
        'indexes': ['portfolio', 'analysis_date', 'period_days']
    }

    @classmethod
    def generate_analytics(cls, portfolio, period_days=30):
        """Generate performance analytics for a period"""
        # This would calculate various metrics from portfolio history
        # Implementation would include statistical calculations
        pass

    def __str__(self):
        return f"Analytics {self.analysis_date.date()} - Return: {self.total_return}%"
