from django.apps import AppConfig
import threading
import logging
import os

logger = logging.getLogger(__name__)


class AiEngineConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "ai_engine"
    
    def ready(self):
        """Called when Django starts up - initialize background services"""
        # Only start the trigger service if we're running the main Django process
        # (not during migrations, management commands, etc.)
        if os.environ.get('RUN_MAIN') or os.environ.get('DJANGO_AUTO_START_TRADING') == 'true':
            self.start_trading_trigger_service()
    
    def start_trading_trigger_service(self):
        """Start the trading trigger service in a background thread"""
        try:
            from ai_engine.trigger_service import TradingTriggerService
            
            # Default configuration - can be customized
            config = {
                'price_change_pct_threshold': 0.5,  # 0.5% price change
                'price_change_dollar_threshold': 500,  # $200 absolute change
                'max_time_between_decisions': 30 * 60,  # 30 minutes
                'min_cooldown_between_decisions': 1 * 60,  # 5 minutes
                'polling_interval': 1,  # 30 seconds
                'symbol': 'BTCUSD'
            }
            
            # Override with environment variables if provided
            if os.environ.get('TRADING_PRICE_PCT_THRESHOLD'):
                config['price_change_pct_threshold'] = float(os.environ.get('TRADING_PRICE_PCT_THRESHOLD'))
            
            if os.environ.get('TRADING_PRICE_DOLLAR_THRESHOLD'):
                config['price_change_dollar_threshold'] = float(os.environ.get('TRADING_PRICE_DOLLAR_THRESHOLD'))
            
            if os.environ.get('TRADING_MAX_TIME_BETWEEN_DECISIONS'):
                config['max_time_between_decisions'] = int(os.environ.get('TRADING_MAX_TIME_BETWEEN_DECISIONS'))
            
            if os.environ.get('TRADING_POLLING_INTERVAL'):
                config['polling_interval'] = int(os.environ.get('TRADING_POLLING_INTERVAL'))
            
            # Create and start the trigger service in a daemon thread
            trigger_service = TradingTriggerService(config)
            
            # Start monitoring in a separate daemon thread
            trading_thread = threading.Thread(
                target=trigger_service.start_monitoring,
                daemon=True,
                name="TradingTriggerService"
            )
            trading_thread.start()
            
            logger.info("🤖 Trading Trigger Service started automatically with Django server")
            logger.info(f"📊 Configuration: {config}")
            
        except Exception as e:
            logger.error(f"Failed to start Trading Trigger Service: {e}")
            # Don't raise - let Django continue to start even if trading service fails
