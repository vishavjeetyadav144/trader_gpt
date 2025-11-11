import logging
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional
from django.conf import settings
from exchanges.services import DeltaExchangeClient
from ai_engine.services import TradingDecisionService
from portfolio.models import Portfolio, Position

logger = logging.getLogger(__name__)


class TradingTriggerService:
    """Service to monitor price changes and trigger trading decisions"""
    
    def __init__(self, config: Optional[dict] = None):
        # Default configuration
        self.config = {
            'price_change_pct_threshold': 1.5,  # 1.5% price change
            'price_change_dollar_threshold': 600,  # $600 absolute change
            'max_time_between_decisions': 20 * 60,  # 20 minutes
            'min_cooldown_between_decisions': 5 * 60,  # 5 minutes
            'polling_interval': 30,  # 30 seconds
            'symbol': 'BTCUSD'
        }
        
        # Override with user config if provided
        if config:
            self.config.update(config)
        
        # Initialize services
        self.delta_client = DeltaExchangeClient()
        self.trading_service = TradingDecisionService()
        
        # State tracking
        self.last_price: Optional[Decimal] = None
        self.last_decision_time: float = 0
        self.last_price_check_time: float = 0
        self.is_running: bool = False
        
        logger.info(f"TradingTriggerService initialized with config: {self.config}")
    
    def start_monitoring(self):
        """Start the price monitoring loop"""
        logger.info("Starting price monitoring...")
        self.is_running = True
        
        try:
            # Initialize with current price
            self._initialize_price()
            
            while self.is_running:
                try:
                    self._check_and_decide()
                    time.sleep(self.config['polling_interval'])
                    
                except KeyboardInterrupt:
                    logger.info("Monitoring stopped by user")
                    break
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(60)  # Wait 1 minute on error before retrying
                    
        except Exception as e:
            logger.error(f"Fatal error in monitoring service: {e}")
        finally:
            self.is_running = False
            logger.info("Price monitoring stopped")
    
    def stop_monitoring(self):
        """Stop the price monitoring loop"""
        logger.info("Stopping price monitoring...")
        self.is_running = False
    
    def _initialize_price(self):
        """Get initial price to start monitoring"""
        try:
            current_price = self._get_current_price()
            if current_price:
                self.last_price = current_price
                self.last_decision_time = time.time()
                logger.info(f"Initialized monitoring at price: ${current_price}")
            else:
                logger.error("Failed to get initial price")
                
        except Exception as e:
            logger.error(f"Error initializing price: {e}")
    
    def _check_and_decide(self):
        """Check if we should make a trading decision"""
        try:
            # Get current price
            current_price = self._get_current_price()
            if not current_price:
                logger.error("Failed to get current price, skipping check")
                return
            
            current_time = time.time()
            self.last_price_check_time = current_time
            
            # Log current status periodically (every 5 minutes)
            if current_time - getattr(self, 'last_status_log', 0) > 300:
                self._log_status(current_price)
                self.last_status_log = current_time
            
            # Check if we should trigger a decision
            should_decide, reason = self._should_make_decision(current_price, current_time)
            logger.info(f"Triggering decision: {should_decide}, reason: {reason}")
            if should_decide:
                self._make_decision_and_update_state(current_price, current_time)
            
        except Exception as e:
            logger.error(f"Error in check and decide: {e}")
    
    def _should_make_decision(self, current_price: Decimal, current_time: float) -> tuple[bool, str]:
        """Determine if we should make a trading decision"""
        
        if not self.last_price:
            return False, "No previous price available"
        
        # Check cooldown period
        time_since_last_decision = current_time - self.last_decision_time
        if time_since_last_decision < self.config['min_cooldown_between_decisions']:
            return False, f"In cooldown period ({time_since_last_decision:.1f}s < {self.config['min_cooldown_between_decisions']}s)"
        
        # Calculate price changes
        price_diff = abs(current_price - self.last_price)
        price_change_pct = (price_diff / self.last_price) * 100
        
        # Check price change triggers
        pct_threshold = self.config['price_change_pct_threshold']
        dollar_threshold = self.config['price_change_dollar_threshold']
        
        price_trigger_pct = price_change_pct >= pct_threshold
        price_trigger_dollar = price_diff >= dollar_threshold
        
        if price_trigger_pct or price_trigger_dollar:
            direction = "up" if current_price > self.last_price else "down"
            return True, f"Price moved {direction} by {price_change_pct:.2f}% (${price_diff})"
        
        # Check time trigger
        time_threshold = self.config['max_time_between_decisions']
        if time_since_last_decision >= time_threshold:
            return True, f"Time threshold reached ({time_since_last_decision:.1f}s >= {time_threshold}s)"
        
        return False, "No triggers met"
    
    def _make_decision_and_update_state(self, current_price: Decimal, current_time: float):
        """Make trading decision and update internal state"""
        try:
            logger.info(f"Making trading decision at price ${current_price}")
            
            # Make the trading decision
            decision_result = self.trading_service.make_trading_decision(
                symbol=self.config['symbol']
            )
            
            if decision_result:
                logger.info(f"Trading decision completed: {decision_result.get('final_decision', {}).get('signal', 'UNKNOWN')}")
                
                # Update state after successful decision
                self.last_price = current_price
                self.last_decision_time = current_time
                
                # Log decision summary
                final_decision = decision_result.get('final_decision', {})
                logger.info(f"Decision: {final_decision.get('signal')} with {final_decision.get('confidence', 0)}% confidence")
                
            else:
                logger.error("Trading decision failed - no result returned")
                
        except Exception as e:
            logger.error(f"Error making trading decision: {e}")
    
    def _get_current_price(self) -> Optional[Decimal]:
        """Get current BTC price from Delta Exchange"""
        try:
            ticker_data = self.delta_client.get_ticker(self.config['symbol'])
            if ticker_data and 'mark_price' in ticker_data:
                return Decimal(str(ticker_data['mark_price']))
            else:
                logger.warning("No mark price in ticker data")
                return None
                
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return None
    
    def _log_status(self, current_price: Decimal):
        """Log current monitoring status"""
        try:
            current_time = time.time()
            time_since_last_decision = current_time - self.last_decision_time
            
            # Get portfolio status
            portfolio = Portfolio.get_primary_portfolio()
            open_positions = Position.get_open_positions()
            
            status_info = {
                "current_price": float(current_price),
                "last_price": float(self.last_price) if self.last_price else None,
                "time_since_last_decision": f"{time_since_last_decision/60:.1f} minutes",
                "portfolio_value": float(portfolio.total_value_usd) if portfolio else 0,
                "open_positions": len(open_positions),
                "monitoring_uptime": f"{(current_time - getattr(self, 'start_time', current_time))/3600:.1f} hours"
            }
            
            logger.info(f"Status: {status_info}")
            
        except Exception as e:
            logger.error(f"Error logging status: {e}")
    
    def get_status(self) -> dict:
        """Get current status information"""
        return {
            "is_running": self.is_running,
            "config": self.config,
            "last_price": float(self.last_price) if self.last_price else None,
            "last_decision_time": datetime.fromtimestamp(self.last_decision_time).isoformat() if self.last_decision_time else None,
            "last_price_check_time": datetime.fromtimestamp(self.last_price_check_time).isoformat() if self.last_price_check_time else None,
        }
    
    def update_config(self, new_config: dict):
        """Update configuration while running"""
        self.config.update(new_config)
        logger.info(f"Configuration updated: {new_config}")


class TradingTriggerManager:
    """Manager class for the trading trigger service"""
    
    def __init__(self):
        self.trigger_service: Optional[TradingTriggerService] = None
    
    def start_with_config(self, config: dict):
        """Start monitoring with custom configuration"""
        if self.trigger_service and self.trigger_service.is_running:
            logger.warning("Trigger service already running, stopping first...")
            self.trigger_service.stop_monitoring()
        
        self.trigger_service = TradingTriggerService(config)
        self.trigger_service.start_monitoring()
    
    def start_with_defaults(self):
        """Start monitoring with default configuration"""
        self.start_with_config({})
    
    def stop(self):
        """Stop the monitoring service"""
        if self.trigger_service:
            self.trigger_service.stop_monitoring()
        else:
            logger.warning("No trigger service running")
    
    def get_status(self) -> dict:
        """Get current status"""
        if self.trigger_service:
            return self.trigger_service.get_status()
        else:
            return {"is_running": False, "message": "No trigger service initialized"}
