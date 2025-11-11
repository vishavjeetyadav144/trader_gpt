import threading
import time
import logging
from datetime import datetime
from django.apps import apps
from accounts.models import TradingUser
from exchanges.services import PortfolioSyncService

logger = logging.getLogger(__name__)


class PositionSyncService:
    """Service to manage position syncing threads for each trading user"""
    
    def __init__(self):
        self.sync_threads = {}
        self.running = False
    
    def start_sync_threads(self):
        """Start position sync threads for all active trading users"""
        if self.running:
            logger.warning("Position sync threads already running")
            return
            
        logger.info("Starting position sync threads...")
        self.running = True
        
        # Get all active trading users
        try:
            active_users = TradingUser.objects(is_active=True)
            logger.info(f"Found {active_users.count()} active trading users")
            
            for user in active_users:
                self.start_user_sync_thread(user)
                
        except Exception as e:
            logger.error(f"Error starting sync threads: {e}")
    
    def start_user_sync_thread(self, user):
        """Start position sync thread for a specific user"""
        if str(user.id) in self.sync_threads:
            logger.warning(f"Sync thread already exists for user {user.username}")
            return
            
        # Create and start thread for this user
        thread = threading.Thread(
            target=self._sync_worker,
            args=(user,),
            name=f"position_sync_{user.username}",
            daemon=True  # Dies when main process dies
        )
        
        self.sync_threads[str(user.id)] = thread
        thread.start()
        
        logger.info(f"Started position sync thread for user: {user.username}")
    
    def _sync_worker(self, user):
        """Worker function that runs in each sync thread"""
        logger.info(f"Position sync worker started for user: {user.username}")
        sync_service = PortfolioSyncService()
        
        while self.running:
            try:
                # Set the user context for this sync
                self._set_user_context(user)
                
                # Sync positions and balances
                logger.debug(f"Syncing positions for user: {user.username}")
                sync_service.sync_positions()
                sync_service.sync_balances()
                
                # Update portfolio metrics
                from portfolio.models import Portfolio
                portfolio = Portfolio.get_primary_portfolio()
                if portfolio and portfolio.user.id == user.id:
                    portfolio.update_metrics()
                    logger.debug(f"Updated portfolio metrics for {user.username}: ${portfolio.total_value_usd}")
                
            except Exception as e:
                logger.error(f"Error syncing positions for user {user.username}: {e}")
            
            # Wait 30 seconds before next sync
            time.sleep(5)
        
        logger.info(f"Position sync worker stopped for user: {user.username}")
    
    def _set_user_context(self, user):
        """Set user context for the current thread (if needed by sync service)"""
        # This can be used to set thread-local user context if needed
        pass
    
    def stop_sync_threads(self):
        """Stop all position sync threads"""
        if not self.running:
            return
            
        logger.info("Stopping position sync threads...")
        self.running = False
        
        # Wait for threads to finish
        for user_id, thread in self.sync_threads.items():
            if thread.is_alive():
                thread.join(timeout=5)  # Wait max 5 seconds
                
        self.sync_threads.clear()
        logger.info("All position sync threads stopped")
    
    def get_thread_status(self):
        """Get status of all sync threads"""
        status = {}
        for user_id, thread in self.sync_threads.items():
            status[user_id] = {
                'thread_name': thread.name,
                'is_alive': thread.is_alive(),
                'daemon': thread.daemon
            }
        return status


# Global instance
position_sync_service = PositionSyncService()
