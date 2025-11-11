from django.apps import AppConfig
import logging

logger = logging.getLogger(__name__)


class PortfolioConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'portfolio'

    def ready(self):
        """Called when Django app is ready - start position sync threads"""
        # Import here to avoid circular imports
        from portfolio.services import position_sync_service
        
        try:
            # Start position syncing threads for all active users
            logger.info("Django portfolio app ready - starting position sync threads")
            position_sync_service.start_sync_threads()
            
        except Exception as e:
            logger.error(f"Error starting position sync threads in portfolio app: {e}")
