import os
import redis
import logging

logger = logging.getLogger(__name__)

# Load from environment for flexibility
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

try:
    redis_client = redis.StrictRedis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,
        decode_responses=True,
        socket_timeout=5,
    )

    # Quick health check
    redis_client.ping()
    logger.info(f"✅ Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")

except Exception as e:
    logger.error(f"❌ Redis connection failed: {e}")
    redis_client = None
