# 🤖 Automated Trading Setup Guide

## Overview
Your crypto trading system now automatically starts monitoring and making decisions when you start the Django server. No need to run separate commands!

## Quick Start

### 1. Configure Environment Variables
Copy `.env.example` to `.env` and update with your API keys:

```bash
cp .env.example .env
```

### 2. Key Configuration Options

```bash
# Auto-start trading when Django server starts
DJANGO_AUTO_START_TRADING=true

# Price change triggers
TRADING_PRICE_PCT_THRESHOLD=1.5        # Trigger decision when price moves 1.5%
TRADING_PRICE_DOLLAR_THRESHOLD=600     # OR when price moves $600

# Time-based triggers  
TRADING_MAX_TIME_BETWEEN_DECISIONS=1200  # Force decision every 20 minutes (1200 seconds)
TRADING_POLLING_INTERVAL=30             # Check price every 30 seconds
```

### 3. Start Django Server
```bash
python manage.py runserver
```

**That's it!** The trading system will automatically:
- ✅ Start monitoring BTC price every 30 seconds
- ✅ Trigger AI decisions when price moves ±1.5% or ±$600
- ✅ Force decisions every 20 minutes regardless of price
- ✅ Execute trades automatically when confidence >80%
- ✅ Update AI memory with trade performance

## Configuration Details

### Price Triggers
- **Percentage threshold**: `TRADING_PRICE_PCT_THRESHOLD=1.5` (1.5% price change)
- **Dollar threshold**: `TRADING_PRICE_DOLLAR_THRESHOLD=600` ($600 absolute change)
- **Logic**: Decision triggered if EITHER threshold is met

### Time Management
- **Max time between decisions**: `TRADING_MAX_TIME_BETWEEN_DECISIONS=1200` (20 minutes)
- **Minimum cooldown**: 5 minutes between decisions (hardcoded)
- **Polling interval**: `TRADING_POLLING_INTERVAL=30` (30 seconds)

### Trading Behavior
- **Confidence threshold**: 80% required for trade execution
- **Position sizing**: Uses available balance with 10x leverage
- **Memory integration**: All decisions stored in Pinecone for learning
- **Performance tracking**: Outcomes automatically update AI memory

## Advanced Usage

### Custom Configuration
You can override any setting via environment variables:

```bash
# More aggressive trading
TRADING_PRICE_PCT_THRESHOLD=0.8     # 0.8% price moves
TRADING_MAX_TIME_BETWEEN_DECISIONS=600   # 10 minutes max wait

# More conservative trading
TRADING_PRICE_PCT_THRESHOLD=3.0     # 3% price moves only  
TRADING_MAX_TIME_BETWEEN_DECISIONS=3600  # 1 hour max wait
```

### Disable Auto-Start
To prevent automatic trading when starting Django:

```bash
DJANGO_AUTO_START_TRADING=false
# OR remove the environment variable entirely
```

### Monitor Trading Activity
Check Django logs to see trading activity:

```bash
python manage.py runserver --verbosity=2
```

You'll see logs like:
```
🤖 Trading Trigger Service started automatically with Django server
📊 Configuration: {'price_change_pct_threshold': 1.5, ...}
🔄 Triggering decision: Price moved up by 1.8% ($1200)
🎯 Decision: BUY with 85% confidence
✅ TRADE EXECUTED SUCCESSFULLY
```

## System Architecture

```
Django Server Start
       ↓
AI Engine App.ready()
       ↓  
TradingTriggerService (background thread)
       ↓
Price Monitoring Loop (every 30s)
       ↓
Decision Triggers → TradingDecisionService
       ↓
DeepSeek AI + Pinecone Memory
       ↓
Trade Execution (if confidence >80%)
       ↓
Performance Feedback → Pinecone Memory
```

## Troubleshooting

### Trading Service Not Starting
1. Check `DJANGO_AUTO_START_TRADING=true` in .env
2. Verify Django logs for error messages
3. Ensure all API keys are configured

### No Trading Decisions
1. Check if price thresholds are too high
2. Verify Delta Exchange API connectivity
3. Check DeepSeek API limits

### Trades Not Executing
1. Verify confidence levels in logs
2. Check Delta Exchange balance
3. Ensure API keys have trading permissions

## Safety Features

- **Daemon threads**: Trading stops when Django stops
- **Error handling**: API failures don't crash Django
- **Cooldown periods**: Prevents spam trading
- **Confidence thresholds**: Only high-confidence trades execute
- **Memory learning**: System improves over time
