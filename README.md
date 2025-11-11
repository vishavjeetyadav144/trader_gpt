# Crypto Swing Trading Agent

A sophisticated AI-powered crypto trading system built with Django, MongoDB, DeepSeek LLM, Pinecone vector database, and Delta Exchange integration.

## 🚀 Features

### Core Architecture
- **Modular Design**: Clean separation of concerns across 6 Django apps
- **Single User Focus**: Optimized for single user initially, easily scalable to multiple users
- **MongoDB Backend**: High-performance document database with MongoEngine ODM
- **AI-Driven Decisions**: DeepSeek LLM for intelligent trading decisions
- **Long-term Memory**: Pinecone vector database for AI context and learning
- **Professional Exchange Integration**: Delta Exchange API for trade execution

### Key Components

#### 🤖 AI Engine (`ai_engine`)
- DeepSeek LLM integration for decision-making
- AI decision tracking and performance analytics
- Conversation history with context preservation
- Long-term memory storage using Pinecone
- Model performance monitoring and optimization

#### 📊 Trading System (`trading`)
- AI-powered trading strategies
- Real-time trade execution and monitoring
- Advanced risk management with position limits
- Trading signals generation and tracking
- Portfolio risk assessment

#### 📈 Market Data (`market_data`)
- Real-time market data collection
- Technical indicators calculation (RSI, MACD, Bollinger Bands)
- Market sentiment analysis
- Price alerts and monitoring
- Multi-timeframe data support

#### 💼 Portfolio Management (`portfolio`)
- Real-time portfolio tracking
- Position management (long/short)
- Performance analytics and metrics
- Historical portfolio snapshots
- Multi-asset balance tracking

#### 🏦 Exchange Integration (`exchanges`)
- Delta Exchange API integration
- Order management and execution
- Balance synchronization
- Transaction history tracking
- Multi-exchange support framework

#### 👤 User Management (`accounts`)
- Single primary trader setup
- Trading session management
- Secure API key storage
- User preferences and risk settings

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.11+
- MongoDB (local or cloud)
- Redis (for caching and Celery)

### 1. Clone and Setup
```bash
git clone <your-repo>
cd super_trader
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Configuration
Copy `.env.example` to `.env` and configure:

```bash
# Django Configuration
SECRET_KEY=your-secret-key-here
DEBUG=True

# MongoDB Configuration
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=crypto_trader

# DeepSeek AI Configuration
DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_MODEL=deepseek-coder

# Pinecone Configuration
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=crypto-trading

# Delta Exchange API
DELTA_API_KEY=your-delta-api-key
DELTA_API_SECRET=your-delta-api-secret
DELTA_SANDBOX=True

# Trading Configuration
MAX_POSITION_SIZE=0.1
RISK_PER_TRADE=0.02
MAX_DAILY_LOSS=0.05

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
```

### 3. Bootstrap the System
```bash
# Check system configuration
python manage.py check

# Bootstrap the trading system
python manage.py bootstrap_trader

# Follow the interactive prompts to set up your primary trader account
```

### 4. Start the System
```bash
# Start Django development server
python manage.py runserver

# In separate terminals, start background services:
celery -A crypto_trader worker -l info
celery -A crypto_trader beat -l info
```

## 📋 API Endpoints (Coming Soon)

The system is designed with REST API endpoints for:
- Portfolio management
- Trading operations  
- Market data access
- AI decision insights
- Risk management controls

## 🏗️ Architecture Highlights

### Single User, Scalable Design
- Primary user concept with `is_primary_trader` flags
- Easy migration to multi-user by removing primary user restrictions
- User-scoped data throughout the system

### MongoDB Schema Design
- Optimized document structure for trading data
- Efficient indexing for high-frequency operations
- Flexible schema for evolving trading strategies

### AI Integration
- DeepSeek LLM for market analysis and decision making
- Pinecone vector database for semantic search and memory
- Performance tracking and model optimization

### Risk Management
- Position-level risk controls
- Portfolio-level exposure limits
- Daily loss limits and circuit breakers
- Real-time risk monitoring

## 🔧 Development

### Project Structure
```
super_trader/
├── accounts/           # User management
├── ai_engine/         # AI/LLM integration
├── exchanges/         # Exchange APIs
├── market_data/       # Market data & indicators
├── portfolio/         # Portfolio management
├── trading/           # Trading logic
├── crypto_trader/     # Django settings
└── requirements.txt   # Dependencies
```

### Key Models
- `TradingUser`: Primary trader account
- `Portfolio`: Portfolio tracking
- `Position`: Individual positions
- `Trade`: Trade execution records
- `AIDecision`: AI decision tracking
- `MarketData`: OHLCV data storage
- `TechnicalIndicator`: Calculated indicators

### Management Commands
- `bootstrap_trader`: Initial system setup
- `setup_exchange_connection`: Configure exchange APIs
- `sync_market_data`: Update market data
- `run_ai_analysis`: Execute AI trading analysis

## 📊 Monitoring & Analytics

The system provides comprehensive monitoring:
- Real-time portfolio performance
- AI decision accuracy tracking
- Risk exposure monitoring
- Trading performance analytics
- Market data quality metrics

## 🔐 Security

- Encrypted API key storage
- Secure MongoDB connections
- Rate limiting and request throttling
- Input validation and sanitization
- Sandbox mode for testing

## 🤝 Contributing

This is a professional trading system. Please ensure:
- Comprehensive testing for any changes
- Proper risk management validation
- Documentation updates
- Code review for security implications

## ⚠️ Risk Disclaimer

This is a trading system that involves financial risk. Always:
- Start with paper trading or sandbox mode
- Use appropriate position sizing
- Monitor risk limits closely
- Test thoroughly before live trading
- Never risk more than you can afford to lose

## 📝 License

[Your chosen license]

## 🆘 Support

For support and questions:
- Review the documentation
- Check the Django logs in `logs/crypto_trader.log`
- Monitor MongoDB and Redis connections
- Verify API key configurations
