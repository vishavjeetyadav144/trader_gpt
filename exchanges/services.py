import requests
import hashlib
import hmac
import time
import json
import logging
from decimal import Decimal
from datetime import datetime, timezone
import pandas as pd
import ta

from django.conf import settings
from accounts.models import TradingUser
from portfolio.models import Portfolio, Position, Balance

logger = logging.getLogger(__name__)


class DeltaExchangeClient:
    """Delta Exchange API client for real trading operations"""
    
    def __init__(self):
        self.base_url = settings.EXCHANGE_CONFIG['DELTA_EXCHANGE']['BASE_URL']
        self.api_key = settings.EXCHANGE_CONFIG['DELTA_EXCHANGE']['API_KEY']
        self.api_secret = settings.EXCHANGE_CONFIG['DELTA_EXCHANGE']['API_SECRET']
        self.is_sandbox = settings.EXCHANGE_CONFIG['DELTA_EXCHANGE']['SANDBOX']
        
        if self.is_sandbox:
            self.base_url = "https://testnet-api.delta.exchange"
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Delta Exchange API credentials not configured")

    def _generate_signature(self, method, endpoint, timestamp, body=""):
        """Generate API signature for authentication"""
        message = method + timestamp + endpoint + body
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _make_request(self, method, endpoint, params=None, data=None):
        """Make authenticated request to Delta Exchange"""
        url = f"{self.base_url}{endpoint}"
        timestamp = str(int(time.time()))
        
        headers = {
            'api-key': self.api_key,
            'timestamp': timestamp,
            'Content-Type': 'application/json'
        }
        
        # Prepare body for signature
        body = ""
        if data:
            body = json.dumps(data)
            
        # Generate signature
        signature = self._generate_signature(method, endpoint, timestamp, body)
        headers['signature'] = signature
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, params=params, timeout=10)
            elif method == 'POST':
                response = requests.post(url, headers=headers, json=data, timeout=10)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=10)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Delta Exchange API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Delta API call: {e}")
            raise

    def get_balance(self):
        """Fetch account balance"""
        try:
            response = self._make_request('GET', '/v2/wallet/balances')
            return response.get('result', [])
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            return []

    def get_positions(self, product_id):
        """Fetch active positions"""
        try:
            response = self._make_request('GET', f'/v2/positions/margined?product_ids={product_id}')
            return response.get('result', [])
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return []

    def place_order(self, quantity, ai_decision):
        """Place trading order"""
        endpoint = '/v2/orders'
        
        order_data = {
            "product_symbol": ai_decision.symbol,
            #"limit_price": ai_decision.recommended_entry,
            "size": quantity,
            "side": ai_decision.decision_type,
            "order_type": "market_order",
            "bracket_stop_loss_price": ai_decision.recommended_stop_loss,
            "bracket_take_profit_price": ai_decision.recommended_take_profit
        }

        for key, value in order_data.items():
            if isinstance(value, Decimal):
                order_data[key] = float(value)
        
        try:
            response = self._make_request('POST', endpoint, data=order_data)
            return response.get('result')
        except requests.exceptions.HTTPError as e:
            # Extract and print detailed API error response
            if e.response is not None:
                logger.error(f"Delta Exchange API error: {e.response.status_code}")
                try:
                    logger.error(f"Response content: {e.response.json()}")
                except Exception:
                    logger.error(f"Response text: {e.response.text}")
            else:
                logger.error(f"HTTPError raised without response: {e}")
            raise

        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise

    def close_position(self, position):

        endpoint = '/v2/orders'
        opposite_side = 'sell' if position.side == 'buy' else 'buy'

        # Construct order payload
        order_data = {
            "product_symbol": position.symbol,
            "size": int(position.quantity),
            "side": opposite_side,
            "order_type": "market_order"
        }

        try:
            response = self._make_request('POST', endpoint, data=order_data)
            result = response.get('result')

            if result:
                logger.info(
                    f"✅ Closed position for {position.symbol} | "
                    f"Side: {opposite_side} | Quantity: {position.quantity}"
                )
            else:
                logger.warning(f"⚠️ No result returned when closing {position.symbol}")

            return result

        except requests.exceptions.HTTPError as e:
            # Detailed API error logging
            if e.response is not None:
                logger.error(f"Delta Exchange API error: {e.response.status_code}")
                try:
                    logger.error(f"Response content: {e.response.json()}")
                except Exception:
                    logger.error(f"Response text: {e.response.text}")
            else:
                logger.error(f"HTTPError raised without response: {e}")
            raise

        except Exception as e:
            logger.error(f"❌ Failed to close position for {position.symbol}: {e}")
            raise

    def cancel_order(self, order_id):
        """Cancel existing order"""
        endpoint = f'/v2/orders/{order_id}'
        try:
            response = self._make_request('DELETE', endpoint)
            return response.get('result')
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            raise

    def get_order_status(self, order_id):
        """Get order status"""
        endpoint = f'/v2/orders/{order_id}'
        try:
            response = self._make_request('GET', endpoint)
            return response.get('result')
        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            return None

    def get_pending_orders(self, symbol=None):
        """Get pending/open orders"""
        endpoint = '/v2/orders'
        params = {'state': 'open'}  # Only open/pending orders
        if symbol:
            params['product_symbol'] = symbol
            
        try:
            response = self._make_request('GET', endpoint, params=params)
            return response.get('result', [])
        except Exception as e:
            logger.error(f"Failed to get pending orders: {e}")
            return []
            
    def is_order_pending(self, order_id):
        """Check if a specific order is still pending"""
        try:
            order_status = self.get_order_status(order_id)
            if order_status:
                state = order_status.get('state', '').lower()
                return state in ['open', 'pending', 'partially_filled']
            return False
        except Exception as e:
            logger.error(f"Failed to check order status for {order_id}: {e}")
            return False

    def _get_product_id(self, symbol):
        """Convert symbol to Delta Exchange product ID"""
        # This would typically fetch from products endpoint and cache
        # For now, using common mappings
        symbol_map = {
            'BTCUSD': 27,  # BTC-USD perpetual
            'ETHUSD': 135,  # ETH-USD perpetual
            # Add more mappings as needed
        }
        return symbol_map.get(symbol, symbol)

    def fetch_candles(self, symbol="MARK:BTCUSD", resolution="1m", start_ts=None, end_ts=None):
        """Fetch historical OHLCV data"""
        url = f"{self.base_url}/v2/history/candles"
        params = {
            "symbol": symbol.replace("MARK:", ""),
            "resolution": resolution,
            "start": start_ts,
            "end": end_ts
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            # Convert to DataFrame
            df = pd.DataFrame(data.get('result', []), 
                            columns=["time", "open", "high", "low", "close", "volume"])
            
            if df.empty:
                return df
                
            # Convert data types
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
            
            # Reverse order (newest first to oldest first)
            df = df.iloc[::-1].reset_index(drop=True)
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch candles: {e}")
            return pd.DataFrame()
    
    def get_ticker(self, symbol="BTCUSD"):
        """Fetch historical OHLCV data"""

        try:
            response = self._make_request('GET', f'/v2/tickers/{symbol}')
            return response.get('result')
            
        except Exception as e:
            logger.error(f"Failed to fetch candles: {e}")
            return pd.DataFrame()


class TechnicalAnalysisService:
    """Service for calculating technical indicators"""
    
    @staticmethod
    def compute_indicators(df):
        """Compute comprehensive technical indicators using ta library"""
        if df.empty or len(df) < 50:  # Need enough data for indicators
            return df
            
        try:
            # Moving averages
            df['ema20'] = ta.trend.ema_indicator(df['close'], window=20)
            df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)
            df['sma20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['sma50'] = ta.trend.sma_indicator(df['close'], window=50)
            
            # MACD
            df['macd'] = ta.trend.macd(df['close'], window_slow=26, window_fast=12)
            df['macd_signal'] = ta.trend.macd_signal(df['close'], window_slow=26, window_fast=12, window_sign=9)
            df['macd_hist'] = ta.trend.macd_diff(df['close'], window_slow=26, window_fast=12, window_sign=9)
            
            # RSI
            df['rsi_7'] = ta.momentum.rsi(df['close'], window=7)
            df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
            
            # ATR
            df['atr_3'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=3)
            df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
            
            # Bollinger Bands
            df['bb_upper'] = ta.volatility.bollinger_hband(df['close'], window=20, window_dev=2)
            df['bb_middle'] = ta.volatility.bollinger_mavg(df['close'], window=20)
            df['bb_lower'] = ta.volatility.bollinger_lband(df['close'], window=20, window_dev=2)
            
            # Volume weighted average (simplified)
            df['vwma20'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
            
            # Calculate average volume
            df['avg_volume'] = df['volume'].rolling(window=20).mean()

            return df
            
        except Exception as e:
            logger.error(f"Error computing indicators: {e}")
            return df

    @staticmethod
    def get_latest_snapshot(symbol="MARK:BTCUSD", resolution="1m", lookback_minutes=1000):
        """Get latest technical analysis snapshot"""
        try:
            client = DeltaExchangeClient()
            
            # Calculate timestamp range
            now = int(datetime.now(timezone.utc).timestamp())
            start = now - lookback_minutes * 60
            end = now
            
            # Fetch data and compute indicators
            df = client.fetch_candles(symbol, resolution, start, end)
            if df.empty:
                return None

            df = TechnicalAnalysisService.compute_indicators(df)
            
            # Get latest values
            latest = df.iloc[-1]
            snapshot = {
                "symbol": symbol.replace("MARK:", ""),
                "timestamp": latest["time"].isoformat(),
                "price": float(latest["close"]),
                "ema20": round(float(latest.get("ema20", 0)), 3),
                "ema50": round(float(latest.get("ema50", 0)), 3),
                "macd": round(float(latest.get("macd", 0)), 3),
                "macd_signal": round(float(latest.get("macd_signal", 0)), 3),
                "macd_hist": round(float(latest.get("macd_hist", 0)), 3),
                "rsi_7": round(float(latest.get("rsi_7", 0)), 3),
                "rsi_14": round(float(latest.get("rsi_14", 0)), 3),
                "atr_3": round(float(latest.get("atr_3", 0)), 3),
                "atr_14": round(float(latest.get("atr_14", 0)), 3),
                "volume": round(float(latest["volume"]), 3),
                "avg_volume": round(float(latest.get("avg_volume", 0)), 3)
            }
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error getting latest snapshot: {e}")
            return None


class PortfolioSyncService:
    """Service to sync portfolio data with exchange"""
    
    def __init__(self):
        self.client = DeltaExchangeClient()
    
    def sync_balances(self):
        """Sync account balances from exchange"""
        try:
            exchange_balances = self.client.get_balance()
            portfolio = Portfolio.get_primary_portfolio()
            
            if not portfolio:
                logger.error("No primary portfolio found")
                return
                
            for balance_data in exchange_balances:
                asset = balance_data.get('asset_symbol')
                if asset != "USD":
                    continue
                    
                total = Decimal(str(balance_data.get('balance', 0)))
                available = Decimal(str(balance_data.get('available_balance', 0)))
                locked = total - available
                
                # Directly update portfolio available balance
                portfolio.total_value_usd = total
                portfolio.available_balance_usd = available
                portfolio.invested_balance_usd = locked
                portfolio.save()
                    
            # Update portfolio metrics
            portfolio.update_metrics()
            
        except Exception as e:
            logger.error(f"Error syncing balances: {e}")
    
    def sync_positions(self):
        """Sync active positions from exchange"""
        try:
            
            portfolio = Portfolio.get_primary_portfolio()
            
            if not portfolio:
                logger.error("No primary portfolio found")
                return
            
            positions = Position.objects(
                        portfolio=portfolio,
                        exchange='delta',
                        is_open=True
                    )
        
            for position in positions:
                
                product_id = position.product_id
                exchange_positions = self.client.get_positions(product_id)
               
                exchange_pos = next(
                    (p for p in exchange_positions if str(p.get("product_id")) == str(product_id)),
                    None
                )

                if not exchange_pos:
                    # Position not found on exchange - check if order is still pending
                    if position.order_id:
                        is_pending = self.client.is_order_pending(position.order_id)
                        if is_pending:
                            logger.info(f"Position {position.symbol} not on exchange but order {position.order_id} still pending - keeping position open")
                            continue
                    
                    # If no order_id or order is filled but no exchange position, close the position
                    logger.warning(f"Position {position.symbol} not found on exchange and no pending order - closing position")
                    data = self.client.get_ticker(position.symbol)
                    if data and data.get('mark_price'):
                        position.close_position(Decimal(str(data.get('mark_price'))))
                    else:
                        # Fallback if ticker data not available
                        position.is_open = False
                        position.closed_at = datetime.utcnow()
                        position.save()
                    continue
            
                size = float(exchange_pos.get('size', 0))
                if size == 0:
                    logger.info(f"Position {position.symbol} size is 0 — closing locally")
                    position.is_open = False
                    position.closed_at = datetime.utcnow()
                    position.save()
                    continue

                side = 'buy' if size > 0 else 'sell'
                quantity = abs(Decimal(str(size)))

                entry_price = Decimal(str(exchange_pos.get('entry_price', 0)))
                current_price = Decimal(str(exchange_pos.get('mark_price', 0)))
                contract_value = Decimal(str(exchange_pos.get('product', {}).get('contract_value', 0.001)))

                entry_value_usd = Decimal(str(exchange_pos.get('margin', 0)))
                unrealized_pnl_usd = Decimal(str(exchange_pos.get('unrealized_pnl', 0)))
                current_value_usd = entry_value_usd+ unrealized_pnl_usd

                # position.side = side
                # position.quantity = quantity
                position.entry_price = entry_price
                position.current_price = current_price
                position.entry_value_usd = entry_value_usd
                position.current_value_usd = current_value_usd
                position.unrealized_pnl_usd = unrealized_pnl_usd
                position.updated_at = datetime.utcnow()
                position.save()

            portfolio.update_metrics()
                
        except Exception as e:
            logger.error(f"Error syncing positions: {e}")
