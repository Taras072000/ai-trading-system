"""
Интеграция с традиционными финансовыми рынками
Поддержка акций, форекса, товаров, облигаций и индексов
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import yfinance as yf
import ccxt

class MarketType(Enum):
    STOCKS = "stocks"
    FOREX = "forex"
    COMMODITIES = "commodities"
    BONDS = "bonds"
    INDICES = "indices"
    CRYPTO = "crypto"

class ExchangeType(Enum):
    NYSE = "nyse"
    NASDAQ = "nasdaq"
    LSE = "lse"
    TSE = "tse"
    EURONEXT = "euronext"
    FOREX_MARKET = "forex"
    COMEX = "comex"
    NYMEX = "nymex"

class DataProvider(Enum):
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    IEX_CLOUD = "iex_cloud"
    QUANDL = "quandl"
    BLOOMBERG = "bloomberg"
    REUTERS = "reuters"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class SessionType(Enum):
    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    AFTER_HOURS = "after_hours"
    EXTENDED = "extended"

@dataclass
class MarketInstrument:
    symbol: str
    name: str
    market_type: MarketType
    exchange: ExchangeType
    currency: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    description: Optional[str] = None

@dataclass
class MarketData:
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None

@dataclass
class TradingOrder:
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    status: OrderStatus
    created_at: datetime
    filled_at: Optional[datetime] = None
    filled_quantity: float = 0.0
    average_price: Optional[float] = None

@dataclass
class Portfolio:
    id: str
    name: str
    total_value: float
    cash_balance: float
    positions: Dict[str, float]  # symbol -> quantity
    daily_pnl: float
    total_pnl: float
    allocation: Dict[MarketType, float]

@dataclass
class MarketAnalysis:
    symbol: str
    timestamp: datetime
    technical_indicators: Dict[str, float]
    sentiment_score: float
    volatility: float
    trend_direction: str
    support_levels: List[float]
    resistance_levels: List[float]
    recommendation: str

class DataFeedManager:
    """Менеджер потоков рыночных данных"""
    
    def __init__(self):
        self.active_feeds = {}
        self.data_cache = defaultdict(deque)
        self.cache_size = 1000
        self.providers = {
            DataProvider.YAHOO_FINANCE: self._yahoo_finance_feed,
            DataProvider.ALPHA_VANTAGE: self._alpha_vantage_feed,
            DataProvider.IEX_CLOUD: self._iex_cloud_feed
        }
    
    async def _yahoo_finance_feed(self, symbol: str) -> Optional[MarketData]:
        """Получение данных через Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d", interval="1m")
            
            if hist.empty:
                return None
            
            latest = hist.iloc[-1]
            return MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                open_price=float(latest['Open']),
                high_price=float(latest['High']),
                low_price=float(latest['Low']),
                close_price=float(latest['Close']),
                volume=int(latest['Volume'])
            )
        except Exception as e:
            logging.error(f"Yahoo Finance feed error for {symbol}: {e}")
            return None
    
    async def _alpha_vantage_feed(self, symbol: str) -> Optional[MarketData]:
        """Получение данных через Alpha Vantage"""
        # Заглушка для Alpha Vantage API
        return MarketData(
            symbol=symbol,
            timestamp=datetime.now(),
            open_price=np.random.uniform(100, 200),
            high_price=np.random.uniform(200, 250),
            low_price=np.random.uniform(90, 100),
            close_price=np.random.uniform(150, 200),
            volume=np.random.randint(1000000, 10000000)
        )
    
    async def _iex_cloud_feed(self, symbol: str) -> Optional[MarketData]:
        """Получение данных через IEX Cloud"""
        # Заглушка для IEX Cloud API
        return MarketData(
            symbol=symbol,
            timestamp=datetime.now(),
            open_price=np.random.uniform(50, 150),
            high_price=np.random.uniform(150, 200),
            low_price=np.random.uniform(40, 50),
            close_price=np.random.uniform(100, 150),
            volume=np.random.randint(500000, 5000000)
        )
    
    async def start_feed(self, symbol: str, provider: DataProvider):
        """Запуск потока данных для символа"""
        if symbol in self.active_feeds:
            return
        
        self.active_feeds[symbol] = {
            "provider": provider,
            "task": asyncio.create_task(self._feed_loop(symbol, provider))
        }
        
        logging.info(f"Started data feed for {symbol} via {provider.value}")
    
    async def stop_feed(self, symbol: str):
        """Остановка потока данных"""
        if symbol in self.active_feeds:
            self.active_feeds[symbol]["task"].cancel()
            del self.active_feeds[symbol]
            logging.info(f"Stopped data feed for {symbol}")
    
    async def _feed_loop(self, symbol: str, provider: DataProvider):
        """Цикл получения данных"""
        while True:
            try:
                data = await self.providers[provider](symbol)
                if data:
                    self.data_cache[symbol].append(data)
                    if len(self.data_cache[symbol]) > self.cache_size:
                        self.data_cache[symbol].popleft()
                
                await asyncio.sleep(60)  # Обновление каждую минуту
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Feed loop error for {symbol}: {e}")
                await asyncio.sleep(30)
    
    def get_latest_data(self, symbol: str) -> Optional[MarketData]:
        """Получение последних данных"""
        if symbol in self.data_cache and self.data_cache[symbol]:
            return self.data_cache[symbol][-1]
        return None
    
    def get_historical_data(self, symbol: str, count: int = 100) -> List[MarketData]:
        """Получение исторических данных"""
        if symbol in self.data_cache:
            return list(self.data_cache[symbol])[-count:]
        return []

class TechnicalAnalyzer:
    """Технический анализ рыночных данных"""
    
    def __init__(self):
        self.indicators = {
            "sma": self._simple_moving_average,
            "ema": self._exponential_moving_average,
            "rsi": self._relative_strength_index,
            "macd": self._macd,
            "bollinger_bands": self._bollinger_bands,
            "stochastic": self._stochastic_oscillator
        }
    
    def _simple_moving_average(self, prices: List[float], period: int = 20) -> float:
        """Простая скользящая средняя"""
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        return sum(prices[-period:]) / period
    
    def _exponential_moving_average(self, prices: List[float], period: int = 20) -> float:
        """Экспоненциальная скользящая средняя"""
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _relative_strength_index(self, prices: List[float], period: int = 14) -> float:
        """Индекс относительной силы"""
        if len(prices) < period + 1:
            return 50.0
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _macd(self, prices: List[float]) -> Dict[str, float]:
        """MACD индикатор"""
        if len(prices) < 26:
            return {"macd": 0.0, "signal": 0.0, "histogram": 0.0}
        
        ema12 = self._exponential_moving_average(prices, 12)
        ema26 = self._exponential_moving_average(prices, 26)
        macd_line = ema12 - ema26
        
        # Упрощенный расчет сигнальной линии
        signal_line = macd_line * 0.9  # Приближение
        histogram = macd_line - signal_line
        
        return {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram
        }
    
    def _bollinger_bands(self, prices: List[float], period: int = 20, std_dev: float = 2.0) -> Dict[str, float]:
        """Полосы Боллинджера"""
        if len(prices) < period:
            price = prices[-1] if prices else 0.0
            return {"upper": price, "middle": price, "lower": price}
        
        sma = self._simple_moving_average(prices, period)
        recent_prices = prices[-period:]
        variance = sum((p - sma) ** 2 for p in recent_prices) / period
        std = variance ** 0.5
        
        return {
            "upper": sma + (std_dev * std),
            "middle": sma,
            "lower": sma - (std_dev * std)
        }
    
    def _stochastic_oscillator(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Dict[str, float]:
        """Стохастический осциллятор"""
        if len(closes) < period:
            return {"k": 50.0, "d": 50.0}
        
        recent_highs = highs[-period:]
        recent_lows = lows[-period:]
        current_close = closes[-1]
        
        highest_high = max(recent_highs)
        lowest_low = min(recent_lows)
        
        if highest_high == lowest_low:
            k_percent = 50.0
        else:
            k_percent = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
        
        # Упрощенный расчет %D
        d_percent = k_percent * 0.9  # Приближение
        
        return {"k": k_percent, "d": d_percent}
    
    async def analyze_symbol(self, symbol: str, data_feed: DataFeedManager) -> MarketAnalysis:
        """Анализ символа"""
        historical_data = data_feed.get_historical_data(symbol, 100)
        
        if not historical_data:
            return MarketAnalysis(
                symbol=symbol,
                timestamp=datetime.now(),
                technical_indicators={},
                sentiment_score=0.5,
                volatility=0.0,
                trend_direction="neutral",
                support_levels=[],
                resistance_levels=[],
                recommendation="hold"
            )
        
        prices = [data.close_price for data in historical_data]
        highs = [data.high_price for data in historical_data]
        lows = [data.low_price for data in historical_data]
        
        # Расчет технических индикаторов
        indicators = {
            "sma_20": self._simple_moving_average(prices, 20),
            "sma_50": self._simple_moving_average(prices, 50),
            "ema_12": self._exponential_moving_average(prices, 12),
            "rsi": self._relative_strength_index(prices),
            "macd": self._macd(prices)["macd"],
            "bb_upper": self._bollinger_bands(prices)["upper"],
            "bb_lower": self._bollinger_bands(prices)["lower"],
            "stoch_k": self._stochastic_oscillator(highs, lows, prices)["k"]
        }
        
        # Определение тренда
        current_price = prices[-1]
        sma_20 = indicators["sma_20"]
        sma_50 = indicators["sma_50"]
        
        if current_price > sma_20 > sma_50:
            trend = "bullish"
        elif current_price < sma_20 < sma_50:
            trend = "bearish"
        else:
            trend = "neutral"
        
        # Расчет волатильности
        if len(prices) > 1:
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            volatility = np.std(returns) * np.sqrt(252)  # Годовая волатильность
        else:
            volatility = 0.0
        
        # Уровни поддержки и сопротивления
        support_levels = [min(lows[-20:]), min(lows[-50:])]
        resistance_levels = [max(highs[-20:]), max(highs[-50:])]
        
        # Рекомендация
        rsi = indicators["rsi"]
        if rsi < 30 and trend == "bullish":
            recommendation = "strong_buy"
        elif rsi < 40 and trend == "bullish":
            recommendation = "buy"
        elif rsi > 70 and trend == "bearish":
            recommendation = "strong_sell"
        elif rsi > 60 and trend == "bearish":
            recommendation = "sell"
        else:
            recommendation = "hold"
        
        return MarketAnalysis(
            symbol=symbol,
            timestamp=datetime.now(),
            technical_indicators=indicators,
            sentiment_score=np.random.uniform(0.3, 0.7),  # Заглушка для sentiment
            volatility=volatility,
            trend_direction=trend,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            recommendation=recommendation
        )

class TraditionalMarketsIntegrator:
    """Главный класс интеграции с традиционными рынками"""
    
    def __init__(self):
        self.data_feed = DataFeedManager()
        self.technical_analyzer = TechnicalAnalyzer()
        self.instruments = {}
        self.portfolios = {}
        self.orders = {}
        self.market_sessions = {}
        
        # Инициализация популярных инструментов
        self._initialize_instruments()
        
        # Настройка логирования
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _initialize_instruments(self):
        """Инициализация популярных торговых инструментов"""
        # Акции
        stocks = [
            ("AAPL", "Apple Inc.", "Technology"),
            ("GOOGL", "Alphabet Inc.", "Technology"),
            ("MSFT", "Microsoft Corporation", "Technology"),
            ("AMZN", "Amazon.com Inc.", "Consumer Discretionary"),
            ("TSLA", "Tesla Inc.", "Consumer Discretionary"),
            ("NVDA", "NVIDIA Corporation", "Technology"),
            ("META", "Meta Platforms Inc.", "Technology"),
            ("BRK-B", "Berkshire Hathaway Inc.", "Financial Services"),
            ("JPM", "JPMorgan Chase & Co.", "Financial Services"),
            ("JNJ", "Johnson & Johnson", "Healthcare")
        ]
        
        for symbol, name, sector in stocks:
            self.instruments[symbol] = MarketInstrument(
                symbol=symbol,
                name=name,
                market_type=MarketType.STOCKS,
                exchange=ExchangeType.NASDAQ if symbol in ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META"] else ExchangeType.NYSE,
                currency="USD",
                sector=sector
            )
        
        # Форекс пары
        forex_pairs = [
            ("EURUSD", "Euro/US Dollar"),
            ("GBPUSD", "British Pound/US Dollar"),
            ("USDJPY", "US Dollar/Japanese Yen"),
            ("USDCHF", "US Dollar/Swiss Franc"),
            ("AUDUSD", "Australian Dollar/US Dollar"),
            ("USDCAD", "US Dollar/Canadian Dollar"),
            ("NZDUSD", "New Zealand Dollar/US Dollar"),
            ("EURGBP", "Euro/British Pound")
        ]
        
        for symbol, name in forex_pairs:
            self.instruments[symbol] = MarketInstrument(
                symbol=symbol,
                name=name,
                market_type=MarketType.FOREX,
                exchange=ExchangeType.FOREX_MARKET,
                currency="USD"
            )
        
        # Товары
        commodities = [
            ("GC=F", "Gold Futures", "COMEX"),
            ("SI=F", "Silver Futures", "COMEX"),
            ("CL=F", "Crude Oil Futures", "NYMEX"),
            ("NG=F", "Natural Gas Futures", "NYMEX"),
            ("ZC=F", "Corn Futures", "CBOT"),
            ("ZW=F", "Wheat Futures", "CBOT")
        ]
        
        for symbol, name, exchange in commodities:
            self.instruments[symbol] = MarketInstrument(
                symbol=symbol,
                name=name,
                market_type=MarketType.COMMODITIES,
                exchange=ExchangeType.COMEX if exchange == "COMEX" else ExchangeType.NYMEX,
                currency="USD"
            )
        
        # Индексы
        indices = [
            ("^GSPC", "S&P 500"),
            ("^DJI", "Dow Jones Industrial Average"),
            ("^IXIC", "NASDAQ Composite"),
            ("^RUT", "Russell 2000"),
            ("^VIX", "CBOE Volatility Index")
        ]
        
        for symbol, name in indices:
            self.instruments[symbol] = MarketInstrument(
                symbol=symbol,
                name=name,
                market_type=MarketType.INDICES,
                exchange=ExchangeType.NYSE,
                currency="USD"
            )
    
    async def start_market_data(self, symbols: List[str], provider: DataProvider = DataProvider.YAHOO_FINANCE):
        """Запуск получения рыночных данных"""
        for symbol in symbols:
            await self.data_feed.start_feed(symbol, provider)
        
        self.logger.info(f"Started market data for {len(symbols)} symbols")
    
    async def stop_market_data(self, symbols: List[str]):
        """Остановка получения рыночных данных"""
        for symbol in symbols:
            await self.data_feed.stop_feed(symbol)
        
        self.logger.info(f"Stopped market data for {len(symbols)} symbols")
    
    def create_portfolio(self, portfolio_id: str, name: str, initial_cash: float) -> Portfolio:
        """Создание портфеля"""
        portfolio = Portfolio(
            id=portfolio_id,
            name=name,
            total_value=initial_cash,
            cash_balance=initial_cash,
            positions={},
            daily_pnl=0.0,
            total_pnl=0.0,
            allocation={}
        )
        
        self.portfolios[portfolio_id] = portfolio
        self.logger.info(f"Created portfolio {name} with ${initial_cash:,.2f}")
        
        return portfolio
    
    async def place_order(self, portfolio_id: str, symbol: str, side: OrderSide, 
                         order_type: OrderType, quantity: float, price: Optional[float] = None) -> TradingOrder:
        """Размещение ордера"""
        if portfolio_id not in self.portfolios:
            raise ValueError(f"Portfolio {portfolio_id} not found")
        
        if symbol not in self.instruments:
            raise ValueError(f"Instrument {symbol} not supported")
        
        order_id = f"order_{len(self.orders)}_{datetime.now().timestamp()}"
        
        order = TradingOrder(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=None,
            status=OrderStatus.PENDING,
            created_at=datetime.now()
        )
        
        self.orders[order_id] = order
        
        # Симуляция исполнения ордера
        await self._simulate_order_execution(order, portfolio_id)
        
        self.logger.info(f"Placed {side.value} order for {quantity} {symbol} at {price}")
        
        return order
    
    async def _simulate_order_execution(self, order: TradingOrder, portfolio_id: str):
        """Симуляция исполнения ордера"""
        portfolio = self.portfolios[portfolio_id]
        
        # Получение текущей цены
        market_data = self.data_feed.get_latest_data(order.symbol)
        if not market_data:
            # Генерация случайной цены для демонстрации
            execution_price = order.price or np.random.uniform(100, 200)
        else:
            execution_price = market_data.close_price
        
        # Расчет стоимости сделки
        trade_value = order.quantity * execution_price
        
        if order.side == OrderSide.BUY:
            if portfolio.cash_balance >= trade_value:
                # Исполнение покупки
                portfolio.cash_balance -= trade_value
                if order.symbol in portfolio.positions:
                    portfolio.positions[order.symbol] += order.quantity
                else:
                    portfolio.positions[order.symbol] = order.quantity
                
                order.status = OrderStatus.FILLED
                order.filled_at = datetime.now()
                order.filled_quantity = order.quantity
                order.average_price = execution_price
                
                self.logger.info(f"Executed BUY order: {order.quantity} {order.symbol} at ${execution_price:.2f}")
            else:
                order.status = OrderStatus.REJECTED
                self.logger.warning(f"Insufficient funds for BUY order: {order.symbol}")
        
        else:  # SELL
            if order.symbol in portfolio.positions and portfolio.positions[order.symbol] >= order.quantity:
                # Исполнение продажи
                portfolio.cash_balance += trade_value
                portfolio.positions[order.symbol] -= order.quantity
                
                if portfolio.positions[order.symbol] == 0:
                    del portfolio.positions[order.symbol]
                
                order.status = OrderStatus.FILLED
                order.filled_at = datetime.now()
                order.filled_quantity = order.quantity
                order.average_price = execution_price
                
                self.logger.info(f"Executed SELL order: {order.quantity} {order.symbol} at ${execution_price:.2f}")
            else:
                order.status = OrderStatus.REJECTED
                self.logger.warning(f"Insufficient position for SELL order: {order.symbol}")
        
        # Обновление стоимости портфеля
        await self._update_portfolio_value(portfolio_id)
    
    async def _update_portfolio_value(self, portfolio_id: str):
        """Обновление стоимости портфеля"""
        portfolio = self.portfolios[portfolio_id]
        total_value = portfolio.cash_balance
        
        for symbol, quantity in portfolio.positions.items():
            market_data = self.data_feed.get_latest_data(symbol)
            if market_data:
                position_value = quantity * market_data.close_price
                total_value += position_value
            else:
                # Использование случайной цены для демонстрации
                position_value = quantity * np.random.uniform(100, 200)
                total_value += position_value
        
        old_value = portfolio.total_value
        portfolio.total_value = total_value
        portfolio.daily_pnl = total_value - old_value
        portfolio.total_pnl += portfolio.daily_pnl
        
        # Обновление распределения по типам активов
        allocation = defaultdict(float)
        for symbol in portfolio.positions:
            if symbol in self.instruments:
                market_type = self.instruments[symbol].market_type
                market_data = self.data_feed.get_latest_data(symbol)
                if market_data:
                    position_value = portfolio.positions[symbol] * market_data.close_price
                    allocation[market_type] += position_value / total_value
        
        portfolio.allocation = dict(allocation)
    
    async def get_market_analysis(self, symbol: str) -> MarketAnalysis:
        """Получение рыночного анализа"""
        return await self.technical_analyzer.analyze_symbol(symbol, self.data_feed)
    
    async def get_portfolio_summary(self, portfolio_id: str) -> Dict:
        """Получение сводки портфеля"""
        if portfolio_id not in self.portfolios:
            raise ValueError(f"Portfolio {portfolio_id} not found")
        
        portfolio = self.portfolios[portfolio_id]
        await self._update_portfolio_value(portfolio_id)
        
        positions_detail = []
        for symbol, quantity in portfolio.positions.items():
            market_data = self.data_feed.get_latest_data(symbol)
            current_price = market_data.close_price if market_data else np.random.uniform(100, 200)
            position_value = quantity * current_price
            
            positions_detail.append({
                "symbol": symbol,
                "quantity": quantity,
                "current_price": current_price,
                "position_value": position_value,
                "weight": position_value / portfolio.total_value if portfolio.total_value > 0 else 0
            })
        
        return {
            "portfolio_id": portfolio_id,
            "name": portfolio.name,
            "total_value": portfolio.total_value,
            "cash_balance": portfolio.cash_balance,
            "daily_pnl": portfolio.daily_pnl,
            "total_pnl": portfolio.total_pnl,
            "positions": positions_detail,
            "allocation": portfolio.allocation,
            "last_updated": datetime.now().isoformat()
        }
    
    async def get_market_overview(self) -> Dict:
        """Получение обзора рынка"""
        overview = {
            "timestamp": datetime.now().isoformat(),
            "markets": {},
            "top_movers": {"gainers": [], "losers": []},
            "market_sentiment": "neutral"
        }
        
        # Анализ по типам рынков
        for market_type in MarketType:
            market_instruments = [
                symbol for symbol, instrument in self.instruments.items()
                if instrument.market_type == market_type
            ]
            
            if market_instruments:
                # Получение данных для первых 5 инструментов каждого типа
                sample_instruments = market_instruments[:5]
                market_data = []
                
                for symbol in sample_instruments:
                    data = self.data_feed.get_latest_data(symbol)
                    if data:
                        market_data.append({
                            "symbol": symbol,
                            "price": data.close_price,
                            "volume": data.volume,
                            "change": np.random.uniform(-5, 5)  # Симуляция изменения
                        })
                
                overview["markets"][market_type.value] = {
                    "instruments_count": len(market_instruments),
                    "sample_data": market_data,
                    "avg_change": np.mean([d["change"] for d in market_data]) if market_data else 0
                }
        
        return overview
    
    async def run_background_tasks(self):
        """Запуск фоновых задач"""
        tasks = [
            asyncio.create_task(self._portfolio_monitoring_loop()),
            asyncio.create_task(self._market_analysis_loop()),
            asyncio.create_task(self._order_management_loop())
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _portfolio_monitoring_loop(self):
        """Цикл мониторинга портфелей"""
        while True:
            try:
                for portfolio_id in self.portfolios:
                    await self._update_portfolio_value(portfolio_id)
                
                await asyncio.sleep(300)  # Обновление каждые 5 минут
                
            except Exception as e:
                self.logger.error(f"Portfolio monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _market_analysis_loop(self):
        """Цикл рыночного анализа"""
        while True:
            try:
                # Анализ топ-10 инструментов
                top_symbols = list(self.instruments.keys())[:10]
                
                for symbol in top_symbols:
                    analysis = await self.get_market_analysis(symbol)
                    
                    # Логирование важных сигналов
                    if analysis.recommendation in ["strong_buy", "strong_sell"]:
                        self.logger.info(
                            f"Strong signal for {symbol}: {analysis.recommendation} "
                            f"(RSI: {analysis.technical_indicators.get('rsi', 0):.1f}, "
                            f"Trend: {analysis.trend_direction})"
                        )
                
                await asyncio.sleep(1800)  # Анализ каждые 30 минут
                
            except Exception as e:
                self.logger.error(f"Market analysis error: {e}")
                await asyncio.sleep(300)
    
    async def _order_management_loop(self):
        """Цикл управления ордерами"""
        while True:
            try:
                # Проверка pending ордеров
                pending_orders = [
                    order for order in self.orders.values()
                    if order.status == OrderStatus.PENDING
                ]
                
                for order in pending_orders:
                    # Симуляция возможного исполнения
                    if np.random.random() < 0.1:  # 10% шанс исполнения
                        # Найти портфель для этого ордера
                        for portfolio_id in self.portfolios:
                            await self._simulate_order_execution(order, portfolio_id)
                            break
                
                await asyncio.sleep(60)  # Проверка каждую минуту
                
            except Exception as e:
                self.logger.error(f"Order management error: {e}")
                await asyncio.sleep(30)

# Пример использования
async def main():
    """Пример использования интегратора традиционных рынков"""
    integrator = TraditionalMarketsIntegrator()
    
    try:
        # Запуск рыночных данных для популярных инструментов
        symbols = ["AAPL", "GOOGL", "MSFT", "EURUSD", "GC=F", "^GSPC"]
        await integrator.start_market_data(symbols)
        
        # Создание портфеля
        portfolio = integrator.create_portfolio("portfolio_1", "Demo Portfolio", 100000.0)
        
        # Размещение тестовых ордеров
        await integrator.place_order(
            "portfolio_1", "AAPL", OrderSide.BUY, OrderType.MARKET, 10
        )
        
        await integrator.place_order(
            "portfolio_1", "GOOGL", OrderSide.BUY, OrderType.MARKET, 5
        )
        
        # Ожидание некоторого времени для накопления данных
        await asyncio.sleep(60)
        
        # Получение анализа рынка
        analysis = await integrator.get_market_analysis("AAPL")
        print(f"AAPL Analysis: {analysis.recommendation} (RSI: {analysis.technical_indicators.get('rsi', 0):.1f})")
        
        # Получение сводки портфеля
        portfolio_summary = await integrator.get_portfolio_summary("portfolio_1")
        print(f"Portfolio Value: ${portfolio_summary['total_value']:,.2f}")
        print(f"Positions: {len(portfolio_summary['positions'])}")
        
        # Получение обзора рынка
        market_overview = await integrator.get_market_overview()
        print(f"Market Overview: {len(market_overview['markets'])} market types")
        
        # Запуск фоновых задач на короткое время
        background_task = asyncio.create_task(integrator.run_background_tasks())
        await asyncio.sleep(300)  # 5 минут
        background_task.cancel()
        
    finally:
        # Остановка рыночных данных
        await integrator.stop_market_data(symbols)

if __name__ == "__main__":
    asyncio.run(main())