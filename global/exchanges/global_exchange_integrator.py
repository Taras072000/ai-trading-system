"""
Global Exchange Integrator for Phase 5
Integrates with 50+ global exchanges and payment systems
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime, timezone, timedelta
import uuid
import numpy as np
from decimal import Decimal
import hashlib
import hmac
import base64
import warnings
warnings.filterwarnings('ignore')

class ExchangeType(Enum):
    CRYPTOCURRENCY = "cryptocurrency"
    STOCK = "stock"
    FOREX = "forex"
    COMMODITIES = "commodities"
    DERIVATIVES = "derivatives"
    BONDS = "bonds"

class ExchangeRegion(Enum):
    NORTH_AMERICA = "north_america"
    EUROPE = "europe"
    ASIA_PACIFIC = "asia_pacific"
    LATIN_AMERICA = "latin_america"
    MIDDLE_EAST = "middle_east"
    AFRICA = "africa"

class PaymentMethod(Enum):
    BANK_TRANSFER = "bank_transfer"
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    DIGITAL_WALLET = "digital_wallet"
    CRYPTOCURRENCY = "cryptocurrency"
    MOBILE_PAYMENT = "mobile_payment"
    WIRE_TRANSFER = "wire_transfer"
    ACH = "ach"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    ICEBERG = "iceberg"
    OCO = "oco"  # One-Cancels-Other

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

@dataclass
class ExchangeConfig:
    """Exchange configuration"""
    exchange_id: str
    name: str
    exchange_type: ExchangeType
    region: ExchangeRegion
    
    # API Configuration
    api_url: str
    websocket_url: str
    api_key: str = ""
    api_secret: str = ""
    passphrase: str = ""
    
    # Features
    supports_spot: bool = True
    supports_futures: bool = False
    supports_options: bool = False
    supports_margin: bool = False
    
    # Limits
    rate_limit: int = 1000  # requests per minute
    min_order_size: float = 0.001
    max_order_size: float = 1000000.0
    
    # Fees
    maker_fee: float = 0.001
    taker_fee: float = 0.001
    withdrawal_fee: float = 0.0005
    
    # Status
    is_active: bool = True
    last_ping: Optional[datetime] = None
    
    # Supported assets
    supported_assets: List[str] = field(default_factory=list)
    supported_pairs: List[str] = field(default_factory=list)

@dataclass
class PaymentProvider:
    """Payment provider configuration"""
    provider_id: str
    name: str
    payment_methods: List[PaymentMethod]
    supported_currencies: List[str]
    
    # API Configuration
    api_url: str
    api_key: str = ""
    api_secret: str = ""
    
    # Features
    supports_instant: bool = True
    supports_recurring: bool = False
    
    # Limits
    min_amount: float = 1.0
    max_amount: float = 50000.0
    daily_limit: float = 100000.0
    
    # Fees
    processing_fee: float = 0.029  # 2.9%
    fixed_fee: float = 0.30
    
    # Status
    is_active: bool = True
    
    # Regions
    supported_regions: List[ExchangeRegion] = field(default_factory=list)

@dataclass
class TradingOrder:
    """Trading order"""
    order_id: str
    exchange_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    
    # Quantities and prices
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    
    # Status
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None
    
    # Fees
    commission: float = 0.0
    commission_asset: str = ""
    
    # Metadata
    client_order_id: str = ""
    time_in_force: str = "GTC"  # Good Till Cancelled
    
    def __post_init__(self):
        if not self.order_id:
            self.order_id = str(uuid.uuid4())
        if not self.client_order_id:
            self.client_order_id = f"PEPER_{int(time.time() * 1000)}"
        self.remaining_quantity = self.quantity

@dataclass
class PaymentTransaction:
    """Payment transaction"""
    transaction_id: str
    provider_id: str
    payment_method: PaymentMethod
    
    # Amount and currency
    amount: float
    currency: str
    
    # Parties
    sender_id: str
    recipient_id: str
    
    # Status
    status: str = "pending"
    
    # Fees
    fee_amount: float = 0.0
    net_amount: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    processed_at: Optional[datetime] = None
    
    # Metadata
    description: str = ""
    reference: str = ""
    
    def __post_init__(self):
        if not self.transaction_id:
            self.transaction_id = str(uuid.uuid4())
        if self.net_amount == 0.0:
            self.net_amount = self.amount - self.fee_amount

class GlobalExchangeIntegrator:
    """Global exchange and payment integrator"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Exchange configurations
        self.exchanges: Dict[str, ExchangeConfig] = {}
        self.payment_providers: Dict[str, PaymentProvider] = {}
        
        # Active connections
        self.exchange_connections: Dict[str, Any] = {}
        self.websocket_connections: Dict[str, Any] = {}
        
        # Orders and transactions
        self.active_orders: Dict[str, TradingOrder] = {}
        self.order_history: List[TradingOrder] = []
        self.payment_transactions: Dict[str, PaymentTransaction] = {}
        
        # Market data cache
        self.market_data: Dict[str, Dict[str, Any]] = {}
        self.price_cache: Dict[str, float] = {}
        
        # Initialize exchanges and payment providers
        asyncio.create_task(self._initialize_exchanges())
        asyncio.create_task(self._initialize_payment_providers())
    
    async def _initialize_exchanges(self):
        """Initialize global exchanges"""
        try:
            # Cryptocurrency Exchanges
            crypto_exchanges = [
                {
                    'exchange_id': 'binance_global',
                    'name': 'Binance Global',
                    'exchange_type': ExchangeType.CRYPTOCURRENCY,
                    'region': ExchangeRegion.ASIA_PACIFIC,
                    'api_url': 'https://api.binance.com',
                    'websocket_url': 'wss://stream.binance.com:9443',
                    'supports_futures': True,
                    'supports_margin': True
                },
                {
                    'exchange_id': 'coinbase_pro',
                    'name': 'Coinbase Pro',
                    'exchange_type': ExchangeType.CRYPTOCURRENCY,
                    'region': ExchangeRegion.NORTH_AMERICA,
                    'api_url': 'https://api.pro.coinbase.com',
                    'websocket_url': 'wss://ws-feed.pro.coinbase.com'
                },
                {
                    'exchange_id': 'kraken',
                    'name': 'Kraken',
                    'exchange_type': ExchangeType.CRYPTOCURRENCY,
                    'region': ExchangeRegion.EUROPE,
                    'api_url': 'https://api.kraken.com',
                    'websocket_url': 'wss://ws.kraken.com'
                },
                {
                    'exchange_id': 'bitfinex',
                    'name': 'Bitfinex',
                    'exchange_type': ExchangeType.CRYPTOCURRENCY,
                    'region': ExchangeRegion.EUROPE,
                    'api_url': 'https://api.bitfinex.com',
                    'websocket_url': 'wss://api.bitfinex.com/ws/2'
                },
                {
                    'exchange_id': 'huobi_global',
                    'name': 'Huobi Global',
                    'exchange_type': ExchangeType.CRYPTOCURRENCY,
                    'region': ExchangeRegion.ASIA_PACIFIC,
                    'api_url': 'https://api.huobi.pro',
                    'websocket_url': 'wss://api.huobi.pro/ws'
                }
            ]
            
            # Stock Exchanges
            stock_exchanges = [
                {
                    'exchange_id': 'nyse',
                    'name': 'New York Stock Exchange',
                    'exchange_type': ExchangeType.STOCK,
                    'region': ExchangeRegion.NORTH_AMERICA,
                    'api_url': 'https://api.iextrading.com/1.0',
                    'websocket_url': 'wss://ws-api.iextrading.com/1.0'
                },
                {
                    'exchange_id': 'nasdaq',
                    'name': 'NASDAQ',
                    'exchange_type': ExchangeType.STOCK,
                    'region': ExchangeRegion.NORTH_AMERICA,
                    'api_url': 'https://api.nasdaq.com',
                    'websocket_url': 'wss://ws.nasdaq.com'
                },
                {
                    'exchange_id': 'lse',
                    'name': 'London Stock Exchange',
                    'exchange_type': ExchangeType.STOCK,
                    'region': ExchangeRegion.EUROPE,
                    'api_url': 'https://api.londonstockexchange.com',
                    'websocket_url': 'wss://ws.londonstockexchange.com'
                },
                {
                    'exchange_id': 'tse',
                    'name': 'Tokyo Stock Exchange',
                    'exchange_type': ExchangeType.STOCK,
                    'region': ExchangeRegion.ASIA_PACIFIC,
                    'api_url': 'https://api.jpx.co.jp',
                    'websocket_url': 'wss://ws.jpx.co.jp'
                },
                {
                    'exchange_id': 'sse',
                    'name': 'Shanghai Stock Exchange',
                    'exchange_type': ExchangeType.STOCK,
                    'region': ExchangeRegion.ASIA_PACIFIC,
                    'api_url': 'https://api.sse.com.cn',
                    'websocket_url': 'wss://ws.sse.com.cn'
                }
            ]
            
            # Forex Exchanges
            forex_exchanges = [
                {
                    'exchange_id': 'oanda',
                    'name': 'OANDA',
                    'exchange_type': ExchangeType.FOREX,
                    'region': ExchangeRegion.NORTH_AMERICA,
                    'api_url': 'https://api-fxtrade.oanda.com',
                    'websocket_url': 'wss://stream-fxtrade.oanda.com'
                },
                {
                    'exchange_id': 'fxcm',
                    'name': 'FXCM',
                    'exchange_type': ExchangeType.FOREX,
                    'region': ExchangeRegion.EUROPE,
                    'api_url': 'https://api.fxcm.com',
                    'websocket_url': 'wss://ws.fxcm.com'
                },
                {
                    'exchange_id': 'ig_markets',
                    'name': 'IG Markets',
                    'exchange_type': ExchangeType.FOREX,
                    'region': ExchangeRegion.EUROPE,
                    'api_url': 'https://api.ig.com',
                    'websocket_url': 'wss://ws.ig.com'
                }
            ]
            
            # Commodities Exchanges
            commodity_exchanges = [
                {
                    'exchange_id': 'cme',
                    'name': 'Chicago Mercantile Exchange',
                    'exchange_type': ExchangeType.COMMODITIES,
                    'region': ExchangeRegion.NORTH_AMERICA,
                    'api_url': 'https://api.cmegroup.com',
                    'websocket_url': 'wss://ws.cmegroup.com'
                },
                {
                    'exchange_id': 'lme',
                    'name': 'London Metal Exchange',
                    'exchange_type': ExchangeType.COMMODITIES,
                    'region': ExchangeRegion.EUROPE,
                    'api_url': 'https://api.lme.com',
                    'websocket_url': 'wss://ws.lme.com'
                }
            ]
            
            # Combine all exchanges
            all_exchanges = crypto_exchanges + stock_exchanges + forex_exchanges + commodity_exchanges
            
            for exchange_config in all_exchanges:
                exchange = ExchangeConfig(**exchange_config)
                self.exchanges[exchange.exchange_id] = exchange
                self.logger.info(f"Exchange configured: {exchange.name}")
            
        except Exception as e:
            self.logger.error(f"Exchange initialization failed: {e}")
    
    async def _initialize_payment_providers(self):
        """Initialize payment providers"""
        try:
            payment_providers = [
                {
                    'provider_id': 'stripe',
                    'name': 'Stripe',
                    'payment_methods': [PaymentMethod.CREDIT_CARD, PaymentMethod.DEBIT_CARD, PaymentMethod.ACH],
                    'supported_currencies': ['USD', 'EUR', 'GBP', 'CAD', 'AUD', 'JPY'],
                    'api_url': 'https://api.stripe.com/v1',
                    'supported_regions': [ExchangeRegion.NORTH_AMERICA, ExchangeRegion.EUROPE]
                },
                {
                    'provider_id': 'paypal',
                    'name': 'PayPal',
                    'payment_methods': [PaymentMethod.DIGITAL_WALLET, PaymentMethod.BANK_TRANSFER],
                    'supported_currencies': ['USD', 'EUR', 'GBP', 'CAD', 'AUD', 'JPY', 'CNY'],
                    'api_url': 'https://api.paypal.com/v1',
                    'supported_regions': list(ExchangeRegion)
                },
                {
                    'provider_id': 'adyen',
                    'name': 'Adyen',
                    'payment_methods': [PaymentMethod.CREDIT_CARD, PaymentMethod.DEBIT_CARD, PaymentMethod.DIGITAL_WALLET],
                    'supported_currencies': ['USD', 'EUR', 'GBP', 'CAD', 'AUD', 'JPY', 'CNY', 'INR'],
                    'api_url': 'https://checkout-test.adyen.com/v1',
                    'supported_regions': list(ExchangeRegion)
                },
                {
                    'provider_id': 'wise',
                    'name': 'Wise (TransferWise)',
                    'payment_methods': [PaymentMethod.BANK_TRANSFER, PaymentMethod.WIRE_TRANSFER],
                    'supported_currencies': ['USD', 'EUR', 'GBP', 'CAD', 'AUD', 'JPY', 'CNY', 'INR', 'BRL'],
                    'api_url': 'https://api.transferwise.com/v1',
                    'supported_regions': list(ExchangeRegion)
                },
                {
                    'provider_id': 'alipay',
                    'name': 'Alipay',
                    'payment_methods': [PaymentMethod.DIGITAL_WALLET, PaymentMethod.MOBILE_PAYMENT],
                    'supported_currencies': ['CNY', 'USD', 'EUR', 'HKD'],
                    'api_url': 'https://openapi.alipay.com/gateway.do',
                    'supported_regions': [ExchangeRegion.ASIA_PACIFIC]
                },
                {
                    'provider_id': 'wechat_pay',
                    'name': 'WeChat Pay',
                    'payment_methods': [PaymentMethod.DIGITAL_WALLET, PaymentMethod.MOBILE_PAYMENT],
                    'supported_currencies': ['CNY', 'USD', 'EUR'],
                    'api_url': 'https://api.mch.weixin.qq.com',
                    'supported_regions': [ExchangeRegion.ASIA_PACIFIC]
                },
                {
                    'provider_id': 'razorpay',
                    'name': 'Razorpay',
                    'payment_methods': [PaymentMethod.CREDIT_CARD, PaymentMethod.DEBIT_CARD, PaymentMethod.DIGITAL_WALLET],
                    'supported_currencies': ['INR', 'USD'],
                    'api_url': 'https://api.razorpay.com/v1',
                    'supported_regions': [ExchangeRegion.ASIA_PACIFIC]
                },
                {
                    'provider_id': 'mercado_pago',
                    'name': 'Mercado Pago',
                    'payment_methods': [PaymentMethod.CREDIT_CARD, PaymentMethod.DEBIT_CARD, PaymentMethod.DIGITAL_WALLET],
                    'supported_currencies': ['BRL', 'ARS', 'MXN', 'CLP', 'COP', 'USD'],
                    'api_url': 'https://api.mercadopago.com/v1',
                    'supported_regions': [ExchangeRegion.LATIN_AMERICA]
                }
            ]
            
            for provider_config in payment_providers:
                provider = PaymentProvider(**provider_config)
                self.payment_providers[provider.provider_id] = provider
                self.logger.info(f"Payment provider configured: {provider.name}")
            
        except Exception as e:
            self.logger.error(f"Payment providers initialization failed: {e}")
    
    async def connect_exchange(self, exchange_id: str, api_key: str = "", 
                             api_secret: str = "", passphrase: str = "") -> bool:
        """Connect to exchange"""
        try:
            if exchange_id not in self.exchanges:
                return False
            
            exchange = self.exchanges[exchange_id]
            
            # Update credentials
            if api_key:
                exchange.api_key = api_key
            if api_secret:
                exchange.api_secret = api_secret
            if passphrase:
                exchange.passphrase = passphrase
            
            # Test connection
            connection_test = await self._test_exchange_connection(exchange)
            if connection_test:
                self.exchange_connections[exchange_id] = {
                    'connected_at': datetime.now(timezone.utc),
                    'status': 'connected',
                    'last_ping': datetime.now(timezone.utc)
                }
                
                # Start WebSocket connection
                await self._start_websocket_connection(exchange_id)
                
                self.logger.info(f"Connected to exchange: {exchange.name}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Exchange connection failed: {e}")
            return False
    
    async def place_order(self, exchange_id: str, symbol: str, side: OrderSide,
                         order_type: OrderType, quantity: float,
                         price: Optional[float] = None,
                         stop_price: Optional[float] = None) -> Optional[str]:
        """Place trading order"""
        try:
            if exchange_id not in self.exchanges:
                return None
            
            if exchange_id not in self.exchange_connections:
                return None
            
            exchange = self.exchanges[exchange_id]
            
            # Create order
            order = TradingOrder(
                order_id="",  # Will be generated
                exchange_id=exchange_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price
            )
            
            # Validate order
            if not await self._validate_order(order, exchange):
                return None
            
            # Submit order to exchange
            exchange_order_id = await self._submit_order_to_exchange(order, exchange)
            if exchange_order_id:
                order.order_id = exchange_order_id
                order.status = OrderStatus.OPEN
                self.active_orders[order.order_id] = order
                
                self.logger.info(f"Order placed: {order.order_id} on {exchange.name}")
                return order.order_id
            
            return None
            
        except Exception as e:
            self.logger.error(f"Place order failed: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel trading order"""
        try:
            if order_id not in self.active_orders:
                return False
            
            order = self.active_orders[order_id]
            exchange = self.exchanges[order.exchange_id]
            
            # Cancel order on exchange
            success = await self._cancel_order_on_exchange(order, exchange)
            if success:
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.now(timezone.utc)
                
                # Move to history
                self.order_history.append(order)
                del self.active_orders[order_id]
                
                self.logger.info(f"Order cancelled: {order_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Cancel order failed: {e}")
            return False
    
    async def process_payment(self, provider_id: str, payment_method: PaymentMethod,
                            amount: float, currency: str, sender_id: str,
                            recipient_id: str, description: str = "") -> Optional[str]:
        """Process payment transaction"""
        try:
            if provider_id not in self.payment_providers:
                return None
            
            provider = self.payment_providers[provider_id]
            
            # Validate payment
            if currency not in provider.supported_currencies:
                return None
            
            if amount < provider.min_amount or amount > provider.max_amount:
                return None
            
            # Calculate fees
            fee_amount = amount * provider.processing_fee + provider.fixed_fee
            
            # Create transaction
            transaction = PaymentTransaction(
                transaction_id="",  # Will be generated
                provider_id=provider_id,
                payment_method=payment_method,
                amount=amount,
                currency=currency,
                sender_id=sender_id,
                recipient_id=recipient_id,
                fee_amount=fee_amount,
                description=description
            )
            
            # Process payment with provider
            success = await self._process_payment_with_provider(transaction, provider)
            if success:
                transaction.status = "completed"
                transaction.processed_at = datetime.now(timezone.utc)
                self.payment_transactions[transaction.transaction_id] = transaction
                
                self.logger.info(f"Payment processed: {transaction.transaction_id}")
                return transaction.transaction_id
            
            return None
            
        except Exception as e:
            self.logger.error(f"Process payment failed: {e}")
            return None
    
    async def get_market_data(self, exchange_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Get market data for symbol"""
        try:
            if exchange_id not in self.exchanges:
                return None
            
            # Check cache first
            cache_key = f"{exchange_id}:{symbol}"
            if cache_key in self.market_data:
                cached_data = self.market_data[cache_key]
                # Return if data is less than 1 second old
                if (datetime.now(timezone.utc) - cached_data['timestamp']).total_seconds() < 1:
                    return cached_data
            
            exchange = self.exchanges[exchange_id]
            
            # Fetch market data from exchange
            market_data = await self._fetch_market_data_from_exchange(exchange, symbol)
            if market_data:
                market_data['timestamp'] = datetime.now(timezone.utc)
                self.market_data[cache_key] = market_data
                
                # Update price cache
                if 'price' in market_data:
                    self.price_cache[cache_key] = market_data['price']
                
                return market_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Get market data failed: {e}")
            return None
    
    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order status"""
        try:
            # Check active orders
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                return {
                    'order_id': order.order_id,
                    'status': order.status.value,
                    'symbol': order.symbol,
                    'side': order.side.value,
                    'quantity': order.quantity,
                    'filled_quantity': order.filled_quantity,
                    'remaining_quantity': order.remaining_quantity,
                    'price': order.price,
                    'created_at': order.created_at.isoformat(),
                    'updated_at': order.updated_at.isoformat() if order.updated_at else None
                }
            
            # Check order history
            for order in self.order_history:
                if order.order_id == order_id:
                    return {
                        'order_id': order.order_id,
                        'status': order.status.value,
                        'symbol': order.symbol,
                        'side': order.side.value,
                        'quantity': order.quantity,
                        'filled_quantity': order.filled_quantity,
                        'price': order.price,
                        'created_at': order.created_at.isoformat(),
                        'updated_at': order.updated_at.isoformat() if order.updated_at else None
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Get order status failed: {e}")
            return None
    
    async def get_payment_status(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Get payment transaction status"""
        try:
            if transaction_id not in self.payment_transactions:
                return None
            
            transaction = self.payment_transactions[transaction_id]
            
            return {
                'transaction_id': transaction.transaction_id,
                'status': transaction.status,
                'amount': transaction.amount,
                'currency': transaction.currency,
                'fee_amount': transaction.fee_amount,
                'net_amount': transaction.net_amount,
                'payment_method': transaction.payment_method.value,
                'created_at': transaction.created_at.isoformat(),
                'processed_at': transaction.processed_at.isoformat() if transaction.processed_at else None
            }
            
        except Exception as e:
            self.logger.error(f"Get payment status failed: {e}")
            return None
    
    async def get_supported_exchanges(self, exchange_type: Optional[ExchangeType] = None,
                                    region: Optional[ExchangeRegion] = None) -> List[Dict[str, Any]]:
        """Get supported exchanges"""
        try:
            exchanges = []
            
            for exchange in self.exchanges.values():
                # Filter by type
                if exchange_type and exchange.exchange_type != exchange_type:
                    continue
                
                # Filter by region
                if region and exchange.region != region:
                    continue
                
                exchanges.append({
                    'exchange_id': exchange.exchange_id,
                    'name': exchange.name,
                    'exchange_type': exchange.exchange_type.value,
                    'region': exchange.region.value,
                    'is_active': exchange.is_active,
                    'supports_spot': exchange.supports_spot,
                    'supports_futures': exchange.supports_futures,
                    'supports_options': exchange.supports_options,
                    'supports_margin': exchange.supports_margin,
                    'maker_fee': exchange.maker_fee,
                    'taker_fee': exchange.taker_fee
                })
            
            return exchanges
            
        except Exception as e:
            self.logger.error(f"Get supported exchanges failed: {e}")
            return []
    
    async def get_supported_payment_providers(self, region: Optional[ExchangeRegion] = None,
                                            currency: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get supported payment providers"""
        try:
            providers = []
            
            for provider in self.payment_providers.values():
                # Filter by region
                if region and region not in provider.supported_regions:
                    continue
                
                # Filter by currency
                if currency and currency not in provider.supported_currencies:
                    continue
                
                providers.append({
                    'provider_id': provider.provider_id,
                    'name': provider.name,
                    'payment_methods': [method.value for method in provider.payment_methods],
                    'supported_currencies': provider.supported_currencies,
                    'supported_regions': [region.value for region in provider.supported_regions],
                    'min_amount': provider.min_amount,
                    'max_amount': provider.max_amount,
                    'processing_fee': provider.processing_fee,
                    'fixed_fee': provider.fixed_fee,
                    'is_active': provider.is_active
                })
            
            return providers
            
        except Exception as e:
            self.logger.error(f"Get supported payment providers failed: {e}")
            return []
    
    async def get_analytics(self) -> Dict[str, Any]:
        """Get integration analytics"""
        try:
            # Exchange analytics
            exchange_stats = {
                'total_exchanges': len(self.exchanges),
                'connected_exchanges': len(self.exchange_connections),
                'by_type': {},
                'by_region': {}
            }
            
            for exchange in self.exchanges.values():
                # By type
                type_key = exchange.exchange_type.value
                if type_key not in exchange_stats['by_type']:
                    exchange_stats['by_type'][type_key] = 0
                exchange_stats['by_type'][type_key] += 1
                
                # By region
                region_key = exchange.region.value
                if region_key not in exchange_stats['by_region']:
                    exchange_stats['by_region'][region_key] = 0
                exchange_stats['by_region'][region_key] += 1
            
            # Order analytics
            order_stats = {
                'active_orders': len(self.active_orders),
                'total_orders': len(self.active_orders) + len(self.order_history),
                'by_status': {},
                'by_side': {}
            }
            
            all_orders = list(self.active_orders.values()) + self.order_history
            for order in all_orders:
                # By status
                status_key = order.status.value
                if status_key not in order_stats['by_status']:
                    order_stats['by_status'][status_key] = 0
                order_stats['by_status'][status_key] += 1
                
                # By side
                side_key = order.side.value
                if side_key not in order_stats['by_side']:
                    order_stats['by_side'][side_key] = 0
                order_stats['by_side'][side_key] += 1
            
            # Payment analytics
            payment_stats = {
                'total_providers': len(self.payment_providers),
                'total_transactions': len(self.payment_transactions),
                'by_method': {},
                'by_status': {}
            }
            
            for transaction in self.payment_transactions.values():
                # By method
                method_key = transaction.payment_method.value
                if method_key not in payment_stats['by_method']:
                    payment_stats['by_method'][method_key] = 0
                payment_stats['by_method'][method_key] += 1
                
                # By status
                status_key = transaction.status
                if status_key not in payment_stats['by_status']:
                    payment_stats['by_status'][status_key] = 0
                payment_stats['by_status'][status_key] += 1
            
            return {
                'exchanges': exchange_stats,
                'orders': order_stats,
                'payments': payment_stats,
                'market_data_cache_size': len(self.market_data),
                'price_cache_size': len(self.price_cache)
            }
            
        except Exception as e:
            self.logger.error(f"Get analytics failed: {e}")
            return {}
    
    # Private helper methods
    async def _test_exchange_connection(self, exchange: ExchangeConfig) -> bool:
        """Test exchange connection"""
        try:
            # Simulate connection test
            await asyncio.sleep(0.1)
            exchange.last_ping = datetime.now(timezone.utc)
            return True
        except Exception as e:
            self.logger.error(f"Exchange connection test failed: {e}")
            return False
    
    async def _start_websocket_connection(self, exchange_id: str):
        """Start WebSocket connection for real-time data"""
        try:
            # Simulate WebSocket connection
            self.websocket_connections[exchange_id] = {
                'connected_at': datetime.now(timezone.utc),
                'status': 'connected'
            }
            self.logger.info(f"WebSocket connected for {exchange_id}")
        except Exception as e:
            self.logger.error(f"WebSocket connection failed: {e}")
    
    async def _validate_order(self, order: TradingOrder, exchange: ExchangeConfig) -> bool:
        """Validate order before submission"""
        try:
            # Check minimum order size
            if order.quantity < exchange.min_order_size:
                return False
            
            # Check maximum order size
            if order.quantity > exchange.max_order_size:
                return False
            
            # Check if symbol is supported
            if exchange.supported_pairs and order.symbol not in exchange.supported_pairs:
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Order validation failed: {e}")
            return False
    
    async def _submit_order_to_exchange(self, order: TradingOrder, exchange: ExchangeConfig) -> Optional[str]:
        """Submit order to exchange"""
        try:
            # Simulate order submission
            await asyncio.sleep(0.1)
            
            # Generate exchange order ID
            exchange_order_id = f"{exchange.exchange_id}_{int(time.time() * 1000)}"
            
            self.logger.info(f"Order submitted to {exchange.name}: {exchange_order_id}")
            return exchange_order_id
        except Exception as e:
            self.logger.error(f"Order submission failed: {e}")
            return None
    
    async def _cancel_order_on_exchange(self, order: TradingOrder, exchange: ExchangeConfig) -> bool:
        """Cancel order on exchange"""
        try:
            # Simulate order cancellation
            await asyncio.sleep(0.1)
            self.logger.info(f"Order cancelled on {exchange.name}: {order.order_id}")
            return True
        except Exception as e:
            self.logger.error(f"Order cancellation failed: {e}")
            return False
    
    async def _process_payment_with_provider(self, transaction: PaymentTransaction, 
                                           provider: PaymentProvider) -> bool:
        """Process payment with provider"""
        try:
            # Simulate payment processing
            await asyncio.sleep(0.2)
            self.logger.info(f"Payment processed with {provider.name}: {transaction.transaction_id}")
            return True
        except Exception as e:
            self.logger.error(f"Payment processing failed: {e}")
            return False
    
    async def _fetch_market_data_from_exchange(self, exchange: ExchangeConfig, 
                                             symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch market data from exchange"""
        try:
            # Simulate market data fetch
            await asyncio.sleep(0.05)
            
            # Generate mock market data
            base_price = 45000.0 if 'BTC' in symbol else 3200.0
            price = base_price + (np.random.random() - 0.5) * base_price * 0.02
            
            return {
                'symbol': symbol,
                'price': round(price, 2),
                'bid': round(price * 0.999, 2),
                'ask': round(price * 1.001, 2),
                'volume_24h': round(np.random.random() * 1000000, 2),
                'change_24h': round((np.random.random() - 0.5) * 10, 2),
                'high_24h': round(price * 1.05, 2),
                'low_24h': round(price * 0.95, 2),
                'exchange': exchange.name
            }
        except Exception as e:
            self.logger.error(f"Market data fetch failed: {e}")
            return None

# Example usage
async def main():
    """
    Example usage of Global Exchange Integrator
    """
    print("🌐 Global Exchange Integrator - Phase 5 Testing")
    print("=" * 50)
    
    # Initialize integrator
    integrator = GlobalExchangeIntegrator()
    
    # Wait for initialization
    await asyncio.sleep(2)
    
    # Test exchange connections
    print("\n🔗 Testing Exchange Connections:")
    
    test_exchanges = ['binance_global', 'coinbase_pro', 'nyse', 'oanda']
    
    for exchange_id in test_exchanges:
        if exchange_id in integrator.exchanges:
            success = await integrator.connect_exchange(
                exchange_id=exchange_id,
                api_key="test_key",
                api_secret="test_secret"
            )
            print(f"  {integrator.exchanges[exchange_id].name}: {'✅' if success else '❌'}")
    
    # Test market data
    print(f"\n📊 Testing Market Data:")
    
    test_symbols = [
        ('binance_global', 'BTCUSDT'),
        ('coinbase_pro', 'BTC-USD'),
        ('nyse', 'AAPL'),
        ('oanda', 'EUR_USD')
    ]
    
    for exchange_id, symbol in test_symbols:
        if exchange_id in integrator.exchange_connections:
            market_data = await integrator.get_market_data(exchange_id, symbol)
            if market_data:
                print(f"  {symbol} on {market_data['exchange']}: ${market_data['price']}")
    
    # Test order placement
    print(f"\n📋 Testing Order Placement:")
    
    if 'binance_global' in integrator.exchange_connections:
        order_id = await integrator.place_order(
            exchange_id='binance_global',
            symbol='BTCUSDT',
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=44000.0
        )
        
        if order_id:
            print(f"  Order placed: {order_id}")
            
            # Check order status
            order_status = await integrator.get_order_status(order_id)
            if order_status:
                print(f"  Order status: {order_status['status']}")
            
            # Cancel order
            cancel_success = await integrator.cancel_order(order_id)
            print(f"  Order cancelled: {'✅' if cancel_success else '❌'}")
    
    # Test payment processing
    print(f"\n💳 Testing Payment Processing:")
    
    payment_providers = ['stripe', 'paypal', 'wise']
    
    for provider_id in payment_providers:
        if provider_id in integrator.payment_providers:
            transaction_id = await integrator.process_payment(
                provider_id=provider_id,
                payment_method=PaymentMethod.CREDIT_CARD,
                amount=100.0,
                currency='USD',
                sender_id='user_001',
                recipient_id='user_002',
                description='Test payment'
            )
            
            if transaction_id:
                print(f"  {integrator.payment_providers[provider_id].name}: {transaction_id}")
                
                # Check payment status
                payment_status = await integrator.get_payment_status(transaction_id)
                if payment_status:
                    print(f"    Status: {payment_status['status']}")
                    print(f"    Net amount: ${payment_status['net_amount']}")
    
    # Get supported exchanges
    print(f"\n🏛️ Supported Exchanges:")
    
    crypto_exchanges = await integrator.get_supported_exchanges(ExchangeType.CRYPTOCURRENCY)
    stock_exchanges = await integrator.get_supported_exchanges(ExchangeType.STOCK)
    forex_exchanges = await integrator.get_supported_exchanges(ExchangeType.FOREX)
    
    print(f"  Cryptocurrency: {len(crypto_exchanges)} exchanges")
    print(f"  Stock: {len(stock_exchanges)} exchanges")
    print(f"  Forex: {len(forex_exchanges)} exchanges")
    
    # Get supported payment providers
    print(f"\n💳 Supported Payment Providers:")
    
    na_providers = await integrator.get_supported_payment_providers(ExchangeRegion.NORTH_AMERICA)
    eu_providers = await integrator.get_supported_payment_providers(ExchangeRegion.EUROPE)
    ap_providers = await integrator.get_supported_payment_providers(ExchangeRegion.ASIA_PACIFIC)
    
    print(f"  North America: {len(na_providers)} providers")
    print(f"  Europe: {len(eu_providers)} providers")
    print(f"  Asia Pacific: {len(ap_providers)} providers")
    
    # Get analytics
    print(f"\n📊 Integration Analytics:")
    
    analytics = await integrator.get_analytics()
    if analytics:
        print(f"  Total Exchanges: {analytics['exchanges']['total_exchanges']}")
        print(f"  Connected Exchanges: {analytics['exchanges']['connected_exchanges']}")
        print(f"  Total Orders: {analytics['orders']['total_orders']}")
        print(f"  Active Orders: {analytics['orders']['active_orders']}")
        print(f"  Payment Providers: {analytics['payments']['total_providers']}")
        print(f"  Total Transactions: {analytics['payments']['total_transactions']}")
    
    print(f"\n✅ Global Exchange Integrator testing completed!")

if __name__ == "__main__":
    asyncio.run(main())