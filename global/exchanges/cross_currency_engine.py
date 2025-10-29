"""
Cross-Currency Trading Engine for Phase 5
Handles multi-currency operations and automatic conversions
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
from decimal import Decimal, ROUND_HALF_UP
import warnings
warnings.filterwarnings('ignore')

class CurrencyType(Enum):
    FIAT = "fiat"
    CRYPTOCURRENCY = "cryptocurrency"
    STABLECOIN = "stablecoin"
    COMMODITY = "commodity"

class ConversionMethod(Enum):
    DIRECT = "direct"
    TRIANGULAR = "triangular"
    MULTI_HOP = "multi_hop"
    CROSS_EXCHANGE = "cross_exchange"

class ConversionStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Currency:
    """Currency definition"""
    code: str
    name: str
    currency_type: CurrencyType
    
    # Precision and formatting
    decimal_places: int = 2
    symbol: str = ""
    
    # Trading properties
    is_tradable: bool = True
    min_trade_amount: float = 0.01
    max_trade_amount: float = 1000000.0
    
    # Supported exchanges
    supported_exchanges: List[str] = field(default_factory=list)
    
    # Metadata
    country_code: str = ""
    region: str = ""
    
    def __post_init__(self):
        if not self.symbol:
            self.symbol = self.code

@dataclass
class ExchangeRate:
    """Exchange rate between two currencies"""
    base_currency: str
    quote_currency: str
    rate: float
    
    # Rate metadata
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    
    # Source information
    source_exchange: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Validity
    is_valid: bool = True
    expires_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.bid == 0.0:
            self.bid = self.rate * 0.999
        if self.ask == 0.0:
            self.ask = self.rate * 1.001
        if self.spread == 0.0:
            self.spread = self.ask - self.bid
        if not self.expires_at:
            self.expires_at = self.timestamp + timedelta(minutes=1)

@dataclass
class ConversionPath:
    """Currency conversion path"""
    path_id: str
    from_currency: str
    to_currency: str
    
    # Path details
    conversion_steps: List[Tuple[str, str, str]] = field(default_factory=list)  # (from, to, exchange)
    total_rate: float = 0.0
    total_fee: float = 0.0
    
    # Path metadata
    method: ConversionMethod = ConversionMethod.DIRECT
    estimated_time: float = 0.0  # seconds
    confidence_score: float = 0.0
    
    # Costs
    gas_cost: float = 0.0
    slippage: float = 0.0
    
    def __post_init__(self):
        if not self.path_id:
            self.path_id = str(uuid.uuid4())

@dataclass
class ConversionRequest:
    """Currency conversion request"""
    request_id: str
    from_currency: str
    to_currency: str
    amount: float
    
    # Conversion preferences
    max_slippage: float = 0.01  # 1%
    max_fee: float = 0.005  # 0.5%
    preferred_exchanges: List[str] = field(default_factory=list)
    
    # Status
    status: ConversionStatus = ConversionStatus.PENDING
    
    # Results
    converted_amount: float = 0.0
    actual_rate: float = 0.0
    actual_fee: float = 0.0
    execution_time: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    
    # Path used
    conversion_path: Optional[ConversionPath] = None
    
    def __post_init__(self):
        if not self.request_id:
            self.request_id = str(uuid.uuid4())

class CrossCurrencyEngine:
    """Cross-currency trading engine"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Currency definitions
        self.currencies: Dict[str, Currency] = {}
        
        # Exchange rates
        self.exchange_rates: Dict[str, ExchangeRate] = {}  # key: "BASE_QUOTE"
        self.rate_history: List[ExchangeRate] = []
        
        # Conversion paths
        self.conversion_paths: Dict[str, List[ConversionPath]] = {}  # key: "FROM_TO"
        
        # Active conversions
        self.active_conversions: Dict[str, ConversionRequest] = {}
        self.conversion_history: List[ConversionRequest] = []
        
        # Rate providers
        self.rate_providers: Dict[str, Any] = {}
        
        # Initialize currencies and rates
        asyncio.create_task(self._initialize_currencies())
        asyncio.create_task(self._start_rate_updates())
    
    async def _initialize_currencies(self):
        """Initialize supported currencies"""
        try:
            # Major fiat currencies
            fiat_currencies = [
                {'code': 'USD', 'name': 'US Dollar', 'symbol': '$', 'country_code': 'US', 'region': 'North America'},
                {'code': 'EUR', 'name': 'Euro', 'symbol': '€', 'country_code': 'EU', 'region': 'Europe'},
                {'code': 'GBP', 'name': 'British Pound', 'symbol': '£', 'country_code': 'GB', 'region': 'Europe'},
                {'code': 'JPY', 'name': 'Japanese Yen', 'symbol': '¥', 'country_code': 'JP', 'region': 'Asia Pacific', 'decimal_places': 0},
                {'code': 'CAD', 'name': 'Canadian Dollar', 'symbol': 'C$', 'country_code': 'CA', 'region': 'North America'},
                {'code': 'AUD', 'name': 'Australian Dollar', 'symbol': 'A$', 'country_code': 'AU', 'region': 'Asia Pacific'},
                {'code': 'CHF', 'name': 'Swiss Franc', 'symbol': 'CHF', 'country_code': 'CH', 'region': 'Europe'},
                {'code': 'CNY', 'name': 'Chinese Yuan', 'symbol': '¥', 'country_code': 'CN', 'region': 'Asia Pacific'},
                {'code': 'INR', 'name': 'Indian Rupee', 'symbol': '₹', 'country_code': 'IN', 'region': 'Asia Pacific'},
                {'code': 'BRL', 'name': 'Brazilian Real', 'symbol': 'R$', 'country_code': 'BR', 'region': 'Latin America'},
                {'code': 'KRW', 'name': 'South Korean Won', 'symbol': '₩', 'country_code': 'KR', 'region': 'Asia Pacific', 'decimal_places': 0},
                {'code': 'SGD', 'name': 'Singapore Dollar', 'symbol': 'S$', 'country_code': 'SG', 'region': 'Asia Pacific'},
                {'code': 'HKD', 'name': 'Hong Kong Dollar', 'symbol': 'HK$', 'country_code': 'HK', 'region': 'Asia Pacific'},
                {'code': 'MXN', 'name': 'Mexican Peso', 'symbol': '$', 'country_code': 'MX', 'region': 'Latin America'},
                {'code': 'RUB', 'name': 'Russian Ruble', 'symbol': '₽', 'country_code': 'RU', 'region': 'Europe'},
                {'code': 'ZAR', 'name': 'South African Rand', 'symbol': 'R', 'country_code': 'ZA', 'region': 'Africa'},
                {'code': 'TRY', 'name': 'Turkish Lira', 'symbol': '₺', 'country_code': 'TR', 'region': 'Europe'},
                {'code': 'AED', 'name': 'UAE Dirham', 'symbol': 'د.إ', 'country_code': 'AE', 'region': 'Middle East'},
                {'code': 'SAR', 'name': 'Saudi Riyal', 'symbol': '﷼', 'country_code': 'SA', 'region': 'Middle East'},
                {'code': 'THB', 'name': 'Thai Baht', 'symbol': '฿', 'country_code': 'TH', 'region': 'Asia Pacific'}
            ]
            
            for currency_data in fiat_currencies:
                currency = Currency(
                    currency_type=CurrencyType.FIAT,
                    **currency_data
                )
                self.currencies[currency.code] = currency
            
            # Major cryptocurrencies
            crypto_currencies = [
                {'code': 'BTC', 'name': 'Bitcoin', 'symbol': '₿', 'decimal_places': 8},
                {'code': 'ETH', 'name': 'Ethereum', 'symbol': 'Ξ', 'decimal_places': 8},
                {'code': 'BNB', 'name': 'Binance Coin', 'symbol': 'BNB', 'decimal_places': 8},
                {'code': 'ADA', 'name': 'Cardano', 'symbol': 'ADA', 'decimal_places': 6},
                {'code': 'SOL', 'name': 'Solana', 'symbol': 'SOL', 'decimal_places': 6},
                {'code': 'DOT', 'name': 'Polkadot', 'symbol': 'DOT', 'decimal_places': 6},
                {'code': 'AVAX', 'name': 'Avalanche', 'symbol': 'AVAX', 'decimal_places': 6},
                {'code': 'MATIC', 'name': 'Polygon', 'symbol': 'MATIC', 'decimal_places': 6},
                {'code': 'LINK', 'name': 'Chainlink', 'symbol': 'LINK', 'decimal_places': 6},
                {'code': 'UNI', 'name': 'Uniswap', 'symbol': 'UNI', 'decimal_places': 6}
            ]
            
            for currency_data in crypto_currencies:
                currency = Currency(
                    currency_type=CurrencyType.CRYPTOCURRENCY,
                    region='Global',
                    **currency_data
                )
                self.currencies[currency.code] = currency
            
            # Stablecoins
            stablecoin_currencies = [
                {'code': 'USDT', 'name': 'Tether', 'symbol': 'USDT', 'decimal_places': 6},
                {'code': 'USDC', 'name': 'USD Coin', 'symbol': 'USDC', 'decimal_places': 6},
                {'code': 'BUSD', 'name': 'Binance USD', 'symbol': 'BUSD', 'decimal_places': 6},
                {'code': 'DAI', 'name': 'Dai', 'symbol': 'DAI', 'decimal_places': 6},
                {'code': 'TUSD', 'name': 'TrueUSD', 'symbol': 'TUSD', 'decimal_places': 6}
            ]
            
            for currency_data in stablecoin_currencies:
                currency = Currency(
                    currency_type=CurrencyType.STABLECOIN,
                    region='Global',
                    **currency_data
                )
                self.currencies[currency.code] = currency
            
            # Commodities
            commodity_currencies = [
                {'code': 'XAU', 'name': 'Gold', 'symbol': 'Au', 'decimal_places': 4},
                {'code': 'XAG', 'name': 'Silver', 'symbol': 'Ag', 'decimal_places': 4},
                {'code': 'XPT', 'name': 'Platinum', 'symbol': 'Pt', 'decimal_places': 4},
                {'code': 'XPD', 'name': 'Palladium', 'symbol': 'Pd', 'decimal_places': 4}
            ]
            
            for currency_data in commodity_currencies:
                currency = Currency(
                    currency_type=CurrencyType.COMMODITY,
                    region='Global',
                    **currency_data
                )
                self.currencies[currency.code] = currency
            
            self.logger.info(f"Initialized {len(self.currencies)} currencies")
            
        except Exception as e:
            self.logger.error(f"Currency initialization failed: {e}")
    
    async def _start_rate_updates(self):
        """Start real-time rate updates"""
        try:
            while True:
                await self._update_exchange_rates()
                await asyncio.sleep(5)  # Update every 5 seconds
        except Exception as e:
            self.logger.error(f"Rate updates failed: {e}")
    
    async def _update_exchange_rates(self):
        """Update exchange rates from various sources"""
        try:
            # Generate mock exchange rates for demonstration
            base_rates = {
                'EUR': 1.08,
                'GBP': 1.25,
                'JPY': 0.0067,
                'CAD': 0.74,
                'AUD': 0.65,
                'CHF': 1.10,
                'CNY': 0.14,
                'INR': 0.012,
                'BRL': 0.20,
                'KRW': 0.00075,
                'SGD': 0.74,
                'HKD': 0.13,
                'MXN': 0.058,
                'RUB': 0.011,
                'ZAR': 0.055,
                'TRY': 0.037,
                'AED': 0.27,
                'SAR': 0.27,
                'THB': 0.028,
                
                # Cryptocurrencies (in USD)
                'BTC': 45000 + np.random.normal(0, 1000),
                'ETH': 3200 + np.random.normal(0, 100),
                'BNB': 320 + np.random.normal(0, 10),
                'ADA': 0.45 + np.random.normal(0, 0.02),
                'SOL': 95 + np.random.normal(0, 5),
                'DOT': 6.5 + np.random.normal(0, 0.3),
                'AVAX': 38 + np.random.normal(0, 2),
                'MATIC': 0.85 + np.random.normal(0, 0.05),
                'LINK': 14.5 + np.random.normal(0, 0.7),
                'UNI': 6.2 + np.random.normal(0, 0.3),
                
                # Stablecoins
                'USDT': 1.0 + np.random.normal(0, 0.001),
                'USDC': 1.0 + np.random.normal(0, 0.001),
                'BUSD': 1.0 + np.random.normal(0, 0.001),
                'DAI': 1.0 + np.random.normal(0, 0.002),
                'TUSD': 1.0 + np.random.normal(0, 0.001),
                
                # Commodities (per ounce in USD)
                'XAU': 2050 + np.random.normal(0, 20),
                'XAG': 24.5 + np.random.normal(0, 0.5),
                'XPT': 980 + np.random.normal(0, 15),
                'XPD': 1250 + np.random.normal(0, 25)
            }
            
            # Update USD-based rates
            for currency, rate in base_rates.items():
                if currency != 'USD':
                    rate_key = f"USD_{currency}"
                    
                    # Add some volatility
                    volatility = np.random.normal(0, 0.001)
                    adjusted_rate = rate * (1 + volatility)
                    
                    exchange_rate = ExchangeRate(
                        base_currency='USD',
                        quote_currency=currency,
                        rate=adjusted_rate,
                        source_exchange='aggregated'
                    )
                    
                    self.exchange_rates[rate_key] = exchange_rate
                    
                    # Also create reverse rate
                    reverse_rate_key = f"{currency}_USD"
                    reverse_rate = ExchangeRate(
                        base_currency=currency,
                        quote_currency='USD',
                        rate=1.0 / adjusted_rate,
                        source_exchange='aggregated'
                    )
                    
                    self.exchange_rates[reverse_rate_key] = reverse_rate
            
            # Generate cross rates for major pairs
            major_currencies = ['EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF']
            
            for base in major_currencies:
                for quote in major_currencies:
                    if base != quote:
                        base_usd_rate = base_rates.get(base, 1.0)
                        quote_usd_rate = base_rates.get(quote, 1.0)
                        
                        cross_rate = base_usd_rate / quote_usd_rate
                        rate_key = f"{base}_{quote}"
                        
                        exchange_rate = ExchangeRate(
                            base_currency=base,
                            quote_currency=quote,
                            rate=cross_rate,
                            source_exchange='calculated'
                        )
                        
                        self.exchange_rates[rate_key] = exchange_rate
            
        except Exception as e:
            self.logger.error(f"Exchange rate update failed: {e}")
    
    async def get_exchange_rate(self, from_currency: str, to_currency: str) -> Optional[float]:
        """Get exchange rate between two currencies"""
        try:
            if from_currency == to_currency:
                return 1.0
            
            # Direct rate
            rate_key = f"{from_currency}_{to_currency}"
            if rate_key in self.exchange_rates:
                rate = self.exchange_rates[rate_key]
                if rate.is_valid and rate.expires_at > datetime.now(timezone.utc):
                    return rate.rate
            
            # Reverse rate
            reverse_key = f"{to_currency}_{from_currency}"
            if reverse_key in self.exchange_rates:
                rate = self.exchange_rates[reverse_key]
                if rate.is_valid and rate.expires_at > datetime.now(timezone.utc):
                    return 1.0 / rate.rate
            
            # USD triangular arbitrage
            usd_from_key = f"USD_{from_currency}"
            usd_to_key = f"USD_{to_currency}"
            
            if usd_from_key in self.exchange_rates and usd_to_key in self.exchange_rates:
                usd_from_rate = self.exchange_rates[usd_from_key]
                usd_to_rate = self.exchange_rates[usd_to_key]
                
                if (usd_from_rate.is_valid and usd_to_rate.is_valid and
                    usd_from_rate.expires_at > datetime.now(timezone.utc) and
                    usd_to_rate.expires_at > datetime.now(timezone.utc)):
                    
                    return usd_to_rate.rate / usd_from_rate.rate
            
            return None
            
        except Exception as e:
            self.logger.error(f"Get exchange rate failed: {e}")
            return None
    
    async def find_conversion_paths(self, from_currency: str, to_currency: str,
                                  max_hops: int = 3) -> List[ConversionPath]:
        """Find optimal conversion paths between currencies"""
        try:
            paths = []
            
            # Direct conversion
            direct_rate = await self.get_exchange_rate(from_currency, to_currency)
            if direct_rate:
                path = ConversionPath(
                    path_id="",
                    from_currency=from_currency,
                    to_currency=to_currency,
                    conversion_steps=[(from_currency, to_currency, "direct")],
                    total_rate=direct_rate,
                    total_fee=0.001,  # 0.1% fee
                    method=ConversionMethod.DIRECT,
                    estimated_time=1.0,
                    confidence_score=0.95
                )
                paths.append(path)
            
            # USD triangular arbitrage
            if from_currency != 'USD' and to_currency != 'USD':
                usd_from_rate = await self.get_exchange_rate(from_currency, 'USD')
                usd_to_rate = await self.get_exchange_rate('USD', to_currency)
                
                if usd_from_rate and usd_to_rate:
                    triangular_rate = usd_from_rate * usd_to_rate
                    
                    path = ConversionPath(
                        path_id="",
                        from_currency=from_currency,
                        to_currency=to_currency,
                        conversion_steps=[
                            (from_currency, 'USD', "triangular"),
                            ('USD', to_currency, "triangular")
                        ],
                        total_rate=triangular_rate,
                        total_fee=0.002,  # 0.2% fee for two conversions
                        method=ConversionMethod.TRIANGULAR,
                        estimated_time=2.0,
                        confidence_score=0.90
                    )
                    paths.append(path)
            
            # Multi-hop through major currencies
            major_intermediates = ['EUR', 'GBP', 'JPY', 'BTC', 'ETH', 'USDT']
            
            for intermediate in major_intermediates:
                if intermediate != from_currency and intermediate != to_currency:
                    rate1 = await self.get_exchange_rate(from_currency, intermediate)
                    rate2 = await self.get_exchange_rate(intermediate, to_currency)
                    
                    if rate1 and rate2:
                        multi_hop_rate = rate1 * rate2
                        
                        path = ConversionPath(
                            path_id="",
                            from_currency=from_currency,
                            to_currency=to_currency,
                            conversion_steps=[
                                (from_currency, intermediate, "multi_hop"),
                                (intermediate, to_currency, "multi_hop")
                            ],
                            total_rate=multi_hop_rate,
                            total_fee=0.003,  # 0.3% fee for multi-hop
                            method=ConversionMethod.MULTI_HOP,
                            estimated_time=3.0,
                            confidence_score=0.85
                        )
                        paths.append(path)
            
            # Sort paths by total cost (rate * (1 + fee))
            paths.sort(key=lambda p: p.total_rate * (1 - p.total_fee), reverse=True)
            
            return paths[:5]  # Return top 5 paths
            
        except Exception as e:
            self.logger.error(f"Find conversion paths failed: {e}")
            return []
    
    async def convert_currency(self, from_currency: str, to_currency: str,
                             amount: float, max_slippage: float = 0.01,
                             preferred_exchanges: List[str] = None) -> Optional[str]:
        """Convert currency amount"""
        try:
            # Create conversion request
            request = ConversionRequest(
                request_id="",
                from_currency=from_currency,
                to_currency=to_currency,
                amount=amount,
                max_slippage=max_slippage,
                preferred_exchanges=preferred_exchanges or []
            )
            
            # Find conversion paths
            paths = await self.find_conversion_paths(from_currency, to_currency)
            if not paths:
                return None
            
            # Select best path
            best_path = paths[0]
            request.conversion_path = best_path
            request.status = ConversionStatus.IN_PROGRESS
            
            # Store active conversion
            self.active_conversions[request.request_id] = request
            
            # Execute conversion
            success = await self._execute_conversion(request)
            
            if success:
                request.status = ConversionStatus.COMPLETED
                request.completed_at = datetime.now(timezone.utc)
                
                # Move to history
                self.conversion_history.append(request)
                del self.active_conversions[request.request_id]
                
                return request.request_id
            else:
                request.status = ConversionStatus.FAILED
                del self.active_conversions[request.request_id]
                return None
            
        except Exception as e:
            self.logger.error(f"Currency conversion failed: {e}")
            return None
    
    async def _execute_conversion(self, request: ConversionRequest) -> bool:
        """Execute currency conversion"""
        try:
            start_time = time.time()
            
            if not request.conversion_path:
                return False
            
            path = request.conversion_path
            current_amount = request.amount
            
            # Execute each step in the conversion path
            for step_from, step_to, exchange in path.conversion_steps:
                # Get current rate
                rate = await self.get_exchange_rate(step_from, step_to)
                if not rate:
                    return False
                
                # Apply slippage
                slippage = np.random.uniform(0, request.max_slippage)
                effective_rate = rate * (1 - slippage)
                
                # Convert amount
                current_amount = current_amount * effective_rate
                
                # Apply fees
                fee_rate = path.total_fee / len(path.conversion_steps)
                fee_amount = current_amount * fee_rate
                current_amount -= fee_amount
                
                # Simulate execution delay
                await asyncio.sleep(0.1)
            
            # Update request with results
            request.converted_amount = current_amount
            request.actual_rate = current_amount / request.amount
            request.actual_fee = request.amount - current_amount / path.total_rate
            request.execution_time = time.time() - start_time
            
            return True
            
        except Exception as e:
            self.logger.error(f"Conversion execution failed: {e}")
            return False
    
    async def get_conversion_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get conversion request status"""
        try:
            # Check active conversions
            if request_id in self.active_conversions:
                request = self.active_conversions[request_id]
                return self._format_conversion_status(request)
            
            # Check conversion history
            for request in self.conversion_history:
                if request.request_id == request_id:
                    return self._format_conversion_status(request)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Get conversion status failed: {e}")
            return None
    
    def _format_conversion_status(self, request: ConversionRequest) -> Dict[str, Any]:
        """Format conversion status for response"""
        return {
            'request_id': request.request_id,
            'status': request.status.value,
            'from_currency': request.from_currency,
            'to_currency': request.to_currency,
            'amount': request.amount,
            'converted_amount': request.converted_amount,
            'actual_rate': request.actual_rate,
            'actual_fee': request.actual_fee,
            'execution_time': request.execution_time,
            'created_at': request.created_at.isoformat(),
            'completed_at': request.completed_at.isoformat() if request.completed_at else None,
            'conversion_path': {
                'method': request.conversion_path.method.value,
                'steps': request.conversion_path.conversion_steps,
                'total_rate': request.conversion_path.total_rate,
                'total_fee': request.conversion_path.total_fee
            } if request.conversion_path else None
        }
    
    async def get_supported_currencies(self, currency_type: Optional[CurrencyType] = None,
                                     region: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get supported currencies"""
        try:
            currencies = []
            
            for currency in self.currencies.values():
                # Filter by type
                if currency_type and currency.currency_type != currency_type:
                    continue
                
                # Filter by region
                if region and currency.region.lower() != region.lower():
                    continue
                
                currencies.append({
                    'code': currency.code,
                    'name': currency.name,
                    'symbol': currency.symbol,
                    'currency_type': currency.currency_type.value,
                    'decimal_places': currency.decimal_places,
                    'region': currency.region,
                    'country_code': currency.country_code,
                    'is_tradable': currency.is_tradable,
                    'min_trade_amount': currency.min_trade_amount,
                    'max_trade_amount': currency.max_trade_amount
                })
            
            return currencies
            
        except Exception as e:
            self.logger.error(f"Get supported currencies failed: {e}")
            return []
    
    async def get_currency_pairs(self, base_currency: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get available currency pairs"""
        try:
            pairs = []
            
            for rate_key, rate in self.exchange_rates.items():
                if not rate.is_valid or rate.expires_at <= datetime.now(timezone.utc):
                    continue
                
                # Filter by base currency
                if base_currency and rate.base_currency != base_currency:
                    continue
                
                pairs.append({
                    'pair': f"{rate.base_currency}/{rate.quote_currency}",
                    'base_currency': rate.base_currency,
                    'quote_currency': rate.quote_currency,
                    'rate': rate.rate,
                    'bid': rate.bid,
                    'ask': rate.ask,
                    'spread': rate.spread,
                    'source_exchange': rate.source_exchange,
                    'timestamp': rate.timestamp.isoformat()
                })
            
            return pairs
            
        except Exception as e:
            self.logger.error(f"Get currency pairs failed: {e}")
            return []
    
    async def calculate_conversion_cost(self, from_currency: str, to_currency: str,
                                      amount: float) -> Optional[Dict[str, Any]]:
        """Calculate conversion cost and fees"""
        try:
            # Find conversion paths
            paths = await self.find_conversion_paths(from_currency, to_currency)
            if not paths:
                return None
            
            cost_analysis = []
            
            for path in paths:
                # Calculate total cost
                gross_amount = amount * path.total_rate
                fee_amount = gross_amount * path.total_fee
                net_amount = gross_amount - fee_amount
                
                cost_analysis.append({
                    'method': path.method.value,
                    'steps': len(path.conversion_steps),
                    'gross_amount': gross_amount,
                    'fee_amount': fee_amount,
                    'net_amount': net_amount,
                    'effective_rate': net_amount / amount,
                    'total_fee_percentage': path.total_fee * 100,
                    'estimated_time': path.estimated_time,
                    'confidence_score': path.confidence_score,
                    'conversion_steps': path.conversion_steps
                })
            
            return {
                'from_currency': from_currency,
                'to_currency': to_currency,
                'amount': amount,
                'paths': cost_analysis,
                'recommended_path': cost_analysis[0] if cost_analysis else None
            }
            
        except Exception as e:
            self.logger.error(f"Calculate conversion cost failed: {e}")
            return None
    
    async def get_analytics(self) -> Dict[str, Any]:
        """Get cross-currency analytics"""
        try:
            # Currency analytics
            currency_stats = {
                'total_currencies': len(self.currencies),
                'by_type': {},
                'by_region': {}
            }
            
            for currency in self.currencies.values():
                # By type
                type_key = currency.currency_type.value
                if type_key not in currency_stats['by_type']:
                    currency_stats['by_type'][type_key] = 0
                currency_stats['by_type'][type_key] += 1
                
                # By region
                region_key = currency.region
                if region_key not in currency_stats['by_region']:
                    currency_stats['by_region'][region_key] = 0
                currency_stats['by_region'][region_key] += 1
            
            # Exchange rate analytics
            rate_stats = {
                'total_rates': len(self.exchange_rates),
                'valid_rates': sum(1 for rate in self.exchange_rates.values() 
                                 if rate.is_valid and rate.expires_at > datetime.now(timezone.utc)),
                'by_source': {}
            }
            
            for rate in self.exchange_rates.values():
                source = rate.source_exchange
                if source not in rate_stats['by_source']:
                    rate_stats['by_source'][source] = 0
                rate_stats['by_source'][source] += 1
            
            # Conversion analytics
            conversion_stats = {
                'active_conversions': len(self.active_conversions),
                'total_conversions': len(self.active_conversions) + len(self.conversion_history),
                'by_status': {},
                'by_method': {}
            }
            
            all_conversions = list(self.active_conversions.values()) + self.conversion_history
            for conversion in all_conversions:
                # By status
                status_key = conversion.status.value
                if status_key not in conversion_stats['by_status']:
                    conversion_stats['by_status'][status_key] = 0
                conversion_stats['by_status'][status_key] += 1
                
                # By method
                if conversion.conversion_path:
                    method_key = conversion.conversion_path.method.value
                    if method_key not in conversion_stats['by_method']:
                        conversion_stats['by_method'][method_key] = 0
                    conversion_stats['by_method'][method_key] += 1
            
            return {
                'currencies': currency_stats,
                'exchange_rates': rate_stats,
                'conversions': conversion_stats
            }
            
        except Exception as e:
            self.logger.error(f"Get analytics failed: {e}")
            return {}

# Example usage
async def main():
    """
    Example usage of Cross-Currency Engine
    """
    print("💱 Cross-Currency Engine - Phase 5 Testing")
    print("=" * 50)
    
    # Initialize engine
    engine = CrossCurrencyEngine()
    
    # Wait for initialization
    await asyncio.sleep(3)
    
    # Test exchange rates
    print("\n💹 Testing Exchange Rates:")
    
    test_pairs = [
        ('USD', 'EUR'),
        ('EUR', 'GBP'),
        ('USD', 'BTC'),
        ('BTC', 'ETH'),
        ('USD', 'JPY'),
        ('GBP', 'CAD')
    ]
    
    for base, quote in test_pairs:
        rate = await engine.get_exchange_rate(base, quote)
        if rate:
            print(f"  {base}/{quote}: {rate:.6f}")
    
    # Test conversion paths
    print(f"\n🛤️ Testing Conversion Paths:")
    
    test_conversions = [
        ('USD', 'EUR'),
        ('EUR', 'BTC'),
        ('GBP', 'JPY'),
        ('BTC', 'USDT')
    ]
    
    for from_curr, to_curr in test_conversions:
        paths = await engine.find_conversion_paths(from_curr, to_curr)
        print(f"  {from_curr} → {to_curr}: {len(paths)} paths found")
        
        if paths:
            best_path = paths[0]
            print(f"    Best: {best_path.method.value} (rate: {best_path.total_rate:.6f})")
    
    # Test currency conversion
    print(f"\n💸 Testing Currency Conversion:")
    
    conversion_tests = [
        ('USD', 'EUR', 1000.0),
        ('BTC', 'USD', 0.1),
        ('EUR', 'JPY', 500.0),
        ('USDT', 'BTC', 10000.0)
    ]
    
    for from_curr, to_curr, amount in conversion_tests:
        request_id = await engine.convert_currency(from_curr, to_curr, amount)
        
        if request_id:
            print(f"  {amount} {from_curr} → {to_curr}: {request_id}")
            
            # Check conversion status
            status = await engine.get_conversion_status(request_id)
            if status:
                print(f"    Result: {status['converted_amount']:.6f} {to_curr}")
                print(f"    Rate: {status['actual_rate']:.6f}")
                print(f"    Fee: {status['actual_fee']:.6f}")
    
    # Test conversion cost calculation
    print(f"\n💰 Testing Conversion Cost Calculation:")
    
    cost_tests = [
        ('USD', 'EUR', 10000.0),
        ('BTC', 'USDT', 1.0),
        ('EUR', 'GBP', 5000.0)
    ]
    
    for from_curr, to_curr, amount in cost_tests:
        cost_analysis = await engine.calculate_conversion_cost(from_curr, to_curr, amount)
        
        if cost_analysis:
            print(f"  {amount} {from_curr} → {to_curr}:")
            recommended = cost_analysis['recommended_path']
            if recommended:
                print(f"    Best rate: {recommended['effective_rate']:.6f}")
                print(f"    Fee: {recommended['total_fee_percentage']:.2f}%")
                print(f"    Method: {recommended['method']}")
    
    # Get supported currencies
    print(f"\n🌍 Supported Currencies:")
    
    fiat_currencies = await engine.get_supported_currencies(CurrencyType.FIAT)
    crypto_currencies = await engine.get_supported_currencies(CurrencyType.CRYPTOCURRENCY)
    stablecoin_currencies = await engine.get_supported_currencies(CurrencyType.STABLECOIN)
    commodity_currencies = await engine.get_supported_currencies(CurrencyType.COMMODITY)
    
    print(f"  Fiat: {len(fiat_currencies)} currencies")
    print(f"  Cryptocurrency: {len(crypto_currencies)} currencies")
    print(f"  Stablecoin: {len(stablecoin_currencies)} currencies")
    print(f"  Commodity: {len(commodity_currencies)} currencies")
    
    # Get currency pairs
    print(f"\n📊 Available Currency Pairs:")
    
    usd_pairs = await engine.get_currency_pairs('USD')
    btc_pairs = await engine.get_currency_pairs('BTC')
    eur_pairs = await engine.get_currency_pairs('EUR')
    
    print(f"  USD pairs: {len(usd_pairs)}")
    print(f"  BTC pairs: {len(btc_pairs)}")
    print(f"  EUR pairs: {len(eur_pairs)}")
    
    # Get analytics
    print(f"\n📈 Cross-Currency Analytics:")
    
    analytics = await engine.get_analytics()
    if analytics:
        print(f"  Total currencies: {analytics['currencies']['total_currencies']}")
        print(f"  Valid exchange rates: {analytics['exchange_rates']['valid_rates']}")
        print(f"  Total conversions: {analytics['conversions']['total_conversions']}")
        print(f"  Active conversions: {analytics['conversions']['active_conversions']}")
    
    print(f"\n✅ Cross-Currency Engine testing completed!")

if __name__ == "__main__":
    asyncio.run(main())