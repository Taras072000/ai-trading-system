"""
Enterprise Institutional Services - Институциональные сервисы
Обеспечивает мульти-аккаунт управление, портфельное управление и институциональные функции
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import redis.asyncio as redis
import pandas as pd
import numpy as np
from decimal import Decimal
from prometheus_client import Counter, Histogram, Gauge
import aiohttp
from collections import defaultdict

class AccountType(Enum):
    """Типы аккаунтов"""
    RETAIL = "retail"
    PROFESSIONAL = "professional"
    INSTITUTIONAL = "institutional"
    FUND = "fund"
    HEDGE_FUND = "hedge_fund"
    FAMILY_OFFICE = "family_office"

class PortfolioType(Enum):
    """Типы портфелей"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"
    ALGORITHMIC = "algorithmic"

class OrderType(Enum):
    """Типы ордеров"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"
    BLOCK = "block"

class RiskLevel(Enum):
    """Уровни риска"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class InstitutionalAccount:
    """Институциональный аккаунт"""
    id: str
    name: str
    account_type: AccountType
    parent_account_id: Optional[str] = None
    sub_accounts: List[str] = None
    trading_limits: Dict[str, Any] = None
    risk_parameters: Dict[str, Any] = None
    compliance_rules: List[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    status: str = "active"
    
    def __post_init__(self):
        if self.sub_accounts is None:
            self.sub_accounts = []
        if self.trading_limits is None:
            self.trading_limits = {}
        if self.risk_parameters is None:
            self.risk_parameters = {}
        if self.compliance_rules is None:
            self.compliance_rules = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

@dataclass
class Portfolio:
    """Портфель"""
    id: str
    account_id: str
    name: str
    portfolio_type: PortfolioType
    base_currency: str
    total_value: Decimal
    available_balance: Decimal
    positions: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    allocation_targets: Dict[str, float]
    rebalancing_rules: Dict[str, Any]
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.positions is None:
            self.positions = {}
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.risk_metrics is None:
            self.risk_metrics = {}
        if self.allocation_targets is None:
            self.allocation_targets = {}
        if self.rebalancing_rules is None:
            self.rebalancing_rules = {}
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

@dataclass
class Position:
    """Позиция"""
    id: str
    portfolio_id: str
    symbol: str
    quantity: Decimal
    average_price: Decimal
    current_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    cost_basis: Decimal
    allocation_percentage: float
    risk_contribution: float
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()

@dataclass
class InstitutionalOrder:
    """Институциональный ордер"""
    id: str
    account_id: str
    portfolio_id: str
    symbol: str
    side: str  # buy/sell
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal]
    time_in_force: str
    execution_strategy: Dict[str, Any]
    parent_order_id: Optional[str] = None
    child_orders: List[str] = None
    status: str = "pending"
    filled_quantity: Decimal = Decimal('0')
    average_fill_price: Optional[Decimal] = None
    commission: Decimal = Decimal('0')
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.child_orders is None:
            self.child_orders = []
        if self.execution_strategy is None:
            self.execution_strategy = {}
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

# Метрики
INSTITUTIONAL_ORDERS = Counter('institutional_orders_total', 'Total institutional orders', ['account_type', 'order_type'])
PORTFOLIO_VALUE = Gauge('portfolio_value_usd', 'Portfolio value in USD', ['portfolio_id', 'account_id'])
RISK_METRICS = Gauge('portfolio_risk_metrics', 'Portfolio risk metrics', ['portfolio_id', 'metric_type'])
EXECUTION_LATENCY = Histogram('order_execution_latency_seconds', 'Order execution latency')

class EnterpriseInstitutionalServices:
    """Enterprise институциональные сервисы"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = None
        self.logger = self._setup_logging()
        
        # Состояние системы
        self.accounts: Dict[str, InstitutionalAccount] = {}
        self.portfolios: Dict[str, Portfolio] = {}
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, InstitutionalOrder] = {}
        
        # Кеши для производительности
        self.price_cache: Dict[str, Dict] = {}
        self.risk_cache: Dict[str, Dict] = {}
        
        # Подключения к биржам
        self.exchange_clients = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Настройка логирования"""
        logger = logging.getLogger('enterprise_institutional')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    async def start(self):
        """Запуск сервиса"""
        # Подключение к Redis
        self.redis_client = redis.Redis(
            host=self.config.get('redis_host', 'localhost'),
            port=self.config.get('redis_port', 6379),
            decode_responses=True
        )
        
        # Загрузка данных из Redis
        await self._load_data_from_redis()
        
        # Запуск фоновых задач
        asyncio.create_task(self._portfolio_monitor())
        asyncio.create_task(self._risk_monitor())
        asyncio.create_task(self._rebalancing_engine())
        asyncio.create_task(self._performance_calculator())
        asyncio.create_task(self._price_updater())
        
        self.logger.info("Enterprise Institutional Services started")
        
    async def stop(self):
        """Остановка сервиса"""
        if self.redis_client:
            await self.redis_client.close()
            
    # === Управление аккаунтами ===
    
    async def create_institutional_account(self, account_data: Dict[str, Any]) -> InstitutionalAccount:
        """Создание институционального аккаунта"""
        try:
            account = InstitutionalAccount(
                id=str(uuid.uuid4()),
                name=account_data['name'],
                account_type=AccountType(account_data['account_type']),
                parent_account_id=account_data.get('parent_account_id'),
                trading_limits=account_data.get('trading_limits', {}),
                risk_parameters=account_data.get('risk_parameters', {}),
                compliance_rules=account_data.get('compliance_rules', [])
            )
            
            self.accounts[account.id] = account
            await self._save_account(account)
            
            # Создание дефолтного портфеля
            await self.create_portfolio({
                'account_id': account.id,
                'name': f"{account.name} - Main Portfolio",
                'portfolio_type': 'moderate',
                'base_currency': 'USD'
            })
            
            self.logger.info(f"Institutional account created: {account.id}")
            return account
            
        except Exception as e:
            self.logger.error(f"Account creation error: {e}")
            raise
            
    async def get_account_hierarchy(self, account_id: str) -> Dict[str, Any]:
        """Получение иерархии аккаунтов"""
        account = self.accounts.get(account_id)
        if not account:
            return {}
            
        hierarchy = {
            'account': asdict(account),
            'sub_accounts': [],
            'portfolios': []
        }
        
        # Получение суб-аккаунтов
        for sub_account_id in account.sub_accounts:
            sub_account = self.accounts.get(sub_account_id)
            if sub_account:
                hierarchy['sub_accounts'].append(asdict(sub_account))
                
        # Получение портфелей
        for portfolio in self.portfolios.values():
            if portfolio.account_id == account_id:
                hierarchy['portfolios'].append(asdict(portfolio))
                
        return hierarchy
        
    async def update_trading_limits(self, account_id: str, limits: Dict[str, Any]):
        """Обновление торговых лимитов"""
        account = self.accounts.get(account_id)
        if not account:
            raise ValueError(f"Account not found: {account_id}")
            
        account.trading_limits.update(limits)
        account.updated_at = datetime.now()
        
        await self._save_account(account)
        self.logger.info(f"Trading limits updated for account: {account_id}")
        
    # === Управление портфелями ===
    
    async def create_portfolio(self, portfolio_data: Dict[str, Any]) -> Portfolio:
        """Создание портфеля"""
        try:
            portfolio = Portfolio(
                id=str(uuid.uuid4()),
                account_id=portfolio_data['account_id'],
                name=portfolio_data['name'],
                portfolio_type=PortfolioType(portfolio_data['portfolio_type']),
                base_currency=portfolio_data.get('base_currency', 'USD'),
                total_value=Decimal('0'),
                available_balance=Decimal('0'),
                positions={},
                performance_metrics={},
                risk_metrics={},
                allocation_targets=portfolio_data.get('allocation_targets', {}),
                rebalancing_rules=portfolio_data.get('rebalancing_rules', {})
            )
            
            self.portfolios[portfolio.id] = portfolio
            await self._save_portfolio(portfolio)
            
            self.logger.info(f"Portfolio created: {portfolio.id}")
            return portfolio
            
        except Exception as e:
            self.logger.error(f"Portfolio creation error: {e}")
            raise
            
    async def get_portfolio_summary(self, portfolio_id: str) -> Dict[str, Any]:
        """Получение сводки портфеля"""
        portfolio = self.portfolios.get(portfolio_id)
        if not portfolio:
            return {}
            
        # Получение позиций
        positions = []
        for position in self.positions.values():
            if position.portfolio_id == portfolio_id:
                positions.append(asdict(position))
                
        # Расчет метрик
        total_value = sum(pos['market_value'] for pos in positions)
        total_pnl = sum(pos['unrealized_pnl'] + pos['realized_pnl'] for pos in positions)
        
        return {
            'portfolio': asdict(portfolio),
            'positions': positions,
            'summary': {
                'total_value': float(total_value),
                'total_pnl': float(total_pnl),
                'position_count': len(positions),
                'currency': portfolio.base_currency
            },
            'performance': portfolio.performance_metrics,
            'risk': portfolio.risk_metrics
        }
        
    async def rebalance_portfolio(self, portfolio_id: str, target_allocations: Dict[str, float]):
        """Ребалансировка портфеля"""
        portfolio = self.portfolios.get(portfolio_id)
        if not portfolio:
            raise ValueError(f"Portfolio not found: {portfolio_id}")
            
        try:
            # Получение текущих позиций
            current_positions = {}
            for position in self.positions.values():
                if position.portfolio_id == portfolio_id:
                    current_positions[position.symbol] = position
                    
            # Расчет целевых размеров позиций
            total_value = portfolio.total_value
            target_positions = {}
            
            for symbol, target_weight in target_allocations.items():
                target_value = total_value * Decimal(str(target_weight))
                current_price = await self._get_current_price(symbol)
                target_quantity = target_value / current_price
                
                target_positions[symbol] = {
                    'target_quantity': target_quantity,
                    'target_value': target_value,
                    'current_quantity': current_positions.get(symbol, Position(
                        id='', portfolio_id='', symbol=symbol, quantity=Decimal('0'),
                        average_price=Decimal('0'), current_price=current_price,
                        market_value=Decimal('0'), unrealized_pnl=Decimal('0'),
                        realized_pnl=Decimal('0'), cost_basis=Decimal('0'),
                        allocation_percentage=0.0, risk_contribution=0.0
                    )).quantity
                }
                
            # Создание ордеров для ребалансировки
            rebalance_orders = []
            for symbol, target_data in target_positions.items():
                quantity_diff = target_data['target_quantity'] - target_data['current_quantity']
                
                if abs(quantity_diff) > Decimal('0.001'):  # Минимальный порог
                    order = await self.create_institutional_order({
                        'account_id': portfolio.account_id,
                        'portfolio_id': portfolio_id,
                        'symbol': symbol,
                        'side': 'buy' if quantity_diff > 0 else 'sell',
                        'order_type': 'market',
                        'quantity': abs(quantity_diff),
                        'execution_strategy': {
                            'type': 'rebalance',
                            'urgency': 'low'
                        }
                    })
                    rebalance_orders.append(order)
                    
            self.logger.info(f"Portfolio rebalancing initiated: {portfolio_id}, {len(rebalance_orders)} orders")
            return rebalance_orders
            
        except Exception as e:
            self.logger.error(f"Portfolio rebalancing error: {e}")
            raise
            
    # === Управление ордерами ===
    
    async def create_institutional_order(self, order_data: Dict[str, Any]) -> InstitutionalOrder:
        """Создание институционального ордера"""
        try:
            start_time = time.time()
            
            order = InstitutionalOrder(
                id=str(uuid.uuid4()),
                account_id=order_data['account_id'],
                portfolio_id=order_data['portfolio_id'],
                symbol=order_data['symbol'],
                side=order_data['side'],
                order_type=OrderType(order_data['order_type']),
                quantity=Decimal(str(order_data['quantity'])),
                price=Decimal(str(order_data['price'])) if order_data.get('price') else None,
                time_in_force=order_data.get('time_in_force', 'GTC'),
                execution_strategy=order_data.get('execution_strategy', {}),
                parent_order_id=order_data.get('parent_order_id')
            )
            
            # Проверка лимитов и рисков
            await self._validate_order(order)
            
            self.orders[order.id] = order
            await self._save_order(order)
            
            # Запуск исполнения
            asyncio.create_task(self._execute_order(order))
            
            # Метрики
            account = self.accounts.get(order.account_id)
            INSTITUTIONAL_ORDERS.labels(
                account_type=account.account_type.value if account else 'unknown',
                order_type=order.order_type.value
            ).inc()
            
            execution_time = time.time() - start_time
            EXECUTION_LATENCY.observe(execution_time)
            
            self.logger.info(f"Institutional order created: {order.id}")
            return order
            
        except Exception as e:
            self.logger.error(f"Order creation error: {e}")
            raise
            
    async def _validate_order(self, order: InstitutionalOrder):
        """Валидация ордера"""
        # Проверка аккаунта
        account = self.accounts.get(order.account_id)
        if not account:
            raise ValueError(f"Account not found: {order.account_id}")
            
        # Проверка торговых лимитов
        if account.trading_limits:
            max_order_size = account.trading_limits.get('max_order_size')
            if max_order_size and order.quantity > Decimal(str(max_order_size)):
                raise ValueError(f"Order size exceeds limit: {order.quantity} > {max_order_size}")
                
            daily_volume_limit = account.trading_limits.get('daily_volume_limit')
            if daily_volume_limit:
                daily_volume = await self._get_daily_volume(order.account_id)
                order_value = order.quantity * (order.price or await self._get_current_price(order.symbol))
                
                if daily_volume + order_value > Decimal(str(daily_volume_limit)):
                    raise ValueError(f"Daily volume limit exceeded")
                    
        # Проверка риск-параметров
        if account.risk_parameters:
            max_position_size = account.risk_parameters.get('max_position_size')
            if max_position_size:
                current_position = await self._get_position_size(order.portfolio_id, order.symbol)
                new_position_size = current_position + (order.quantity if order.side == 'buy' else -order.quantity)
                
                if abs(new_position_size) > Decimal(str(max_position_size)):
                    raise ValueError(f"Position size limit exceeded")
                    
    async def _execute_order(self, order: InstitutionalOrder):
        """Исполнение ордера"""
        try:
            if order.order_type == OrderType.TWAP:
                await self._execute_twap_order(order)
            elif order.order_type == OrderType.VWAP:
                await self._execute_vwap_order(order)
            elif order.order_type == OrderType.ICEBERG:
                await self._execute_iceberg_order(order)
            elif order.order_type == OrderType.BLOCK:
                await self._execute_block_order(order)
            else:
                await self._execute_simple_order(order)
                
        except Exception as e:
            order.status = 'failed'
            order.updated_at = datetime.now()
            await self._save_order(order)
            self.logger.error(f"Order execution error: {e}")
            
    async def _execute_twap_order(self, order: InstitutionalOrder):
        """Исполнение TWAP ордера"""
        strategy = order.execution_strategy
        duration_minutes = strategy.get('duration_minutes', 60)
        slice_count = strategy.get('slice_count', 10)
        
        slice_size = order.quantity / slice_count
        slice_interval = (duration_minutes * 60) / slice_count
        
        order.status = 'executing'
        await self._save_order(order)
        
        for i in range(slice_count):
            if order.status == 'cancelled':
                break
                
            # Создание дочернего ордера
            child_order = InstitutionalOrder(
                id=str(uuid.uuid4()),
                account_id=order.account_id,
                portfolio_id=order.portfolio_id,
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.MARKET,
                quantity=slice_size,
                parent_order_id=order.id,
                time_in_force='IOC'
            )
            
            await self._execute_simple_order(child_order)
            order.child_orders.append(child_order.id)
            
            # Обновление заполнения родительского ордера
            order.filled_quantity += child_order.filled_quantity
            
            if i < slice_count - 1:  # Не ждать после последнего слайса
                await asyncio.sleep(slice_interval)
                
        order.status = 'filled' if order.filled_quantity == order.quantity else 'partially_filled'
        order.updated_at = datetime.now()
        await self._save_order(order)
        
    async def _execute_vwap_order(self, order: InstitutionalOrder):
        """Исполнение VWAP ордера"""
        # Получение исторических данных объемов
        volume_profile = await self._get_volume_profile(order.symbol)
        
        strategy = order.execution_strategy
        duration_minutes = strategy.get('duration_minutes', 60)
        
        # Распределение ордера по времени согласно профилю объемов
        time_slices = self._calculate_vwap_slices(order.quantity, volume_profile, duration_minutes)
        
        order.status = 'executing'
        await self._save_order(order)
        
        for slice_time, slice_size in time_slices:
            if order.status == 'cancelled':
                break
                
            await asyncio.sleep(slice_time)
            
            child_order = InstitutionalOrder(
                id=str(uuid.uuid4()),
                account_id=order.account_id,
                portfolio_id=order.portfolio_id,
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.MARKET,
                quantity=slice_size,
                parent_order_id=order.id,
                time_in_force='IOC'
            )
            
            await self._execute_simple_order(child_order)
            order.child_orders.append(child_order.id)
            order.filled_quantity += child_order.filled_quantity
            
        order.status = 'filled' if order.filled_quantity == order.quantity else 'partially_filled'
        order.updated_at = datetime.now()
        await self._save_order(order)
        
    async def _execute_iceberg_order(self, order: InstitutionalOrder):
        """Исполнение Iceberg ордера"""
        strategy = order.execution_strategy
        visible_size = Decimal(str(strategy.get('visible_size', float(order.quantity) * 0.1)))
        
        order.status = 'executing'
        await self._save_order(order)
        
        remaining_quantity = order.quantity
        
        while remaining_quantity > 0 and order.status != 'cancelled':
            current_slice = min(visible_size, remaining_quantity)
            
            child_order = InstitutionalOrder(
                id=str(uuid.uuid4()),
                account_id=order.account_id,
                portfolio_id=order.portfolio_id,
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.LIMIT,
                quantity=current_slice,
                price=order.price,
                parent_order_id=order.id,
                time_in_force='GTC'
            )
            
            await self._execute_simple_order(child_order)
            order.child_orders.append(child_order.id)
            
            filled = child_order.filled_quantity
            order.filled_quantity += filled
            remaining_quantity -= filled
            
            if filled < current_slice:
                # Частичное заполнение, ждем
                await asyncio.sleep(5)
                
        order.status = 'filled' if remaining_quantity == 0 else 'partially_filled'
        order.updated_at = datetime.now()
        await self._save_order(order)
        
    async def _execute_block_order(self, order: InstitutionalOrder):
        """Исполнение блочного ордера"""
        # Поиск встречных блочных ордеров
        matching_orders = await self._find_matching_block_orders(order)
        
        if matching_orders:
            # Исполнение через блочную сеть
            await self._execute_block_crossing(order, matching_orders)
        else:
            # Размещение в блочной сети
            await self._place_in_block_network(order)
            
    async def _execute_simple_order(self, order: InstitutionalOrder):
        """Исполнение простого ордера"""
        # Симуляция исполнения (в реальной системе здесь будет интеграция с биржей)
        order.status = 'executing'
        await self._save_order(order)
        
        # Симуляция задержки исполнения
        await asyncio.sleep(0.1)
        
        # Симуляция заполнения
        current_price = await self._get_current_price(order.symbol)
        
        if order.order_type == OrderType.MARKET:
            order.filled_quantity = order.quantity
            order.average_fill_price = current_price
            order.status = 'filled'
        elif order.order_type == OrderType.LIMIT:
            if (order.side == 'buy' and current_price <= order.price) or \
               (order.side == 'sell' and current_price >= order.price):
                order.filled_quantity = order.quantity
                order.average_fill_price = order.price
                order.status = 'filled'
            else:
                order.status = 'open'
                
        order.updated_at = datetime.now()
        await self._save_order(order)
        
        # Обновление позиции
        if order.filled_quantity > 0:
            await self._update_position(order)
            
    async def _update_position(self, order: InstitutionalOrder):
        """Обновление позиции после исполнения ордера"""
        position_key = f"{order.portfolio_id}_{order.symbol}"
        position = self.positions.get(position_key)
        
        if not position:
            position = Position(
                id=position_key,
                portfolio_id=order.portfolio_id,
                symbol=order.symbol,
                quantity=Decimal('0'),
                average_price=Decimal('0'),
                current_price=order.average_fill_price,
                market_value=Decimal('0'),
                unrealized_pnl=Decimal('0'),
                realized_pnl=Decimal('0'),
                cost_basis=Decimal('0'),
                allocation_percentage=0.0,
                risk_contribution=0.0
            )
            self.positions[position_key] = position
            
        # Обновление количества и средней цены
        if order.side == 'buy':
            new_quantity = position.quantity + order.filled_quantity
            if new_quantity > 0:
                position.average_price = (
                    (position.quantity * position.average_price + 
                     order.filled_quantity * order.average_fill_price) / new_quantity
                )
            position.quantity = new_quantity
        else:  # sell
            position.quantity -= order.filled_quantity
            
        # Обновление рыночной стоимости
        position.current_price = order.average_fill_price
        position.market_value = position.quantity * position.current_price
        position.cost_basis = position.quantity * position.average_price
        position.unrealized_pnl = position.market_value - position.cost_basis
        position.last_updated = datetime.now()
        
        await self._save_position(position)
        
        # Обновление портфеля
        await self._update_portfolio_value(order.portfolio_id)
        
    # === Мониторинг и аналитика ===
    
    async def _portfolio_monitor(self):
        """Мониторинг портфелей"""
        while True:
            try:
                for portfolio in self.portfolios.values():
                    # Обновление стоимости портфеля
                    await self._update_portfolio_value(portfolio.id)
                    
                    # Обновление метрик производительности
                    await self._calculate_portfolio_performance(portfolio.id)
                    
                    # Проверка лимитов риска
                    await self._check_risk_limits(portfolio.id)
                    
                await asyncio.sleep(60)  # Обновление каждую минуту
                
            except Exception as e:
                self.logger.error(f"Portfolio monitor error: {e}")
                await asyncio.sleep(30)
                
    async def _risk_monitor(self):
        """Мониторинг рисков"""
        while True:
            try:
                for portfolio in self.portfolios.values():
                    risk_metrics = await self._calculate_risk_metrics(portfolio.id)
                    
                    # Обновление метрик в Prometheus
                    for metric_name, value in risk_metrics.items():
                        RISK_METRICS.labels(
                            portfolio_id=portfolio.id,
                            metric_type=metric_name
                        ).set(value)
                        
                    # Проверка превышения лимитов
                    await self._check_risk_breaches(portfolio.id, risk_metrics)
                    
                await asyncio.sleep(300)  # Обновление каждые 5 минут
                
            except Exception as e:
                self.logger.error(f"Risk monitor error: {e}")
                await asyncio.sleep(60)
                
    async def _rebalancing_engine(self):
        """Движок ребалансировки"""
        while True:
            try:
                for portfolio in self.portfolios.values():
                    if portfolio.rebalancing_rules.get('enabled', False):
                        await self._check_rebalancing_triggers(portfolio.id)
                        
                await asyncio.sleep(3600)  # Проверка каждый час
                
            except Exception as e:
                self.logger.error(f"Rebalancing engine error: {e}")
                await asyncio.sleep(1800)
                
    async def _performance_calculator(self):
        """Калькулятор производительности"""
        while True:
            try:
                for portfolio in self.portfolios.values():
                    performance = await self._calculate_portfolio_performance(portfolio.id)
                    
                    portfolio.performance_metrics.update(performance)
                    await self._save_portfolio(portfolio)
                    
                await asyncio.sleep(1800)  # Обновление каждые 30 минут
                
            except Exception as e:
                self.logger.error(f"Performance calculator error: {e}")
                await asyncio.sleep(900)
                
    async def _price_updater(self):
        """Обновление цен"""
        while True:
            try:
                # Получение списка всех символов
                symbols = set()
                for position in self.positions.values():
                    symbols.add(position.symbol)
                    
                # Обновление цен
                for symbol in symbols:
                    price = await self._fetch_current_price(symbol)
                    self.price_cache[symbol] = {
                        'price': price,
                        'timestamp': time.time()
                    }
                    
                await asyncio.sleep(5)  # Обновление каждые 5 секунд
                
            except Exception as e:
                self.logger.error(f"Price updater error: {e}")
                await asyncio.sleep(30)
                
    # === Вспомогательные методы ===
    
    async def _get_current_price(self, symbol: str) -> Decimal:
        """Получение текущей цены"""
        cached_price = self.price_cache.get(symbol)
        if cached_price and time.time() - cached_price['timestamp'] < 60:
            return cached_price['price']
            
        price = await self._fetch_current_price(symbol)
        self.price_cache[symbol] = {
            'price': price,
            'timestamp': time.time()
        }
        return price
        
    async def _fetch_current_price(self, symbol: str) -> Decimal:
        """Получение цены с биржи"""
        # Симуляция получения цены
        # В реальной системе здесь будет API биржи
        base_price = 50000 if symbol == 'BTCUSDT' else 3000
        variation = np.random.normal(0, 0.01)
        price = base_price * (1 + variation)
        return Decimal(str(round(price, 2)))
        
    async def _update_portfolio_value(self, portfolio_id: str):
        """Обновление стоимости портфеля"""
        portfolio = self.portfolios.get(portfolio_id)
        if not portfolio:
            return
            
        total_value = Decimal('0')
        
        for position in self.positions.values():
            if position.portfolio_id == portfolio_id:
                current_price = await self._get_current_price(position.symbol)
                position.current_price = current_price
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = position.market_value - position.cost_basis
                
                total_value += position.market_value
                await self._save_position(position)
                
        portfolio.total_value = total_value
        portfolio.updated_at = datetime.now()
        await self._save_portfolio(portfolio)
        
        # Обновление метрики
        PORTFOLIO_VALUE.labels(
            portfolio_id=portfolio_id,
            account_id=portfolio.account_id
        ).set(float(total_value))
        
    async def _calculate_portfolio_performance(self, portfolio_id: str) -> Dict[str, float]:
        """Расчет производительности портфеля"""
        # Здесь должна быть сложная логика расчета производительности
        # Возвращаем упрощенные метрики
        return {
            'daily_return': np.random.normal(0.001, 0.02),
            'weekly_return': np.random.normal(0.005, 0.05),
            'monthly_return': np.random.normal(0.02, 0.1),
            'sharpe_ratio': np.random.normal(1.5, 0.5),
            'max_drawdown': np.random.uniform(0.01, 0.1),
            'volatility': np.random.uniform(0.1, 0.3)
        }
        
    async def _calculate_risk_metrics(self, portfolio_id: str) -> Dict[str, float]:
        """Расчет метрик риска"""
        return {
            'var_95': np.random.uniform(0.01, 0.05),
            'var_99': np.random.uniform(0.02, 0.08),
            'expected_shortfall': np.random.uniform(0.03, 0.1),
            'beta': np.random.normal(1.0, 0.3),
            'correlation_to_market': np.random.uniform(0.5, 0.9)
        }
        
    # === Сохранение данных ===
    
    async def _save_account(self, account: InstitutionalAccount):
        """Сохранение аккаунта"""
        await self.redis_client.setex(
            f"account:{account.id}",
            86400 * 30,
            json.dumps(asdict(account), default=str)
        )
        
    async def _save_portfolio(self, portfolio: Portfolio):
        """Сохранение портфеля"""
        await self.redis_client.setex(
            f"portfolio:{portfolio.id}",
            86400 * 30,
            json.dumps(asdict(portfolio), default=str)
        )
        
    async def _save_position(self, position: Position):
        """Сохранение позиции"""
        await self.redis_client.setex(
            f"position:{position.id}",
            86400 * 7,
            json.dumps(asdict(position), default=str)
        )
        
    async def _save_order(self, order: InstitutionalOrder):
        """Сохранение ордера"""
        await self.redis_client.setex(
            f"order:{order.id}",
            86400 * 7,
            json.dumps(asdict(order), default=str)
        )
        
    async def _load_data_from_redis(self):
        """Загрузка данных из Redis"""
        # Загрузка аккаунтов
        account_keys = await self.redis_client.keys("account:*")
        for key in account_keys:
            data = await self.redis_client.get(key)
            if data:
                account_data = json.loads(data)
                account = InstitutionalAccount(**account_data)
                self.accounts[account.id] = account
                
        # Загрузка портфелей
        portfolio_keys = await self.redis_client.keys("portfolio:*")
        for key in portfolio_keys:
            data = await self.redis_client.get(key)
            if data:
                portfolio_data = json.loads(data)
                portfolio = Portfolio(**portfolio_data)
                self.portfolios[portfolio.id] = portfolio

async def main():
    """Основная функция запуска"""
    config = {
        'redis_host': 'localhost',
        'redis_port': 6379
    }
    
    institutional_services = EnterpriseInstitutionalServices(config)
    await institutional_services.start()
    
    print("Enterprise Institutional Services started")
    
    try:
        await asyncio.Future()  # Бесконечное ожидание
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await institutional_services.stop()

if __name__ == '__main__':
    asyncio.run(main())