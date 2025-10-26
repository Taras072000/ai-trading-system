"""
Многоуровневая AI-система для торговли
Архитектура с разделением ответственности:
1. Lava AI - технический анализ индикаторов и паттернов
2. Trading AI - анализ рыночных условий и торговых сигналов
3. LGBM AI - управление рисками и машинное обучение
4. Mistral AI - финальные торговые решения
5. AI Manager - координация всех модулей
"""

import asyncio
import logging
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, time
import json
from dataclasses import dataclass, asdict
from utils.timezone_utils import get_utc_now

# Импорты реальных AI модулей
from .lava_ai import LavaAI
from .trading_ai import TradingAI
from .lgbm_ai import LGBMAI
from .mistral_ai import MistralAI
from .ai_manager import AIManager, AIModuleType

# Импорты для обучения с подкреплением
from .reinforcement_learning_engine import ReinforcementLearningEngine, ReinforcementConfig
from .mistral_server_manager import MistralServerManager

logger = logging.getLogger(__name__)

@dataclass
class AISignal:
    """Структура сигнала от AI модуля"""
    module_name: str
    signal_type: str
    confidence: float
    data: Dict[str, Any]
    timestamp: datetime
    reasoning: str

@dataclass
class TradingDecision:
    """Финальное торговое решение"""
    action: str  # 'LONG', 'SHORT', 'HOLD', 'CLOSE'
    confidence: float
    entry_price: Optional[float]
    position_size: float
    stop_loss: float
    take_profits: List[Dict[str, float]]  # Сетка из 5 тейк-профитов
    dynamic_stop: Dict[str, Any]
    reasoning: str
    risk_score: float
    timestamp: datetime

class MultiAIOrchestrator:
    """
    Главный оркестратор многоуровневой AI-системы
    Координирует работу всех AI-модулей и принимает финальные решения
    """
    
    def __init__(self, backtest_mode: bool = False, reinforcement_learning: bool = False):
        # Инициализируем AI Manager для координации модулей
        self.ai_manager = AIManager()
        
        # Инициализируем реальные AI модули
        self.lava_ai = LavaAI()          # Технический анализ
        self.trading_ai = TradingAI()    # Рыночные условия и торговые сигналы
        self.lgbm_ai = LGBMAI()          # Риск-менеджмент и ML
        self.mistral_ai = MistralAI()    # Финальные решения
        
        self.is_initialized = False
        self.signal_history = []
        self.decision_history = []
        self.backtest_mode = backtest_mode  # Режим бэктестинга
        
        # Система обучения с подкреплением
        self.reinforcement_learning = reinforcement_learning
        self.rl_engine: Optional[ReinforcementLearningEngine] = None
        self.mistral_server_manager: Optional[MistralServerManager] = None
        
        # Веса для разных AI модулей (будут управляться RL если включено)
        self.module_weights = {
            'lava_ai': 0.35,      # Технический анализ
            'trading_ai': 0.25,   # Рыночные условия
            'lgbm_ai': 0.40,      # Риск-менеджмент
            'mistral_ai': 0.0     # Начинаем с 0, будет увеличиваться при хороших результатах
        }
        
        # История торговых результатов для обучения с подкреплением
        self.trade_results_history = []
        
        # Адаптивные параметры для разных типов активов с улучшенным риск-менеджментом
        self.asset_configs = {
            'BTCUSDT': {
                'volatility_threshold': 0.2,  # Экстремально снижен
                'directional_threshold': 10,  # Экстремально снижен
                'movement_24h_threshold': 0.3,  # Экстремально снижен
                'base_position_size': 0.06,
                'winrate_multiplier': 1.0,
                'stop_loss_percent': 1.8,
                'take_profit_levels': [3.6, 5.4, 7.2, 9.6, 12.0],
                'risk_reward_ratio': 2.5,
                'market_commission': 0.001,
                'limit_commission': 0.001,
                'dynamic_risk_enabled': True,
                'volatility_multiplier': 0.3,  # Экстремально снижен
                'min_confidence_threshold': 0.15  # Очень низкий порог для активной торговли
            },
            'ETHUSDT': {
                'enabled': True,
                'volatility_threshold': 0.2,  # Экстремально снижен
                'directional_threshold': 10,  # Экстремально снижен
                'movement_24h_threshold': 0.3,  # Экстремально снижен
                'base_position_size': 0.045,
                'winrate_multiplier': 1.0,
                'stop_loss_percent': 1.8,
                'take_profit_levels': [3.0, 4.5, 6.0, 8.0, 10.0],
                'risk_reward_ratio': 2.5,
                'market_commission': 0.001,
                'limit_commission': 0.001,
                'dynamic_risk_enabled': True,
                'volatility_multiplier': 0.3,  # Экстремально снижен
                'min_confidence_threshold': 0.15,  # Очень низкий порог для активной торговли
                'max_daily_trades': 5
            },
            'BNBUSDT': {
                'volatility_threshold': 0.2,  # Экстремально снижен
                'directional_threshold': 10,  # Экстремально снижен
                'movement_24h_threshold': 0.3,  # Экстремально снижен
                'base_position_size': 0.08,
                'winrate_multiplier': 1.0,
                'stop_loss_percent': 2.2,
                'take_profit_levels': [4.4, 6.6, 8.8, 11.0, 13.2],
                'risk_reward_ratio': 2.2,
                'market_commission': 0.001,
                'limit_commission': 0.001,
                'dynamic_risk_enabled': True,
                'volatility_multiplier': 0.3,  # Экстремально снижен
                'min_confidence_threshold': 0.15  # Очень низкий порог для активной торговли
            },
            'ADAUSDT': {
                'volatility_threshold': 0.2,  # Экстремально снижен
                'directional_threshold': 10,  # Экстремально снижен
                'movement_24h_threshold': 0.3,  # Экстремально снижен
                'base_position_size': 0.07,
                'winrate_multiplier': 1.0,
                'stop_loss_percent': 2.0,
                'take_profit_levels': [4.0, 6.0, 8.0, 10.5, 13.0],
                'risk_reward_ratio': 2.3,
                'market_commission': 0.001,
                'limit_commission': 0.001,
                'dynamic_risk_enabled': True,
                'volatility_multiplier': 0.3,  # Экстремально снижен
                'min_confidence_threshold': 0.15  # Очень низкий порог для активной торговли
            },
            'SOLUSDT': {
                'volatility_threshold': 0.2,  # Экстремально низкий для генерации сделок
                'directional_threshold': 10,  # Экстремально низкий
                'movement_24h_threshold': 0.3,  # Экстремально низкий
                'base_position_size': 0.05,
                'winrate_multiplier': 1.0,
                'stop_loss_percent': 1.5,
                'take_profit_levels': [3.0, 4.5, 6.0, 7.5, 9.0],
                'risk_reward_ratio': 2.0,
                'market_commission': 0.001,
                'limit_commission': 0.001,
                'dynamic_risk_enabled': True,
                'volatility_multiplier': 0.2,  # Экстремально низкий
                'min_confidence_threshold': 0.15  # Очень низкий порог для активной торговли
            },
            'default': {
                'volatility_threshold': 0.5,
                'directional_threshold': 25,
                'movement_24h_threshold': 0.6,
                'base_position_size': 0.10,
                'winrate_multiplier': 1.0,
                'stop_loss_percent': 2.0,  # Уменьшен для соотношения 1:2
                'take_profit_levels': [4.0, 6.0, 8.0, 10.0, 12.0],  # Минимум 1:2
                'risk_reward_ratio': 2.0,
                'market_commission': 0.001,
                'limit_commission': 0.001,
                'dynamic_risk_enabled': True,
                'volatility_multiplier': 1.0
            }
        }
        
        # Временные фильтры (UTC время)
        self.optimal_trading_hours = {
            'default': {
                'start': time(8, 0),   # 08:00 UTC
                'end': time(20, 0)     # 20:00 UTC
            },
            # Специальные временные окна для проблемных активов
            'ETHUSDT': {
                'start': time(10, 0),  # 10:00 UTC - более узкое окно
                'end': time(18, 0)     # 18:00 UTC - избегаем волатильных часов
            },
            'SOLUSDT': {
                'start': time(8, 0),   # 08:00 UTC - расширенное окно для тестирования
                'end': time(20, 0)     # 20:00 UTC - расширенное окно для тестирования
            }
        }
        
        logger.info("MultiAI Orchestrator инициализирован с адаптивными фильтрами")
    
    async def initialize(self):
        """Инициализация всех AI модулей"""
        try:
            logger.info("Инициализация AI модулей...")
            
            # Инициализируем AI Manager
            await self.ai_manager.initialize()
            
            # Инициализируем систему обучения с подкреплением если включена
            if self.reinforcement_learning:
                await self._initialize_reinforcement_learning()
            
            # Инициализируем все модули параллельно
            await asyncio.gather(
                self.lava_ai.initialize(),
                self.trading_ai.initialize(),
                self.lgbm_ai.initialize(),
                self.mistral_ai.initialize()
            )
            
            self.is_initialized = True
            logger.info("✅ Все AI модули успешно инициализированы")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации AI модулей: {e}")
            raise
    
    async def _initialize_reinforcement_learning(self):
        """Инициализация системы обучения с подкреплением"""
        try:
            logger.info("🧠 Инициализация системы обучения с подкреплением...")
            
            # Создаем движок обучения с подкреплением
            self.rl_engine = ReinforcementLearningEngine()
            
            # Создаем менеджер Mistral сервера
            self.mistral_server_manager = MistralServerManager()
            
            # Синхронизируем веса с движком обучения
            if self.rl_engine.get_model_weights():
                self.module_weights = self.rl_engine.get_model_weights()
                logger.info(f"📊 Загружены веса из RL движка: {self.module_weights}")
            else:
                # Устанавливаем начальные веса в движок
                self.rl_engine.set_model_weights(self.module_weights)
                logger.info(f"📊 Установлены начальные веса в RL движок: {self.module_weights}")
            
            logger.info("✅ Система обучения с подкреплением инициализирована")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации системы обучения с подкреплением: {e}")
            raise
    
    async def analyze_and_decide(self, 
                               symbol: str, 
                               data: pd.DataFrame,
                               current_position: Optional[Dict] = None) -> TradingDecision:
        """
        Главный метод анализа и принятия торгового решения
        Координирует работу всех AI модулей
        """
        try:
            if not self.is_initialized:
                await self.initialize()
            
            logger.info(f"🔍 Начинаем анализ для {symbol}")
            
            # Проверяем, включен ли актив для торговли
            asset_config = self.asset_configs.get(symbol, {})
            if not asset_config.get('enabled', True):
                logger.info(f"🚫 Актив {symbol} временно отключен от торговли")
                return TradingDecision(
                    action='HOLD',
                    confidence=0.0,
                    entry_price=None,
                    position_size=0.0,
                    stop_loss=0.0,
                    take_profits=[],
                    dynamic_stop={'enabled': False},
                    reasoning=f"Актив {symbol} временно отключен от торговли",
                    risk_score=0.0,
                    timestamp=get_utc_now()
                )
            
            # Проверяем временной фильтр
            time_filter = self._check_time_filter(symbol)
            if not time_filter['passed']:
                logger.info(f"⏰ Торговля заблокирована временным фильтром: {time_filter['reason']}")
                return TradingDecision(
                    action='HOLD',
                    confidence=0.0,
                    entry_price=None,
                    position_size=0.0,
                    stop_loss=0.0,
                    take_profits=[],
                    dynamic_stop={'enabled': False},
                    reasoning=f"Временной фильтр: {time_filter['reason']}",
                    risk_score=0.0,
                    timestamp=datetime.now()
                )
            
            # Проверяем адаптивный фильтр волатильности
            volatility_check = self._check_adaptive_volatility_filter(symbol, data)
            if not volatility_check['passed']:
                logger.info(f"📊 Торговля заблокирована фильтром волатильности: {volatility_check['reason']}")
                return TradingDecision(
                    action='HOLD',
                    confidence=0.0,
                    entry_price=None,
                    position_size=0.0,
                    stop_loss=0.0,
                    take_profits=[],
                    dynamic_stop={'enabled': False},
                    reasoning=f"Фильтр волатильности: {volatility_check['reason']}",
                    risk_score=0.0,
                    timestamp=datetime.now()
                )
            
            # Этап 1: Параллельный анализ всех AI модулей
            signals = await self._collect_ai_signals(symbol, data, current_position)
            
            # Этап 2: Агрегация и валидация сигналов
            aggregated_signals = self._aggregate_signals(signals)
            
            # Этап 3: Mistral AI принимает финальное решение
            final_decision = await self._make_final_decision(aggregated_signals)
            
            # Создаем торговое решение с улучшенными параметрами
            signals_data = {
                'aggregated_signals': aggregated_signals,
                'market_summary': self._summarize_market_data(data),
                'volatility_check': volatility_check
            }
            
            decision = self._create_trading_decision(
                symbol=symbol,
                data=data,
                final_signal=final_decision['action'],
                confidence=final_decision['confidence'],
                reasoning=final_decision['reasoning'],
                signals_data=signals_data
            )
            
            # Сохраняем историю
            self.signal_history.append(signals)
            self.decision_history.append(decision)
            
            logger.info(f"✅ Решение принято: {decision.action} (уверенность: {decision.confidence*100:.1f}%, размер: {decision.position_size:.3f})")
            return decision
            
        except Exception as e:
            logger.error(f"Ошибка в процессе анализа и принятия решения: {e}")
            # Возвращаем безопасное решение
            return self._create_safe_decision(str(e))
    
    async def _apply_solusdt_improvements(self, symbol: str, data: pd.DataFrame, signals: List[AISignal]) -> List[AISignal]:
        """Специальные улучшения для торговли SOLUSDT на основе анализа неудачных сделок"""
        if symbol != 'SOLUSDT':
            return signals
        
        try:
            # Анализ причин неудач SOLUSDT:
            # 1. Много ложных пробоев (7 из 12 сделок убыточные)
            # 2. Слишком быстрые стоп-лоссы (большинство закрылись за 1-6 часов)
            # 3. Неудачные входы на локальных максимумах
            
            current_price = data['close'].iloc[-1]
            
            # Дополнительная проверка на ложные пробои
            price_volatility_24h = abs(data['close'].iloc[-1] - data['close'].iloc[-25]) / data['close'].iloc[-25] * 100 if len(data) >= 25 else 0
            
            # Анализ RSI для избежания входов на экстремумах
            rsi_period = min(14, len(data) - 1)
            if rsi_period > 0:
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            else:
                current_rsi = 50
            
            # Анализ объема для подтверждения движения
            volume_sma = data['volume'].rolling(window=10).mean()
            current_volume_ratio = data['volume'].iloc[-1] / volume_sma.iloc[-1] if volume_sma.iloc[-1] > 0 else 1
            
            # Модификация сигналов для SOLUSDT
            improved_signals = []
            for signal in signals:
                modified_signal = signal
                
                # Снижаем уверенность при неблагоприятных условиях
                confidence_penalty = 0
                reasoning_additions = []
                
                # Штраф за экстремальные значения RSI
                if current_rsi > 70:
                    confidence_penalty += 0.2
                    reasoning_additions.append(f"RSI перекуплен ({current_rsi:.1f})")
                elif current_rsi < 30:
                    confidence_penalty += 0.2
                    reasoning_additions.append(f"RSI перепродан ({current_rsi:.1f})")
                
                # Штраф за низкий объем
                if current_volume_ratio < 0.8:
                    confidence_penalty += 0.15
                    reasoning_additions.append(f"Низкий объем ({current_volume_ratio:.2f}x)")
                
                # Бонус за высокий объем
                if current_volume_ratio > 1.5:
                    confidence_penalty -= 0.1
                    reasoning_additions.append(f"Высокий объем ({current_volume_ratio:.2f}x)")
                
                # Штраф за высокую волатильность (может быть ложный пробой)
                if price_volatility_24h > 8:
                    confidence_penalty += 0.25
                    reasoning_additions.append(f"Высокая волатильность ({price_volatility_24h:.1f}%)")
                
                # Применяем модификации
                new_confidence = max(0.1, min(0.9, signal.confidence - confidence_penalty))
                new_reasoning = signal.reasoning
                if reasoning_additions:
                    new_reasoning += f" | SOLUSDT корректировки: {'; '.join(reasoning_additions)}"
                
                # Создаем модифицированный сигнал
                modified_signal = AISignal(
                    module_name=signal.module_name,
                    signal_type=signal.signal_type,
                    confidence=new_confidence,
                    data=signal.data.copy(),
                    timestamp=signal.timestamp,
                    reasoning=new_reasoning
                )
                
                # Добавляем дополнительные данные для SOLUSDT
                modified_signal.data.update({
                    'solusdt_rsi': current_rsi,
                    'solusdt_volume_ratio': current_volume_ratio,
                    'solusdt_volatility_24h': price_volatility_24h,
                    'solusdt_confidence_penalty': confidence_penalty
                })
                
                improved_signals.append(modified_signal)
            
            logger.info(f"SOLUSDT улучшения применены: RSI={current_rsi:.1f}, Объем={current_volume_ratio:.2f}x, Волатильность={price_volatility_24h:.1f}%")
            return improved_signals
            
        except Exception as e:
            logger.error(f"Ошибка в улучшениях SOLUSDT: {e}")
            return signals

    async def _collect_ai_signals(self, 
                                symbol: str, 
                                data: pd.DataFrame,
                                current_position: Optional[Dict]) -> List[AISignal]:
        """Сбор сигналов от всех AI модулей с параллельным выполнением"""
        try:
            # Параллельный запуск всех AI модулей
            tasks = [
                self._get_lava_signal(symbol, data),
                self._get_trading_signal(symbol, data),
                self._get_lgbm_signal(symbol, data, current_position)
            ]
            
            signals = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Фильтруем успешные сигналы
            valid_signals = []
            for signal in signals:
                if isinstance(signal, AISignal):
                    valid_signals.append(signal)
                elif isinstance(signal, Exception):
                    logger.error(f"Ошибка получения сигнала: {signal}")
            
            # Применяем специальные улучшения для SOLUSDT
            improved_signals = await self._apply_solusdt_improvements(symbol, data, valid_signals)
            
            logger.info(f"Собрано {len(improved_signals)} сигналов от AI модулей для {symbol}")
            return improved_signals
            
        except Exception as e:
            logger.error(f"Ошибка сбора AI сигналов: {e}")
            return []

    async def _get_lava_signal(self, symbol: str, data: pd.DataFrame) -> AISignal:
        """Получение сигнала от Lava AI (технический анализ)"""
        try:
            # Используем Lava AI для технического анализа
            analysis = await self.lava_ai.analyze_market_data(symbol, data, 'comprehensive')
            
            # Получаем торговые сигналы
            signals = await self.lava_ai.generate_trading_signals(data)
            
            return AISignal(
                module_name="lava_ai",
                signal_type=signals.get('signal', 'HOLD'),
                confidence=signals.get('confidence', 0.5),
                data={
                    'technical_analysis': analysis,  # analysis уже является словарем
                    'trading_signals': signals,
                    'patterns': await self.lava_ai.analyze_patterns(data),
                    'support_resistance': await self.lava_ai.identify_support_resistance(data)
                },
                timestamp=datetime.now(),
                reasoning=f"Технический анализ: {signals.get('reasoning', 'Анализ индикаторов и паттернов')}"
            )
        except Exception as e:
            logger.error(f"Ошибка в Lava AI: {e}")
            return self._create_fallback_signal("lava_ai", str(e))
    
    async def _get_trading_signal(self, symbol: str, data: pd.DataFrame) -> AISignal:
        """Получение сигнала от Trading AI (рыночные условия)"""
        try:
            # Используем Trading AI для анализа рыночных условий
            market_signal = await self.trading_ai.analyze_market(symbol, data)
            
            # Получаем анализ рисков
            risk_analysis = await self.trading_ai.analyze_risk_management(symbol, data)
            
            return AISignal(
                module_name="trading_ai",
                signal_type=market_signal.action,
                confidence=market_signal.confidence,
                data={
                    'market_signal': {
                        'action': market_signal.action,
                        'price': market_signal.price,
                        'reason': market_signal.reason
                    },
                    'risk_analysis': risk_analysis
                },
                timestamp=datetime.now(),
                reasoning=f"Рыночные условия: {market_signal.reason}"
            )
        except Exception as e:
            logger.error(f"Ошибка в Trading AI: {e}")
            return self._create_fallback_signal("trading_ai", str(e))
    
    async def _get_lgbm_signal(self, 
                             symbol: str, 
                             data: pd.DataFrame,
                             current_position: Optional[Dict]) -> AISignal:
        """Получение сигнала от LGBM AI (риск-менеджмент и ML)"""
        try:
            # Создаем фичи для ML модели
            features = await self.lgbm_ai.create_trading_features(data)
            
            # Получаем предсказание движения цены
            price_prediction = await self.lgbm_ai.predict_price_movement(data)
            
            # Анализируем риски позиции
            position_analysis = {}
            if current_position:
                position_analysis = await self.trading_ai.optimize_position_sizing(
                    price_data=data,
                    volatility=data['close'].pct_change().std()
                )
            
            # Определяем сигнал на основе ML предсказания
            signal_type = 'HOLD'
            confidence = price_prediction.confidence
            
            if price_prediction.prediction > 0.6:
                signal_type = 'LONG'
            elif price_prediction.prediction < 0.4:
                signal_type = 'SHORT'
            
            return AISignal(
                module_name="lgbm_ai",
                signal_type=signal_type,
                confidence=confidence,
                data={
                    'ml_prediction': {
                        'prediction': price_prediction.prediction,
                        'confidence': price_prediction.confidence,
                        'feature_importance': price_prediction.feature_importance
                    },
                    'position_analysis': position_analysis,
                    'features': features.to_dict() if hasattr(features, 'to_dict') else {}
                },
                timestamp=datetime.now(),
                reasoning=f"ML анализ: предсказание {price_prediction.prediction:.3f} с уверенностью {confidence*100:.1f}%"
            )
        except Exception as e:
            logger.error(f"Ошибка в LGBM AI: {e}")
            return self._create_fallback_signal("lgbm_ai", str(e))
    
    def _aggregate_signals(self, signals: List[AISignal]) -> Dict[str, Any]:
        """Агрегация сигналов от всех AI модулей"""
        if not signals:
            return {
                'final_signal': 'HOLD',
                'confidence': 0.0,
                'reasoning': 'Нет доступных сигналов'
            }
        
        # Подсчет голосов по типам сигналов
        signal_votes = {'LONG': 0, 'SHORT': 0, 'HOLD': 0}
        weighted_confidence = 0
        total_weight = 0
        
        for signal in signals:
            weight = self.module_weights.get(signal.module_name, 1.0)
            
            # Нормализируем типы сигналов
            normalized_signal = signal.signal_type.upper()
            if normalized_signal in ['BUY', 'LONG']:
                normalized_signal = 'LONG'
            elif normalized_signal in ['SELL', 'SHORT']:
                normalized_signal = 'SHORT'
            elif normalized_signal in ['HOLD', 'WAIT']:
                normalized_signal = 'HOLD'
            else:
                normalized_signal = 'HOLD'  # По умолчанию
            
            signal_votes[normalized_signal] += weight * signal.confidence
            weighted_confidence += weight * signal.confidence
            total_weight += weight
        
        # Определение финального сигнала
        final_signal = max(signal_votes, key=signal_votes.get)
        final_confidence = weighted_confidence / total_weight if total_weight > 0 else 0
        
        # Создание детального анализа
        reasoning_parts = []
        for signal in signals:
            weight = self.module_weights.get(signal.module_name, 1.0)
            reasoning_parts.append(
                f"{signal.module_name}: {signal.signal_type} "
                f"(уверенность: {signal.confidence*100:.1f}%, вес: {weight})"
            )
        
        return {
            'final_signal': final_signal,
            'confidence': final_confidence,
            'signal_votes': signal_votes,
            'individual_signals': [
                {
                    'module': signal.module_name,
                    'signal': signal.signal_type,
                    'confidence': signal.confidence,
                    'reasoning': signal.reasoning,
                    'data': signal.data
                }
                for signal in signals
            ],
            'reasoning': f"Агрегированный анализ: {'; '.join(reasoning_parts)}"
        }
    
    async def _make_final_decision(self, aggregated_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Принятие финального решения с использованием Mistral AI"""
        try:
            # Подготавливаем данные для анализа Mistral AI
            analysis_data = {
                'aggregated_signals': aggregated_signals,
                'market_context': {
                    'timestamp': datetime.now().isoformat(),
                    'signal_strength': aggregated_signals['confidence'],
                    'consensus': aggregated_signals['final_signal']
                }
            }
            
            # Используем Mistral AI для финального анализа
            mistral_analysis = await self.mistral_ai.analyze_trading_data(analysis_data)
            
            # Получаем рекомендации по торговле
            trading_recommendation = await self.mistral_ai.generate_trading_recommendation(
                aggregated_signals=aggregated_signals,
                market_summary=analysis_data
            )
            
            return {
                'action': aggregated_signals['final_signal'],
                'confidence': aggregated_signals['confidence'],
                'mistral_analysis': mistral_analysis.text if hasattr(mistral_analysis, 'text') else str(mistral_analysis),
                'trading_recommendation': trading_recommendation,
                'reasoning': f"Финальное решение на основе консенсуса AI модулей и анализа Mistral: {mistral_analysis.text if hasattr(mistral_analysis, 'text') else str(mistral_analysis)}",
                'individual_signals': aggregated_signals['individual_signals'],
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Ошибка в Mistral AI при принятии финального решения: {e}")
            # Возвращаем решение на основе агрегации без Mistral
            return {
                'action': aggregated_signals['final_signal'],
                'confidence': aggregated_signals['confidence'],
                'mistral_analysis': f"Ошибка анализа: {str(e)}",
                'trading_recommendation': None,
                'reasoning': aggregated_signals['reasoning'],
                'individual_signals': aggregated_signals['individual_signals'],
                'timestamp': datetime.now()
            }
    
    def _check_adaptive_volatility_filter(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Адаптивный фильтр волатильности с индивидуальными параметрами для каждого актива"""
        try:
            if len(data) < 24:
                return {
                    'passed': False,
                    'reason': 'Недостаточно данных для анализа волатильности',
                    'metrics': {}
                }
            
            # Получаем конфигурацию для конкретного актива
            config = self.asset_configs.get(symbol, {
                'volatility_threshold': 0.5,
                'directional_threshold': 20,
                'movement_24h_threshold': 0.5
            })
            
            # В режиме бэктестинга используем более мягкие критерии
            if self.backtest_mode:
                config = {
                    'volatility_threshold': 0.1,  # Снижено с 0.5 до 0.1
                    'directional_threshold': 5,   # Снижено с 20 до 5
                    'movement_24h_threshold': 0.1 # Снижено с 0.5 до 0.1
                }
            
            # Расчет ATR (Average True Range) за последние 14 периодов
            high = data['high'].tail(14)
            low = data['low'].tail(14)
            close = data['close'].tail(15)
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1).tail(14))
            tr3 = abs(low - close.shift(1).tail(14))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.mean()
            
            # Текущая цена
            current_price = data['close'].iloc[-1]
            
            # Процентная волатильность (ATR / цена)
            volatility_percent = (atr / current_price) * 100
            
            # Анализ направленности движения (последние 24 часа)
            price_24h_ago = data['close'].iloc[-25] if len(data) >= 25 else data['close'].iloc[0]
            price_change_24h = abs(current_price - price_24h_ago) / price_24h_ago * 100
            
            # Анализ трендовости (соотношение направленного движения к общему)
            price_changes = data['close'].tail(12).diff().dropna()
            directional_movement = abs(price_changes.sum()) / price_changes.abs().sum() * 100 if len(price_changes) > 0 else 0
            
            # Адаптивные критерии фильтра
            min_volatility = config['volatility_threshold']
            min_directional = config['directional_threshold']
            min_24h_movement = config['movement_24h_threshold']
            
            # Проверка условий
            volatility_ok = volatility_percent >= min_volatility
            directional_ok = directional_movement >= min_directional
            movement_24h_ok = price_change_24h >= min_24h_movement
            
            passed = volatility_ok and directional_ok and movement_24h_ok
            
            metrics = {
                'volatility_percent': round(volatility_percent, 3),
                'directional_movement': round(directional_movement, 2),
                'price_change_24h': round(price_change_24h, 3),
                'atr': round(atr, 6),
                'current_price': round(current_price, 6),
                'backtest_mode': self.backtest_mode,
                'thresholds_used': {
                    'volatility': min_volatility,
                    'directional': min_directional,
                    'movement_24h': min_24h_movement
                }
            }
            
            if not passed:
                reasons = []
                if not volatility_ok:
                    reasons.append(f"Низкая волатильность: {volatility_percent:.2f}% < {min_volatility}%")
                if not directional_ok:
                    reasons.append(f"Слабая направленность: {directional_movement:.1f}% < {min_directional}%")
                if not movement_24h_ok:
                    reasons.append(f"Малое движение за 24ч: {price_change_24h:.2f}% < {min_24h_movement}%")
                
                return {
                    'passed': False,
                    'reason': '; '.join(reasons),
                    'metrics': metrics
                }
            
            return {
                'passed': True,
                'reason': f'Волатильность: {volatility_percent:.2f}%, Направленность: {directional_movement:.1f}%, Движение 24ч: {price_change_24h:.2f}%',
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Ошибка в адаптивном фильтре волатильности: {e}")
            return {
                'passed': False,
                'reason': f'Ошибка расчета: {str(e)}',
                'metrics': {}
            }

    def _check_time_filter(self, symbol: str = None) -> Dict[str, Any]:
        """Проверка временного фильтра для оптимального времени торговли"""
        try:
            # В режиме бэктестинга пропускаем временной фильтр
            if self.backtest_mode:
                return {
                    'passed': True,
                    'reason': 'Режим бэктестинга: временной фильтр отключен'
                }
            
            current_time = datetime.utcnow().time()
            
            # Получаем временные настройки для конкретного актива или дефолтные
            time_config = self.optimal_trading_hours.get(symbol, self.optimal_trading_hours['default'])
            start_time = time_config['start']
            end_time = time_config['end']
            
            # Проверяем, находится ли текущее время в оптимальном диапазоне
            if start_time <= current_time <= end_time:
                return {
                    'passed': True,
                    'reason': f'Оптимальное время торговли для {symbol or "default"}: {current_time.strftime("%H:%M")} UTC'
                }
            else:
                return {
                    'passed': False,
                    'reason': f'Неоптимальное время для {symbol or "default"}: {current_time.strftime("%H:%M")} UTC (оптимально: {start_time.strftime("%H:%M")}-{end_time.strftime("%H:%M")})'
                }
                
        except Exception as e:
            logger.error(f"Ошибка в временном фильтре: {e}")
            return {
                'passed': True,  # В случае ошибки разрешаем торговлю
                'reason': f'Ошибка временного фильтра: {str(e)}'
            }

    def _calculate_dynamic_position_size(self, symbol: str, confidence: float) -> float:
        """Расчет динамического размера позиции на основе конфигурации актива и уверенности"""
        try:
            # Получаем конфигурацию для конкретного актива
            config = self.asset_configs.get(symbol, {
                'base_position_size': 0.10,
                'winrate_multiplier': 1.0
            })
            
            base_size = config['base_position_size']
            winrate_multiplier = config['winrate_multiplier']
            
            # Корректировка на основе уверенности модели (0.0 - 1.0)
            confidence_multiplier = 0.5 + (confidence * 0.5)  # От 0.5 до 1.0
            
            # Финальный размер позиции
            final_size = base_size * winrate_multiplier * confidence_multiplier
            
            # Ограничиваем размер позиции (минимум 2%, максимум 20%)
            final_size = max(0.02, min(0.20, final_size))
            
            logger.debug(f"Размер позиции для {symbol}: базовый={base_size:.3f}, винрейт_множитель={winrate_multiplier:.2f}, уверенность_множитель={confidence_multiplier:.2f}, финальный={final_size:.3f}")
            
            return final_size
            
        except Exception as e:
            logger.error(f"Ошибка расчета размера позиции: {e}")
            return 0.10  # Возвращаем стандартный размер в случае ошибки

    def _check_volatility_filter(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Устаревший метод - оставлен для совместимости"""
        # Используем адаптивный фильтр с дефолтными параметрами
        return self._check_adaptive_volatility_filter('DEFAULT', data)

    def _create_fallback_signal(self, module_name: str, error_msg: str) -> AISignal:
        """Создание резервного сигнала при ошибке модуля"""
        return AISignal(
            module_name=module_name,
            signal_type='HOLD',
            confidence=0.1,
            data={'error': error_msg},
            timestamp=datetime.now(),
            reasoning=f"Ошибка в модуле {module_name}: {error_msg}"
        )
    
    def _summarize_market_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        return {
            'current_price': float(data['close'].iloc[-1]),
            'price_change_24h': float((data['close'].iloc[-1] - data['close'].iloc[-24]) / data['close'].iloc[-24] * 100) if len(data) >= 24 else 0,
            'volume_24h': float(data['volume'].iloc[-24:].sum()) if len(data) >= 24 else float(data['volume'].sum()),
            'high_24h': float(data['high'].iloc[-24:].max()) if len(data) >= 24 else float(data['high'].max()),
            'low_24h': float(data['low'].iloc[-24:].min()) if len(data) >= 24 else float(data['low'].min()),
            'volatility': float(data['close'].pct_change().std() * 100)
        }
    
    def _create_safe_decision(self, error_msg: str) -> TradingDecision:
        """Создание безопасного решения при ошибке"""
        return TradingDecision(
            action='HOLD',
            confidence=0.0,
            entry_price=None,
            position_size=0.0,
            stop_loss=0.0,
            take_profits=[],
            dynamic_stop={},
            reasoning=f'Безопасное решение из-за ошибки: {error_msg}',
            risk_score=1.0,
            timestamp=datetime.now()
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Получение статуса системы"""
        ai_modules = {
            'lava_ai': {'status': 'ready' if hasattr(self, 'lava_ai') else 'not_initialized'},
            'trading_ai': {'status': 'ready' if hasattr(self, 'trading_ai') else 'not_initialized'},
            'lgbm_ai': {'status': 'ready' if hasattr(self, 'lgbm_ai') else 'not_initialized'},
            'mistral_ai': {'status': 'ready' if hasattr(self, 'mistral_ai') else 'not_initialized'},
            'ai_manager': {'status': 'ready' if hasattr(self, 'ai_manager') else 'not_initialized'}
        }
        
        return {
            'status': 'active' if self.is_initialized else 'inactive',
            'initialized': self.is_initialized,
            'ai_modules': ai_modules,
            'modules': ai_modules,  # Для обратной совместимости
            'signal_history_count': len(self.signal_history),
            'decision_history_count': len(self.decision_history),
            'module_weights': self.module_weights
        }
    
    async def cleanup(self):
        """Очистка ресурсов"""
        try:
            cleanup_tasks = []
            
            # Очистка всех AI модулей
            if hasattr(self, 'lava_ai') and hasattr(self.lava_ai, 'cleanup'):
                cleanup_tasks.append(self.lava_ai.cleanup())
            
            if hasattr(self, 'trading_ai') and hasattr(self.trading_ai, 'cleanup'):
                cleanup_tasks.append(self.trading_ai.cleanup())
                
            if hasattr(self, 'lgbm_ai') and hasattr(self.lgbm_ai, 'cleanup'):
                cleanup_tasks.append(self.lgbm_ai.cleanup())
                
            if hasattr(self, 'mistral_ai') and hasattr(self.mistral_ai, 'cleanup'):
                cleanup_tasks.append(self.mistral_ai.cleanup())
                
            if hasattr(self, 'ai_manager') and hasattr(self.ai_manager, 'cleanup'):
                cleanup_tasks.append(self.ai_manager.cleanup())
            
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
                
            logger.info("✅ MultiAI Orchestrator очищен")
        except Exception as e:
            logger.error(f"Ошибка при очистке: {e}")

    def _calculate_solusdt_risk_params(self, data: pd.DataFrame, signal_type: str, confidence: float) -> Dict[str, float]:
        """Специальный расчет параметров риска для SOLUSDT"""
        try:
            # Анализ ATR для динамического стоп-лосса
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift())
            low_close = abs(data['low'] - data['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean().iloc[-1]
            
            current_price = data['close'].iloc[-1]
            atr_percent = (atr / current_price) * 100
            
            # Базовые параметры с учетом анализа неудачных сделок SOLUSDT
            base_stop_loss = 3.0  # Увеличиваем с 2% до 3% (многие сделки закрылись слишком быстро)
            base_take_profit_1 = 2.5  # Первый тейк-профит ближе
            base_take_profit_2 = 5.0  # Второй тейк-профит дальше
            base_take_profit_3 = 8.0  # Третий тейк-профит для сильных движений
            
            # Адаптация под волатильность
            volatility_multiplier = max(0.8, min(1.5, atr_percent / 2.0))
            
            # Адаптация под уверенность сигнала
            confidence_multiplier = 0.7 + (confidence * 0.6)  # От 0.7 до 1.3
            
            # Адаптация под тип сигнала
            signal_multiplier = 1.0
            if signal_type == 'BUY':
                # Для покупок в SOLUSDT более консервативный подход
                signal_multiplier = 0.9
            elif signal_type == 'SELL':
                # Для продаж можно быть более агрессивным
                signal_multiplier = 1.1
            
            # Финальные параметры
            stop_loss = base_stop_loss * volatility_multiplier * signal_multiplier
            take_profit_1 = base_take_profit_1 * confidence_multiplier * signal_multiplier
            take_profit_2 = base_take_profit_2 * confidence_multiplier * signal_multiplier
            take_profit_3 = base_take_profit_3 * confidence_multiplier * signal_multiplier
            
            # Ограничиваем экстремальные значения
            stop_loss = max(2.0, min(5.0, stop_loss))
            take_profit_1 = max(1.5, min(4.0, take_profit_1))
            take_profit_2 = max(3.0, min(8.0, take_profit_2))
            take_profit_3 = max(5.0, min(12.0, take_profit_3))
            
            logger.info(f"SOLUSDT риск-параметры: SL={stop_loss:.2f}%, TP1={take_profit_1:.2f}%, TP2={take_profit_2:.2f}%, TP3={take_profit_3:.2f}%")
            
            return {
                'stop_loss_percent': stop_loss,
                'take_profit_1_percent': take_profit_1,
                'take_profit_2_percent': take_profit_2,
                'take_profit_3_percent': take_profit_3,
                'atr_percent': atr_percent,
                'volatility_multiplier': volatility_multiplier,
                'confidence_multiplier': confidence_multiplier
            }
            
        except Exception as e:
            logger.error(f"Ошибка расчета SOLUSDT риск-параметров: {e}")
            # Возвращаем консервативные значения по умолчанию
            return {
                'stop_loss_percent': 3.0,
                'take_profit_1_percent': 2.5,
                'take_profit_2_percent': 5.0,
                'take_profit_3_percent': 8.0,
                'atr_percent': 2.0,
                'volatility_multiplier': 1.0,
                'confidence_multiplier': 1.0
            }

    def _create_trading_decision(self, symbol: str, data: pd.DataFrame, 
                               final_signal: str, confidence: float, 
                               reasoning: str, signals_data: dict) -> TradingDecision:
        """
        Создание торгового решения с улучшенным риск-менеджментом
        """
        try:
            # Получение конфигурации для актива
            config = self.asset_configs.get(symbol, self.asset_configs['default'])
            
            # Проверка минимального порога уверенности для проблемных активов (снижен для более активной торговли)
            min_confidence = config.get('min_confidence_threshold', 0.35)  # Снижен с 0.6 до 0.35
            if confidence < min_confidence:
                return self._create_safe_decision(
                    f"❌ Сигнал отклонен для {symbol}: уверенность {confidence:.1%} < минимального порога {min_confidence:.1%}"
                )
            
            # Создаем market_data из данных DataFrame
            market_data = {
                'close': data['close'].iloc[-1] if not data.empty else 0,
                'high': data['high'].iloc[-1] if not data.empty else 0,
                'low': data['low'].iloc[-1] if not data.empty else 0,
                'volume': data['volume'].iloc[-1] if not data.empty else 0
            }
            
            # Расчет динамических параметров риска
            dynamic_risk = self._calculate_dynamic_risk_parameters(symbol, market_data)
            
            # Базовый размер позиции с учетом динамической корректировки
            base_position_size = config['base_position_size']
            position_size_multiplier = dynamic_risk['position_size_multiplier']
            winrate_multiplier = config.get('winrate_multiplier', 1.0)
            
            # Итоговый размер позиции
            position_size = base_position_size * position_size_multiplier * winrate_multiplier
            position_size = max(0.05, min(0.20, position_size))  # Ограничения 5-20%
            
            # Получение текущей цены
            current_price = market_data.get('close', 0)
            if current_price <= 0:
                return self._create_safe_decision("Некорректная цена актива")
            
            # Корректировка уровней с учетом комиссий
            commission_adjusted = self._adjust_levels_for_commission(
                symbol, current_price, 
                dynamic_risk['stop_loss_percent'], 
                dynamic_risk['take_profit_levels'],
                position_size
            )
            
            # Финальные параметры риск-менеджмента
            stop_loss_percent = commission_adjusted['adjusted_stop_loss_percent']
            take_profit_levels = commission_adjusted['adjusted_take_profit_levels']
            
            # Расчет уровней стоп-лосса и тейк-профитов
            stop_loss_price = current_price * (1 - stop_loss_percent / 100)
            take_profit_prices = [current_price * (1 + tp / 100) for tp in take_profit_levels]
            
            # Проверка минимального соотношения риск/прибыль
            risk_reward_ratio = dynamic_risk['risk_reward_ratio']
            min_take_profit_price = current_price * (1 + stop_loss_percent * risk_reward_ratio / 100)
            
            # Фильтрация тейк-профитов по минимальному соотношению
            valid_take_profits = [tp for tp in take_profit_prices if tp >= min_take_profit_price]
            
            if not valid_take_profits:
                # Создаем тейк-профиты на основе минимального соотношения
                valid_take_profits = [
                    min_take_profit_price,
                    min_take_profit_price * 1.25,
                    min_take_profit_price * 1.5,
                    min_take_profit_price * 1.75,
                    min_take_profit_price * 2.0
                ]
            
            # Расчет потенциальной прибыли с учетом комиссий
            estimated_profit_per_tp = []
            for tp_price in valid_take_profits:
                commission_impact = self._calculate_commission_impact(
                    symbol, position_size, current_price, tp_price, 'market'
                )
                estimated_profit_per_tp.append(commission_impact['net_pnl'])
            
            # Расчет потенциального убытка с учетом комиссий
            loss_commission_impact = self._calculate_commission_impact(
                symbol, position_size, current_price, stop_loss_price, 'market'
            )
            estimated_loss = abs(loss_commission_impact['net_pnl'])
            
            # Расширенное обоснование решения
            reasoning_parts = [
                f"🎯 Динамический риск-менеджмент для {symbol}:",
                f"📊 Волатильность: {dynamic_risk.get('volatility_ratio', 1.0):.2f}x от базовой",
                f"💰 Размер позиции: {position_size:.1%} (базовый: {base_position_size:.1%}, "
                f"корректировка: {position_size_multiplier:.2f}x, винрейт: {winrate_multiplier:.2f}x)",
                f"🛡️ Стоп-лосс: {stop_loss_percent:.2f}% (${stop_loss_price:.4f})",
                f"🎯 Соотношение риск/прибыль: минимум 1:{risk_reward_ratio:.1f}",
                f"💸 Влияние комиссий: {commission_adjusted['commission_impact_percent']:.3f}%",
                f"📈 Тейк-профиты: {len(valid_take_profits)} уровней",
                f"💵 Ожидаемый убыток: ${estimated_loss:.2f}",
                f"💰 Ожидаемая прибыль (1-й TP): ${estimated_profit_per_tp[0]:.2f}" if estimated_profit_per_tp else "",
                f"⚖️ R/R первого TP: 1:{estimated_profit_per_tp[0]/estimated_loss:.2f}" if estimated_profit_per_tp and estimated_loss > 0 else ""
            ]
            
            # Добавляем информацию о сигналах из signals_data
            signal_info = []
            aggregated_signals = signals_data.get('aggregated_signals', {})
            individual_signals = aggregated_signals.get('individual_signals', [])
            
            # individual_signals это список словарей, а не словарь
            for signal_data in individual_signals:
                if signal_data and signal_data.get('signal') != 'HOLD':
                    confidence_val = signal_data.get('confidence', 0)
                    confidence_emoji = "🔥" if confidence_val > 0.8 else "✅" if confidence_val > 0.6 else "⚠️"
                    module_name = signal_data.get('module', 'Unknown')
                    signal_type = signal_data.get('signal', 'HOLD')
                    signal_info.append(f"{confidence_emoji} {module_name}: {signal_type} ({confidence_val:.1%})")
            
            if signal_info:
                reasoning_parts.extend(["", "🤖 AI Сигналы:"] + signal_info)
            
            reasoning = "\n".join(filter(None, reasoning_parts))
            
            return TradingDecision(
                action=final_signal,
                confidence=confidence,
                entry_price=current_price,
                position_size=position_size,
                stop_loss=stop_loss_price,
                take_profits=[{'level': i+1, 'price': tp} for i, tp in enumerate(valid_take_profits[:5])],
                dynamic_stop={'enabled': True, 'trailing_percent': dynamic_risk.get('trailing_stop_percent', 2.0)},
                reasoning=reasoning,
                risk_score=1.0 - confidence,  # Обратная зависимость от уверенности
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Ошибка создания торгового решения с улучшенным риск-менеджментом: {e}")
            return self._create_safe_decision(str(e))

    def _calculate_dynamic_risk_parameters(self, symbol: str, market_data: dict) -> dict:
        """
        Динамическое управление рисками на основе волатильности и рыночных условий
        """
        config = self.asset_configs.get(symbol, self.asset_configs['default'])
        
        if not config.get('dynamic_risk_enabled', False):
            return {
                'stop_loss_percent': config['stop_loss_percent'],
                'take_profit_levels': config['take_profit_levels'],
                'position_size_multiplier': 1.0
            }
        
        # Расчет текущей волатильности (ATR)
        current_volatility = market_data.get('atr_percent', 2.0)
        base_volatility = 2.0  # Базовая волатильность
        volatility_ratio = current_volatility / base_volatility
        
        # Корректировка параметров на основе волатильности
        volatility_multiplier = config.get('volatility_multiplier', 1.0)
        adjusted_volatility = volatility_ratio * volatility_multiplier
        
        # Динамическая корректировка стоп-лосса
        base_stop_loss = config['stop_loss_percent']
        dynamic_stop_loss = base_stop_loss * max(0.5, min(2.0, adjusted_volatility))
        
        # Динамическая корректировка тейк-профитов с соблюдением соотношения 1:2
        risk_reward_ratio = config.get('risk_reward_ratio', 2.0)
        base_take_profits = config['take_profit_levels']
        
        # Убеждаемся, что первый тейк-профит соответствует минимальному соотношению
        min_take_profit = dynamic_stop_loss * risk_reward_ratio
        dynamic_take_profits = []
        
        for tp in base_take_profits:
            adjusted_tp = tp * max(0.7, min(1.5, adjusted_volatility))
            # Проверяем минимальное соотношение риск/прибыль
            if adjusted_tp >= min_take_profit:
                dynamic_take_profits.append(adjusted_tp)
        
        # Если нет подходящих тейк-профитов, создаем их на основе минимального соотношения
        if not dynamic_take_profits:
            dynamic_take_profits = [
                min_take_profit,
                min_take_profit * 1.5,
                min_take_profit * 2.0,
                min_take_profit * 2.5,
                min_take_profit * 3.0
            ]
        
        # Корректировка размера позиции на основе волатильности
        position_size_multiplier = 1.0 / max(0.5, min(2.0, adjusted_volatility))
        
        return {
            'stop_loss_percent': dynamic_stop_loss,
            'take_profit_levels': dynamic_take_profits,
            'position_size_multiplier': position_size_multiplier,
            'volatility_ratio': volatility_ratio,
            'risk_reward_ratio': risk_reward_ratio
        }
    
    def _calculate_commission_impact(self, symbol: str, position_size: float, entry_price: float, 
                                   exit_price: float, order_type: str = 'market') -> dict:
        """
        Расчет влияния комиссий на прибыльность сделки
        """
        config = self.asset_configs.get(symbol, self.asset_configs['default'])
        
        # Выбор комиссии в зависимости от типа ордера
        if order_type == 'market':
            commission_rate = config.get('market_commission', 0.001)
        else:
            commission_rate = config.get('limit_commission', 0.001)
        
        # Расчет комиссий
        entry_commission = position_size * entry_price * commission_rate
        exit_commission = position_size * exit_price * commission_rate
        total_commission = entry_commission + exit_commission
        
        # Расчет прибыли/убытка с учетом комиссий
        gross_pnl = position_size * (exit_price - entry_price)
        net_pnl = gross_pnl - total_commission
        
        # Минимальная прибыль для покрытия комиссий
        breakeven_price_change = total_commission / position_size
        
        return {
            'entry_commission': entry_commission,
            'exit_commission': exit_commission,
            'total_commission': total_commission,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'breakeven_price_change': breakeven_price_change,
            'commission_rate': commission_rate
        }
    
    def _adjust_levels_for_commission(self, symbol: str, entry_price: float, 
                                    stop_loss_percent: float, take_profit_levels: list,
                                    position_size: float) -> dict:
        """
        Корректировка уровней стоп-лосса и тейк-профитов с учетом комиссий
        """
        config = self.asset_configs.get(symbol, self.asset_configs['default'])
        
        # Расчет минимального изменения цены для покрытия комиссий
        commission_impact = self._calculate_commission_impact(
            symbol, position_size, entry_price, entry_price * 1.01, 'market'
        )
        
        min_price_change_percent = (commission_impact['breakeven_price_change'] / entry_price) * 100
        
        # Корректировка стоп-лосса (увеличиваем для учета комиссий)
        adjusted_stop_loss = stop_loss_percent + (min_price_change_percent * 0.5)
        
        # Корректировка тейк-профитов (увеличиваем для учета комиссий)
        adjusted_take_profits = []
        for tp in take_profit_levels:
            adjusted_tp = tp + min_price_change_percent
            # Проверяем соблюдение минимального соотношения риск/прибыль
            risk_reward_ratio = config.get('risk_reward_ratio', 2.0)
            min_tp = adjusted_stop_loss * risk_reward_ratio
            
            if adjusted_tp >= min_tp:
                adjusted_take_profits.append(adjusted_tp)
        
        # Если после корректировки не осталось подходящих тейк-профитов
        if not adjusted_take_profits:
            min_tp = adjusted_stop_loss * config.get('risk_reward_ratio', 2.0)
            adjusted_take_profits = [
                min_tp,
                min_tp * 1.5,
                min_tp * 2.0,
                min_tp * 2.5,
                min_tp * 3.0
            ]
        
        return {
            'adjusted_stop_loss_percent': adjusted_stop_loss,
            'adjusted_take_profit_levels': adjusted_take_profits,
            'commission_impact_percent': min_price_change_percent,
            'original_stop_loss': stop_loss_percent,
            'original_take_profits': take_profit_levels
        }

    # ==================== REINFORCEMENT LEARNING METHODS ====================
    
    async def apply_trade_result(self, symbol: str, action: str, pnl: float, confidence: float, 
                          entry_price: float, exit_price: float, duration_minutes: int = None):
        """
        Применить результат сделки для обучения с подкреплением
        """
        if not self.reinforcement_learning or not self.rl_engine:
            return
        
        try:
            # Создаем результат сделки
            trade_result = {
                'symbol': symbol,
                'action': action,
                'pnl': pnl,
                'confidence': confidence,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'duration_minutes': duration_minutes,
                'timestamp': datetime.now()
            }
            
            # Добавляем в историю
            self.trade_results_history.append(trade_result)
            
            # Применяем обучение с подкреплением
            if pnl > 0:
                # Поощрение за прибыльную сделку - нужно передать model_name первым параметром
                # Используем "multi_ai_orchestrator" как имя модели для этого контекста
                await self.rl_engine.apply_reward("multi_ai_orchestrator", pnl, confidence)
                logger.info(f"✅ Применено поощрение: PnL={pnl:.2f}, confidence={confidence*100:.1f}%")
            else:
                # Наказание за убыточную сделку - нужно передать model_name первым параметром
                await self.rl_engine.apply_punishment("multi_ai_orchestrator", abs(pnl), confidence)
                logger.info(f"❌ Применено наказание: PnL={pnl:.2f}, confidence={confidence*100:.1f}%")
            
            # Обновляем веса модулей
            self._sync_weights_from_rl_engine()
            
            # Логируем текущие веса
            current_weights = self.rl_engine.get_model_weights()
            logger.info(f"🔄 Обновленные веса: {current_weights}")
            
        except Exception as e:
            logger.error(f"Ошибка применения результата сделки для RL: {e}")
    
    def get_reinforcement_learning_stats(self) -> dict:
        """
        Получить статистику обучения с подкреплением
        """
        if not self.reinforcement_learning or not self.rl_engine:
            return {}
        
        try:
            stats = {
                'current_weights': self.rl_engine.get_model_weights(),
                'performance_metrics': self.rl_engine.get_performance_summary(),
                'total_trades': len(self.trade_results_history),
                'profitable_trades': len([t for t in self.trade_results_history if t['pnl'] > 0]),
                'losing_trades': len([t for t in self.trade_results_history if t['pnl'] <= 0]),
                'total_pnl': sum(t['pnl'] for t in self.trade_results_history),
                'average_confidence': sum(t['confidence'] for t in self.trade_results_history) / len(self.trade_results_history) if self.trade_results_history else 0
            }
            
            if stats['total_trades'] > 0:
                stats['win_rate'] = stats['profitable_trades'] / stats['total_trades']
                stats['average_pnl'] = stats['total_pnl'] / stats['total_trades']
            else:
                stats['win_rate'] = 0
                stats['average_pnl'] = 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Ошибка получения статистики RL: {e}")
            return {}
    
    def save_reinforcement_learning_session(self, session_name: str = None) -> bool:
        """
        Сохранить сессию обучения с подкреплением
        """
        if not self.reinforcement_learning or not self.rl_engine:
            return False
        
        try:
            if session_name is None:
                session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            success = self.rl_engine.end_session(session_name)
            if success:
                logger.info(f"✅ Сессия RL сохранена: {session_name}")
            else:
                logger.error(f"❌ Ошибка сохранения сессии RL: {session_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Ошибка сохранения сессии RL: {e}")
            return False
    
    def load_reinforcement_learning_session(self, session_name: str) -> bool:
        """
        Загрузить сессию обучения с подкреплением
        """
        if not self.reinforcement_learning or not self.rl_engine:
            return False
        
        try:
            success = self.rl_engine.load_session(session_name)
            if success:
                self._sync_weights_from_rl_engine()
                logger.info(f"✅ Сессия RL загружена: {session_name}")
            else:
                logger.error(f"❌ Ошибка загрузки сессии RL: {session_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Ошибка загрузки сессии RL: {e}")
            return False
    
    def reset_reinforcement_learning(self) -> bool:
        """
        Сбросить обучение с подкреплением к начальным весам
        """
        if not self.reinforcement_learning or not self.rl_engine:
            return False
        
        try:
            self.rl_engine.reset_weights()
            self._sync_weights_from_rl_engine()
            self.trade_results_history.clear()
            logger.info("🔄 Обучение с подкреплением сброшено к начальным весам")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка сброса RL: {e}")
            return False
    
    def _sync_weights_from_rl_engine(self):
        """
        Синхронизировать веса модулей из RL движка
        """
        if not self.rl_engine:
            return
        
        try:
            rl_weights = self.rl_engine.get_model_weights()
            
            # Обновляем веса модулей
            self.module_weights.update(rl_weights)
            
            # Нормализуем веса
            total_weight = sum(self.module_weights.values())
            if total_weight > 0:
                for module in self.module_weights:
                    self.module_weights[module] = self.module_weights[module] / total_weight
            
            logger.debug(f"Веса синхронизированы: {self.module_weights}")
            
        except Exception as e:
            logger.error(f"Ошибка синхронизации весов: {e}")
    
    def get_mistral_server_status(self) -> dict:
        """
        Получить статус Mistral сервера
        """
        if not self.mistral_server_manager:
            return {'status': 'not_initialized', 'message': 'Mistral server manager not initialized'}
        
        try:
            return self.mistral_server_manager.get_server_status()
        except Exception as e:
            logger.error(f"Ошибка получения статуса Mistral сервера: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def start_mistral_server(self) -> bool:
        """
        Запустить Mistral сервер
        """
        if not self.mistral_server_manager:
            logger.error("Mistral server manager не инициализирован")
            return False
        
        try:
            return self.mistral_server_manager.start_server()
        except Exception as e:
            logger.error(f"Ошибка запуска Mistral сервера: {e}")
            return False
    
    def stop_mistral_server(self) -> bool:
        """
        Остановить Mistral сервер
        """
        if not self.mistral_server_manager:
            logger.error("Mistral server manager не инициализирован")
            return False
        
        try:
            return self.mistral_server_manager.stop_server()
        except Exception as e:
            logger.error(f"Ошибка остановки Mistral сервера: {e}")
            return False