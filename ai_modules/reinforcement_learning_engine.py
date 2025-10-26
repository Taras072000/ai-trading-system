#!/usr/bin/env python3
"""
Система адаптивного обучения с подкреплением для торговой системы Peper Binance v4
Основной движок обучения с подкреплением для AI моделей
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class ReinforcementConfig:
    """Конфигурация системы обучения с подкреплением"""
    learning_rate: float = 0.01
    reward_multiplier: float = 1.5
    punishment_multiplier: float = 0.8
    weight_decay: float = 0.001
    min_weight: float = 0.05
    max_weight: float = 0.70
    max_iterations: int = 10
    target_win_rate: float = 0.65
    min_trades_per_symbol: int = 5

@dataclass
class WeightChange:
    """Запись об изменении весов модели"""
    change_id: str
    session_id: str
    model_id: str
    old_weight: float
    new_weight: float
    timestamp: datetime
    reason: str
    trade_context: Dict[str, Any]

@dataclass
class ReinforcementSession:
    """Сессия обучения с подкреплением"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    config: ReinforcementConfig
    status: str  # 'running', 'completed', 'failed', 'stopped'
    initial_weights: Dict[str, float]
    final_weights: Dict[str, float]
    performance_metrics: Dict[str, Any]

class ReinforcementLearningEngine:
    """
    Основной движок обучения с подкреплением
    Управляет весами AI моделей на основе результатов торговли
    """
    
    def __init__(self, config: ReinforcementConfig = None):
        self.config = config or ReinforcementConfig()
        
        # Начальные веса AI моделей (сумма должна быть 1.0)
        self.model_weights = {
            'trading_ai': 0.25,
            'lava_ai': 0.35, 
            'lgbm_ai': 0.40,
            'mistral_ai': 0.0  # Начинаем с 0, будет увеличиваться при хороших результатах
        }
        
        # История изменений весов
        self.weight_history: List[WeightChange] = []
        
        # Метрики производительности для каждой модели
        self.model_performance = {
            'trading_ai': {'wins': 0, 'losses': 0, 'total_pnl': 0.0, 'trades': []},
            'lava_ai': {'wins': 0, 'losses': 0, 'total_pnl': 0.0, 'trades': []},
            'lgbm_ai': {'wins': 0, 'losses': 0, 'total_pnl': 0.0, 'trades': []},
            'mistral_ai': {'wins': 0, 'losses': 0, 'total_pnl': 0.0, 'trades': []}
        }
        
        # Текущая сессия обучения
        self.current_session: Optional[ReinforcementSession] = None
        
        logger.info("🧠 ReinforcementLearningEngine инициализирован")
        logger.info(f"📊 Начальные веса моделей: {self.model_weights}")
    
    def start_session(self, session_id: str = None) -> str:
        """Начать новую сессию обучения с подкреплением"""
        if session_id is None:
            session_id = f"rl_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        self.current_session = ReinforcementSession(
            session_id=session_id,
            start_time=datetime.now(),
            end_time=None,
            config=self.config,
            status='running',
            initial_weights=self.model_weights.copy(),
            final_weights={},
            performance_metrics={}
        )
        
        logger.info(f"🚀 Начата сессия обучения с подкреплением: {session_id}")
        return session_id
    
    def end_session(self) -> Optional[ReinforcementSession]:
        """Завершить текущую сессию обучения"""
        if not self.current_session:
            logger.warning("⚠️ Нет активной сессии для завершения")
            return None
        
        self.current_session.end_time = datetime.now()
        self.current_session.status = 'completed'
        self.current_session.final_weights = self.model_weights.copy()
        self.current_session.performance_metrics = self._calculate_session_metrics()
        
        session = self.current_session
        self.current_session = None
        
        logger.info(f"✅ Сессия обучения завершена: {session.session_id}")
        logger.info(f"📈 Финальные веса: {session.final_weights}")
        
        return session
    
    async def apply_reward(self, model_name: str, trade_pnl: float, confidence: float, trade_context: Dict[str, Any] = None) -> bool:
        """
        Применение поощрения за прибыльную сделку
        
        Args:
            model_name: Название AI модели
            trade_pnl: Прибыль/убыток от сделки
            confidence: Уверенность модели в сигнале
            trade_context: Контекст сделки (символ, время, цена и т.д.)
        """
        if model_name not in self.model_weights:
            logger.warning(f"⚠️ Неизвестная модель: {model_name}")
            return False
        
        if trade_pnl <= 0:
            logger.warning(f"⚠️ Попытка поощрения за убыточную сделку: {trade_pnl}")
            return False
        
        # Рассчитываем размер поощрения на основе прибыли и уверенности
        reward_factor = self._calculate_reward_factor(trade_pnl, confidence)
        old_weight = self.model_weights[model_name]
        
        # Увеличиваем вес модели
        weight_increase = self.config.learning_rate * reward_factor * self.config.reward_multiplier
        new_weight = min(old_weight + weight_increase, self.config.max_weight)
        
        # Применяем изменение
        self.model_weights[model_name] = new_weight
        
        # Нормализуем веса
        await self.normalize_weights()
        
        # Записываем изменение
        change = WeightChange(
            change_id=str(uuid.uuid4()),
            session_id=self.current_session.session_id if self.current_session else "no_session",
            model_id=model_name,
            old_weight=old_weight,
            new_weight=self.model_weights[model_name],
            timestamp=datetime.now(),
            reason=f"reward_pnl_{trade_pnl:.4f}_conf_{confidence*100:.1f}%",
            trade_context=trade_context or {}
        )
        self.weight_history.append(change)
        
        # Обновляем метрики производительности
        self.model_performance[model_name]['wins'] += 1
        self.model_performance[model_name]['total_pnl'] += trade_pnl
        self.model_performance[model_name]['trades'].append({
            'pnl': trade_pnl,
            'confidence': confidence,
            'timestamp': datetime.now(),
            'type': 'win'
        })
        
        logger.info(f"🎉 Поощрение для {model_name}: {old_weight:.4f} → {self.model_weights[model_name]:.4f} (PnL: {trade_pnl:.4f}, Conf: {confidence*100:.1f}%)")
        
        return True
    
    async def apply_punishment(self, model_name: str, trade_pnl: float, confidence: float, trade_context: Dict[str, Any] = None) -> bool:
        """
        Применение наказания за убыточную сделку
        
        Args:
            model_name: Название AI модели
            trade_pnl: Прибыль/убыток от сделки
            confidence: Уверенность модели в сигнале
            trade_context: Контекст сделки (символ, время, цена и т.д.)
        """
        if model_name not in self.model_weights:
            logger.warning(f"⚠️ Неизвестная модель: {model_name}")
            return False
        
        if trade_pnl >= 0:
            logger.warning(f"⚠️ Попытка наказания за прибыльную сделку: {trade_pnl}")
            return False
        
        # Рассчитываем размер наказания на основе убытка и уверенности
        punishment_factor = self._calculate_punishment_factor(trade_pnl, confidence)
        old_weight = self.model_weights[model_name]
        
        # Уменьшаем вес модели
        weight_decrease = self.config.learning_rate * punishment_factor * self.config.punishment_multiplier
        new_weight = max(old_weight - weight_decrease, self.config.min_weight)
        
        # Применяем изменение
        self.model_weights[model_name] = new_weight
        
        # Нормализуем веса
        await self.normalize_weights()
        
        # Записываем изменение
        change = WeightChange(
            change_id=str(uuid.uuid4()),
            session_id=self.current_session.session_id if self.current_session else "no_session",
            model_id=model_name,
            old_weight=old_weight,
            new_weight=self.model_weights[model_name],
            timestamp=datetime.now(),
            reason=f"punishment_pnl_{trade_pnl:.4f}_conf_{confidence*100:.1f}%",
            trade_context=trade_context or {}
        )
        self.weight_history.append(change)
        
        # Обновляем метрики производительности
        self.model_performance[model_name]['losses'] += 1
        self.model_performance[model_name]['total_pnl'] += trade_pnl
        self.model_performance[model_name]['trades'].append({
            'pnl': trade_pnl,
            'confidence': confidence,
            'timestamp': datetime.now(),
            'type': 'loss'
        })
        
        logger.info(f"💥 Наказание для {model_name}: {old_weight:.4f} → {self.model_weights[model_name]:.4f} (PnL: {trade_pnl:.4f}, Conf: {confidence*100:.1f}%)")
        
        return True
    
    async def normalize_weights(self):
        """Нормализация весов чтобы сумма равнялась 1.0"""
        total_weight = sum(self.model_weights.values())
        
        if total_weight == 0:
            # Если все веса стали 0, восстанавливаем равномерное распределение
            logger.warning("⚠️ Все веса стали 0, восстанавливаем равномерное распределение")
            for model in self.model_weights:
                self.model_weights[model] = 1.0 / len(self.model_weights)
        elif total_weight != 1.0:
            # Нормализуем веса
            for model in self.model_weights:
                self.model_weights[model] = self.model_weights[model] / total_weight
        
        # Применяем weight decay для предотвращения переобучения
        await self._apply_weight_decay()
    
    async def _apply_weight_decay(self):
        """Применение weight decay для предотвращения переобучения"""
        if self.config.weight_decay > 0:
            for model in self.model_weights:
                self.model_weights[model] *= (1 - self.config.weight_decay)
            
            # Перенормализуем после decay
            total_weight = sum(self.model_weights.values())
            if total_weight > 0:
                for model in self.model_weights:
                    self.model_weights[model] = self.model_weights[model] / total_weight
    
    def _calculate_reward_factor(self, trade_pnl: float, confidence: float) -> float:
        """Расчет фактора поощрения на основе прибыли и уверенности"""
        # Базовый фактор на основе прибыли (логарифмическая шкала)
        pnl_factor = np.log1p(abs(trade_pnl) * 100)  # log1p для стабильности
        
        # Фактор уверенности (квадратичная зависимость)
        confidence_factor = confidence ** 2
        
        # Комбинированный фактор
        reward_factor = pnl_factor * confidence_factor
        
        # Ограничиваем максимальный фактор
        return min(reward_factor, 5.0)
    
    def _calculate_punishment_factor(self, trade_pnl: float, confidence: float) -> float:
        """Расчет фактора наказания на основе убытка и уверенности"""
        # Базовый фактор на основе убытка (логарифмическая шкала)
        pnl_factor = np.log1p(abs(trade_pnl) * 100)  # log1p для стабильности
        
        # Фактор уверенности (линейная зависимость для наказания)
        confidence_factor = confidence
        
        # Комбинированный фактор (более агрессивное наказание за высокую уверенность в неправильном сигнале)
        punishment_factor = pnl_factor * (1 + confidence_factor)
        
        # Ограничиваем максимальный фактор
        return min(punishment_factor, 3.0)
    
    def _calculate_session_metrics(self) -> Dict[str, Any]:
        """Расчет метрик производительности для текущей сессии"""
        metrics = {}
        
        for model_name, performance in self.model_performance.items():
            total_trades = performance['wins'] + performance['losses']
            win_rate = performance['wins'] / total_trades if total_trades > 0 else 0.0
            
            metrics[model_name] = {
                'total_trades': total_trades,
                'wins': performance['wins'],
                'losses': performance['losses'],
                'win_rate': win_rate,
                'total_pnl': performance['total_pnl'],
                'avg_pnl': performance['total_pnl'] / total_trades if total_trades > 0 else 0.0,
                'weight_change': self.model_weights[model_name] - self.current_session.initial_weights[model_name]
            }
        
        # Общие метрики
        total_trades = sum(m['total_trades'] for m in metrics.values())
        total_wins = sum(m['wins'] for m in metrics.values())
        total_pnl = sum(m['total_pnl'] for m in metrics.values())
        
        metrics['overall'] = {
            'total_trades': total_trades,
            'total_wins': total_wins,
            'overall_win_rate': total_wins / total_trades if total_trades > 0 else 0.0,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / total_trades if total_trades > 0 else 0.0,
            'session_duration': (datetime.now() - self.current_session.start_time).total_seconds() / 3600  # в часах
        }
        
        return metrics
    
    def get_model_weights(self) -> Dict[str, float]:
        """Получить текущие веса моделей"""
        return self.model_weights.copy()
    
    def get_current_weights(self) -> Dict[str, float]:
        """Получить текущие веса моделей (алиас для get_model_weights)"""
        return self.get_model_weights()
    
    def set_model_weights(self, weights: Dict[str, float]) -> bool:
        """Установить веса моделей"""
        try:
            # Проверяем валидность весов
            if not all(model in weights for model in self.model_weights.keys()):
                logger.error("❌ Неполный набор весов моделей")
                return False
            
            if not all(0 <= weight <= 1 for weight in weights.values()):
                logger.error("❌ Веса должны быть в диапазоне [0, 1]")
                return False
            
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 0.001:
                logger.warning(f"⚠️ Сумма весов не равна 1.0: {total_weight}, нормализуем")
                # Нормализуем веса
                for model in weights:
                    weights[model] = weights[model] / total_weight
            
            # Применяем новые веса
            old_weights = self.model_weights.copy()
            self.model_weights = weights.copy()
            
            logger.info(f"✅ Веса моделей обновлены: {old_weights} → {self.model_weights}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка установки весов: {e}")
            return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Получить сводку производительности"""
        summary = {
            'current_weights': self.model_weights.copy(),
            'model_performance': {},
            'weight_changes_count': len(self.weight_history),
            'session_info': {
                'active': self.current_session is not None,
                'session_id': self.current_session.session_id if self.current_session else None,
                'start_time': self.current_session.start_time if self.current_session else None
            }
        }
        
        # Добавляем метрики производительности для каждой модели
        for model_name, performance in self.model_performance.items():
            total_trades = performance['wins'] + performance['losses']
            win_rate = performance['wins'] / total_trades if total_trades > 0 else 0.0
            
            summary['model_performance'][model_name] = {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_pnl': performance['total_pnl'],
                'current_weight': self.model_weights[model_name]
            }
        
        return summary
    
    def reset_performance_metrics(self):
        """Сброс метрик производительности"""
        for model in self.model_performance:
            self.model_performance[model] = {'wins': 0, 'losses': 0, 'total_pnl': 0.0, 'trades': []}
        
        logger.info("🔄 Метрики производительности сброшены")
    
    def export_session_data(self, session: ReinforcementSession) -> Dict[str, Any]:
        """Экспорт данных сессии для сохранения"""
        return {
            'session': asdict(session),
            'weight_history': [asdict(change) for change in self.weight_history],
            'model_performance': self.model_performance.copy()
        }