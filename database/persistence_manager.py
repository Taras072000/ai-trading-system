#!/usr/bin/env python3
"""
Persistence Manager for Reinforcement Learning System
Менеджер персистентности для системы обучения с подкреплением
"""

import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .reinforcement_learning_db import (
    ReinforcementLearningDatabase,
    SessionRecord,
    TradeRecord,
    WeightEvolutionRecord
)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PersistenceConfig:
    """Конфигурация системы персистентности"""
    db_path: str = "data/reinforcement_learning.db"
    auto_backup: bool = True
    backup_interval_hours: int = 24
    max_backups: int = 30
    evolution_snapshot_interval: int = 10  # Каждые N сделок
    enable_compression: bool = True

class PersistenceManager:
    """
    Менеджер персистентности для системы обучения с подкреплением
    Обеспечивает сохранение и загрузку данных обучения
    """
    
    def __init__(self, config: PersistenceConfig):
        self.config = config
        self.db = ReinforcementLearningDatabase(config.db_path)
        self.current_session_id: Optional[str] = None
        self.trade_counter = 0
        self.last_backup_time = datetime.now()
        
        logger.info("✅ Менеджер персистентности инициализирован")
    
    def start_session(self, 
                     session_name: str,
                     profile_name: str,
                     initial_weights: Dict[str, float],
                     config_snapshot: Dict[str, Any]) -> str:
        """Начало новой сессии обучения"""
        try:
            session_id = f"rl_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            session_record = SessionRecord(
                session_id=session_id,
                session_name=session_name,
                start_time=datetime.now(),
                end_time=None,
                profile_name=profile_name,
                total_trades=0,
                profitable_trades=0,
                total_pnl=0.0,
                win_rate=0.0,
                initial_weights=initial_weights.copy(),
                final_weights={},
                config_snapshot=config_snapshot.copy(),
                status='running',
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            if self.db.create_session(session_record):
                self.current_session_id = session_id
                self.trade_counter = 0
                
                # Создаем первую запись эволюции весов
                self._save_weight_evolution(
                    ai_weights=initial_weights,
                    win_rate=0.0,
                    total_pnl=0.0,
                    learning_metrics={}
                )
                
                logger.info(f"✅ Начата сессия обучения: {session_id}")
                return session_id
            else:
                logger.error(f"❌ Не удалось создать сессию: {session_id}")
                return ""
                
        except Exception as e:
            logger.error(f"❌ Ошибка начала сессии: {e}")
            return ""
    
    def end_session(self, 
                   final_weights: Dict[str, float],
                   total_trades: int,
                   profitable_trades: int,
                   total_pnl: float,
                   win_rate: float) -> bool:
        """Завершение текущей сессии"""
        try:
            if not self.current_session_id:
                logger.warning("⚠️ Нет активной сессии для завершения")
                return False
            
            updates = {
                'end_time': datetime.now(),
                'final_weights': final_weights,
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'status': 'completed'
            }
            
            if self.db.update_session(self.current_session_id, updates):
                # Сохраняем финальную эволюцию весов
                self._save_weight_evolution(
                    ai_weights=final_weights,
                    win_rate=win_rate,
                    total_pnl=total_pnl,
                    learning_metrics={'session_completed': True}
                )
                
                logger.info(f"✅ Завершена сессия: {self.current_session_id}")
                
                # Создаем резервную копию
                if self.config.auto_backup:
                    self._create_backup_if_needed()
                
                self.current_session_id = None
                self.trade_counter = 0
                return True
            else:
                logger.error(f"❌ Не удалось завершить сессию: {self.current_session_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка завершения сессии: {e}")
            return False
    
    def save_trade_result(self,
                         symbol: str,
                         action: str,
                         entry_price: float,
                         exit_price: float,
                         entry_time: datetime,
                         exit_time: datetime,
                         pnl: float,
                         pnl_percent: float,
                         confidence: float,
                         ai_weights_before: Dict[str, float],
                         ai_weights_after: Dict[str, float],
                         reward_applied: float = 0.0,
                         punishment_applied: float = 0.0) -> bool:
        """Сохранение результата сделки"""
        try:
            if not self.current_session_id:
                logger.warning("⚠️ Нет активной сессии для сохранения сделки")
                return False
            
            trade_id = f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            duration_minutes = int((exit_time - entry_time).total_seconds() / 60)
            is_profitable = pnl > 0
            
            trade_record = TradeRecord(
                trade_id=trade_id,
                session_id=self.current_session_id,
                symbol=symbol,
                action=action,
                entry_price=entry_price,
                exit_price=exit_price,
                entry_time=entry_time,
                exit_time=exit_time,
                pnl=pnl,
                pnl_percent=pnl_percent,
                confidence=confidence,
                duration_minutes=duration_minutes,
                ai_weights_before=ai_weights_before.copy(),
                ai_weights_after=ai_weights_after.copy(),
                reward_applied=reward_applied,
                punishment_applied=punishment_applied,
                is_profitable=is_profitable,
                created_at=datetime.now()
            )
            
            if self.db.add_trade(trade_record):
                self.trade_counter += 1
                
                # Сохраняем эволюцию весов через определенные интервалы
                if self.trade_counter % self.config.evolution_snapshot_interval == 0:
                    # Получаем текущую статистику сессии
                    session_stats = self._calculate_session_stats()
                    self._save_weight_evolution(
                        ai_weights=ai_weights_after,
                        win_rate=session_stats['win_rate'],
                        total_pnl=session_stats['total_pnl'],
                        learning_metrics=session_stats['metrics']
                    )
                
                logger.debug(f"✅ Сохранена сделка: {trade_id}")
                return True
            else:
                logger.error(f"❌ Не удалось сохранить сделку: {trade_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения сделки: {e}")
            return False
    
    def _save_weight_evolution(self,
                              ai_weights: Dict[str, float],
                              win_rate: float,
                              total_pnl: float,
                              learning_metrics: Dict[str, Any]) -> bool:
        """Сохранение эволюции весов"""
        try:
            if not self.current_session_id:
                return False
            
            evolution_id = f"evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            evolution_record = WeightEvolutionRecord(
                evolution_id=evolution_id,
                session_id=self.current_session_id,
                timestamp=datetime.now(),
                trade_count=self.trade_counter,
                ai_weights=ai_weights.copy(),
                win_rate=win_rate,
                total_pnl=total_pnl,
                learning_metrics=learning_metrics.copy(),
                created_at=datetime.now()
            )
            
            return self.db.add_weight_evolution(evolution_record)
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения эволюции весов: {e}")
            return False
    
    def _calculate_session_stats(self) -> Dict[str, Any]:
        """Расчет статистики текущей сессии"""
        try:
            if not self.current_session_id:
                return {'win_rate': 0.0, 'total_pnl': 0.0, 'metrics': {}}
            
            trades = self.db.get_session_trades(self.current_session_id)
            
            if not trades:
                return {'win_rate': 0.0, 'total_pnl': 0.0, 'metrics': {}}
            
            total_trades = len(trades)
            profitable_trades = sum(1 for trade in trades if trade.is_profitable)
            win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0.0
            total_pnl = sum(trade.pnl for trade in trades)
            avg_confidence = sum(trade.confidence for trade in trades) / total_trades
            avg_duration = sum(trade.duration_minutes for trade in trades) / total_trades
            
            # Статистика по символам
            symbol_stats = {}
            for trade in trades:
                if trade.symbol not in symbol_stats:
                    symbol_stats[trade.symbol] = {'count': 0, 'profitable': 0, 'pnl': 0.0}
                
                symbol_stats[trade.symbol]['count'] += 1
                if trade.is_profitable:
                    symbol_stats[trade.symbol]['profitable'] += 1
                symbol_stats[trade.symbol]['pnl'] += trade.pnl
            
            metrics = {
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'avg_confidence': avg_confidence,
                'avg_duration_minutes': avg_duration,
                'symbol_statistics': symbol_stats,
                'last_updated': datetime.now().isoformat()
            }
            
            return {
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета статистики сессии: {e}")
            return {'win_rate': 0.0, 'total_pnl': 0.0, 'metrics': {}}
    
    def update_session_progress(self,
                               total_trades: int,
                               profitable_trades: int,
                               total_pnl: float,
                               win_rate: float) -> bool:
        """Обновление прогресса сессии"""
        try:
            if not self.current_session_id:
                logger.warning("⚠️ Нет активной сессии для обновления")
                return False
            
            updates = {
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'total_pnl': total_pnl,
                'win_rate': win_rate
            }
            
            return self.db.update_session(self.current_session_id, updates)
            
        except Exception as e:
            logger.error(f"❌ Ошибка обновления прогресса сессии: {e}")
            return False
    
    def get_session_data(self, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Получение данных сессии"""
        try:
            target_session_id = session_id or self.current_session_id
            if not target_session_id:
                logger.warning("⚠️ Не указан ID сессии")
                return None
            
            session = self.db.get_session(target_session_id)
            if not session:
                logger.warning(f"⚠️ Сессия не найдена: {target_session_id}")
                return None
            
            trades = self.db.get_session_trades(target_session_id)
            weight_evolution = self.db.get_weight_evolution(target_session_id)
            
            return {
                'session': session,
                'trades': trades,
                'weight_evolution': weight_evolution,
                'trade_count': len(trades),
                'evolution_count': len(weight_evolution)
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения данных сессии: {e}")
            return None
    
    def get_recent_sessions(self, limit: int = 10) -> List[SessionRecord]:
        """Получение последних сессий"""
        try:
            return self.db.get_recent_sessions(limit)
        except Exception as e:
            logger.error(f"❌ Ошибка получения последних сессий: {e}")
            return []
    
    def get_best_performing_weights(self, days: int = 30) -> Optional[Dict[str, float]]:
        """Получение весов с лучшей производительностью"""
        try:
            recent_sessions = self.db.get_recent_sessions(100)  # Больше сессий для анализа
            
            if not recent_sessions:
                return None
            
            # Фильтруем завершенные сессии за указанный период
            cutoff_date = datetime.now() - timedelta(days=days)
            completed_sessions = [
                session for session in recent_sessions
                if session.status == 'completed' and session.start_time >= cutoff_date
            ]
            
            if not completed_sessions:
                return None
            
            # Находим сессию с лучшим винрейтом
            best_session = max(completed_sessions, key=lambda s: s.win_rate)
            
            logger.info(f"✅ Найдена лучшая сессия: {best_session.session_id} (винрейт: {best_session.win_rate:.2f}%)")
            return best_session.final_weights
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения лучших весов: {e}")
            return None
    
    def export_session(self, session_id: str, export_path: str) -> bool:
        """Экспорт данных сессии"""
        try:
            return self.db.export_session_data(session_id, export_path)
        except Exception as e:
            logger.error(f"❌ Ошибка экспорта сессии {session_id}: {e}")
            return False
    
    def _create_backup_if_needed(self) -> bool:
        """Создание резервной копии при необходимости"""
        try:
            now = datetime.now()
            hours_since_backup = (now - self.last_backup_time).total_seconds() / 3600
            
            if hours_since_backup >= self.config.backup_interval_hours:
                backup_path = self.db.create_backup()
                if backup_path:
                    self.last_backup_time = now
                    self.db.cleanup_old_backups(self.config.max_backups)
                    logger.info(f"✅ Создана резервная копия: {backup_path}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания резервной копии: {e}")
            return False
    
    def get_performance_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Получение статистики производительности"""
        try:
            return self.db.get_performance_statistics(days)
        except Exception as e:
            logger.error(f"❌ Ошибка получения статистики: {e}")
            return {}
    
    def get_database_info(self) -> Dict[str, Any]:
        """Получение информации о базе данных"""
        try:
            return self.db.get_database_info()
        except Exception as e:
            logger.error(f"❌ Ошибка получения информации о БД: {e}")
            return {}
    
    def cleanup_old_data(self, days: int = 90) -> bool:
        """Очистка старых данных"""
        try:
            # Эта функция может быть реализована для удаления старых данных
            # Пока оставляем заглушку
            logger.info(f"🧹 Очистка данных старше {days} дней (не реализовано)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка очистки старых данных: {e}")
            return False
    
    def get_current_session_id(self) -> Optional[str]:
        """Получение ID текущей сессии"""
        return self.current_session_id
    
    def is_session_active(self) -> bool:
        """Проверка активности сессии"""
        return self.current_session_id is not None

# Пример использования
def main():
    """Пример использования менеджера персистентности"""
    config = PersistenceConfig(
        db_path="data/test_rl.db",
        auto_backup=True,
        evolution_snapshot_interval=5
    )
    
    manager = PersistenceManager(config)
    
    # Начинаем сессию
    session_id = manager.start_session(
        session_name="Test Session",
        profile_name="balanced",
        initial_weights={'lava_ai': 0.3, 'trading_ai': 0.3, 'lgbm_ai': 0.4},
        config_snapshot={'learning_rate': 0.01}
    )
    
    print(f"Начата сессия: {session_id}")
    
    # Сохраняем несколько сделок
    for i in range(3):
        manager.save_trade_result(
            symbol="BTCUSDT",
            action="LONG",
            entry_price=50000.0,
            exit_price=50100.0,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            pnl=100.0,
            pnl_percent=0.2,
            confidence=0.8,
            ai_weights_before={'lava_ai': 0.3, 'trading_ai': 0.3, 'lgbm_ai': 0.4},
            ai_weights_after={'lava_ai': 0.31, 'trading_ai': 0.29, 'lgbm_ai': 0.4}
        )
    
    # Завершаем сессию
    manager.end_session(
        final_weights={'lava_ai': 0.32, 'trading_ai': 0.28, 'lgbm_ai': 0.4},
        total_trades=3,
        profitable_trades=3,
        total_pnl=300.0,
        win_rate=100.0
    )
    
    print("Сессия завершена")
    
    # Получаем статистику
    stats = manager.get_performance_statistics(30)
    print(f"Статистика: {stats}")

if __name__ == "__main__":
    main()