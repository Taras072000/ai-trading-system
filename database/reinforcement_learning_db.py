#!/usr/bin/env python3
"""
Reinforcement Learning Database System
Система персистентности для обучения с подкреплением
"""

import sqlite3
import json
import logging
import os
import shutil
import gzip
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
from contextlib import contextmanager

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SessionRecord:
    """Запись сессии обучения"""
    session_id: str
    session_name: str
    start_time: datetime
    end_time: Optional[datetime]
    profile_name: str
    total_trades: int
    profitable_trades: int
    total_pnl: float
    win_rate: float
    initial_weights: Dict[str, float]
    final_weights: Dict[str, float]
    config_snapshot: Dict[str, Any]
    status: str  # 'running', 'completed', 'failed'
    created_at: datetime
    updated_at: datetime

@dataclass
class TradeRecord:
    """Запись сделки"""
    trade_id: str
    session_id: str
    symbol: str
    action: str  # 'LONG', 'SHORT'
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_percent: float
    confidence: float
    duration_minutes: int
    ai_weights_before: Dict[str, float]
    ai_weights_after: Dict[str, float]
    reward_applied: float
    punishment_applied: float
    is_profitable: bool
    created_at: datetime

@dataclass
class WeightEvolutionRecord:
    """Запись эволюции весов"""
    evolution_id: str
    session_id: str
    timestamp: datetime
    trade_count: int
    ai_weights: Dict[str, float]
    win_rate: float
    total_pnl: float
    learning_metrics: Dict[str, float]
    created_at: datetime

class ReinforcementLearningDatabase:
    """
    База данных для системы обучения с подкреплением
    """
    
    def __init__(self, db_path: str = "data/reinforcement_learning.db"):
        self.db_path = db_path
        self.backup_dir = os.path.join(os.path.dirname(db_path), "backups")
        
        # Создаем директории
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Инициализируем базу данных
        self._initialize_database()
    
    def _initialize_database(self):
        """Инициализация базы данных"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Создаем таблицы
                self._create_sessions_table(cursor)
                self._create_trades_table(cursor)
                self._create_weight_evolution_table(cursor)
                self._create_indexes(cursor)
                
                conn.commit()
                logger.info("✅ База данных инициализирована")
                
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации базы данных: {e}")
            raise
    
    def _create_sessions_table(self, cursor):
        """Создание таблицы сессий"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                session_name TEXT NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                profile_name TEXT NOT NULL,
                total_trades INTEGER DEFAULT 0,
                profitable_trades INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0.0,
                win_rate REAL DEFAULT 0.0,
                initial_weights TEXT NOT NULL,  -- JSON
                final_weights TEXT,  -- JSON
                config_snapshot TEXT NOT NULL,  -- JSON
                status TEXT DEFAULT 'running',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    def _create_trades_table(self, cursor):
        """Создание таблицы сделок"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                entry_time TIMESTAMP NOT NULL,
                exit_time TIMESTAMP NOT NULL,
                pnl REAL NOT NULL,
                pnl_percent REAL NOT NULL,
                confidence REAL NOT NULL,
                duration_minutes INTEGER NOT NULL,
                ai_weights_before TEXT NOT NULL,  -- JSON
                ai_weights_after TEXT NOT NULL,  -- JSON
                reward_applied REAL DEFAULT 0.0,
                punishment_applied REAL DEFAULT 0.0,
                is_profitable BOOLEAN NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        """)
    
    def _create_weight_evolution_table(self, cursor):
        """Создание таблицы эволюции весов"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS weight_evolution (
                evolution_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                trade_count INTEGER NOT NULL,
                ai_weights TEXT NOT NULL,  -- JSON
                win_rate REAL NOT NULL,
                total_pnl REAL NOT NULL,
                learning_metrics TEXT,  -- JSON
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        """)
    
    def _create_indexes(self, cursor):
        """Создание индексов"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_sessions_start_time ON sessions(start_time)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status)",
            "CREATE INDEX IF NOT EXISTS idx_trades_session_id ON trades(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)",
            "CREATE INDEX IF NOT EXISTS idx_trades_is_profitable ON trades(is_profitable)",
            "CREATE INDEX IF NOT EXISTS idx_weight_evolution_session_id ON weight_evolution(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_weight_evolution_timestamp ON weight_evolution(timestamp)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
    
    @contextmanager
    def _get_connection(self):
        """Контекстный менеджер для подключения к базе данных"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row  # Для доступа к колонкам по имени
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"❌ Ошибка подключения к базе данных: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def create_session(self, session_record: SessionRecord) -> bool:
        """Создание новой сессии"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO sessions (
                        session_id, session_name, start_time, profile_name,
                        initial_weights, config_snapshot, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_record.session_id,
                    session_record.session_name,
                    session_record.start_time,
                    session_record.profile_name,
                    json.dumps(session_record.initial_weights),
                    json.dumps(session_record.config_snapshot),
                    session_record.status,
                    session_record.created_at,
                    session_record.updated_at
                ))
                
                conn.commit()
                logger.info(f"✅ Создана сессия: {session_record.session_id}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Ошибка создания сессии: {e}")
            return False
    
    def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Обновление сессии"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Подготавливаем SQL для обновления
                set_clauses = []
                values = []
                
                for key, value in updates.items():
                    if key in ['initial_weights', 'final_weights', 'config_snapshot']:
                        set_clauses.append(f"{key} = ?")
                        values.append(json.dumps(value))
                    else:
                        set_clauses.append(f"{key} = ?")
                        values.append(value)
                
                # Добавляем updated_at
                set_clauses.append("updated_at = ?")
                values.append(datetime.now())
                values.append(session_id)
                
                sql = f"UPDATE sessions SET {', '.join(set_clauses)} WHERE session_id = ?"
                
                cursor.execute(sql, values)
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"✅ Обновлена сессия: {session_id}")
                    return True
                else:
                    logger.warning(f"⚠️ Сессия не найдена: {session_id}")
                    return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка обновления сессии {session_id}: {e}")
            return False
    
    def add_trade(self, trade_record: TradeRecord) -> bool:
        """Добавление сделки"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO trades (
                        trade_id, session_id, symbol, action, entry_price, exit_price,
                        entry_time, exit_time, pnl, pnl_percent, confidence, duration_minutes,
                        ai_weights_before, ai_weights_after, reward_applied, punishment_applied,
                        is_profitable, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_record.trade_id,
                    trade_record.session_id,
                    trade_record.symbol,
                    trade_record.action,
                    trade_record.entry_price,
                    trade_record.exit_price,
                    trade_record.entry_time,
                    trade_record.exit_time,
                    trade_record.pnl,
                    trade_record.pnl_percent,
                    trade_record.confidence,
                    trade_record.duration_minutes,
                    json.dumps(trade_record.ai_weights_before),
                    json.dumps(trade_record.ai_weights_after),
                    trade_record.reward_applied,
                    trade_record.punishment_applied,
                    trade_record.is_profitable,
                    trade_record.created_at
                ))
                
                conn.commit()
                logger.debug(f"✅ Добавлена сделка: {trade_record.trade_id}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Ошибка добавления сделки: {e}")
            return False
    
    def add_weight_evolution(self, evolution_record: WeightEvolutionRecord) -> bool:
        """Добавление записи эволюции весов"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO weight_evolution (
                        evolution_id, session_id, timestamp, trade_count,
                        ai_weights, win_rate, total_pnl, learning_metrics, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    evolution_record.evolution_id,
                    evolution_record.session_id,
                    evolution_record.timestamp,
                    evolution_record.trade_count,
                    json.dumps(evolution_record.ai_weights),
                    evolution_record.win_rate,
                    evolution_record.total_pnl,
                    json.dumps(evolution_record.learning_metrics),
                    evolution_record.created_at
                ))
                
                conn.commit()
                logger.debug(f"✅ Добавлена эволюция весов: {evolution_record.evolution_id}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Ошибка добавления эволюции весов: {e}")
            return False
    
    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """Получение сессии по ID"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
                row = cursor.fetchone()
                
                if row:
                    return SessionRecord(
                        session_id=row['session_id'],
                        session_name=row['session_name'],
                        start_time=datetime.fromisoformat(row['start_time']),
                        end_time=datetime.fromisoformat(row['end_time']) if row['end_time'] else None,
                        profile_name=row['profile_name'],
                        total_trades=row['total_trades'],
                        profitable_trades=row['profitable_trades'],
                        total_pnl=row['total_pnl'],
                        win_rate=row['win_rate'],
                        initial_weights=json.loads(row['initial_weights']),
                        final_weights=json.loads(row['final_weights']) if row['final_weights'] else {},
                        config_snapshot=json.loads(row['config_snapshot']),
                        status=row['status'],
                        created_at=datetime.fromisoformat(row['created_at']),
                        updated_at=datetime.fromisoformat(row['updated_at'])
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"❌ Ошибка получения сессии {session_id}: {e}")
            return None
    
    def get_session_trades(self, session_id: str) -> List[TradeRecord]:
        """Получение всех сделок сессии"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM trades 
                    WHERE session_id = ? 
                    ORDER BY entry_time
                """, (session_id,))
                
                trades = []
                for row in cursor.fetchall():
                    trades.append(TradeRecord(
                        trade_id=row['trade_id'],
                        session_id=row['session_id'],
                        symbol=row['symbol'],
                        action=row['action'],
                        entry_price=row['entry_price'],
                        exit_price=row['exit_price'],
                        entry_time=datetime.fromisoformat(row['entry_time']),
                        exit_time=datetime.fromisoformat(row['exit_time']),
                        pnl=row['pnl'],
                        pnl_percent=row['pnl_percent'],
                        confidence=row['confidence'],
                        duration_minutes=row['duration_minutes'],
                        ai_weights_before=json.loads(row['ai_weights_before']),
                        ai_weights_after=json.loads(row['ai_weights_after']),
                        reward_applied=row['reward_applied'],
                        punishment_applied=row['punishment_applied'],
                        is_profitable=bool(row['is_profitable']),
                        created_at=datetime.fromisoformat(row['created_at'])
                    ))
                
                return trades
                
        except Exception as e:
            logger.error(f"❌ Ошибка получения сделок сессии {session_id}: {e}")
            return []
    
    def get_weight_evolution(self, session_id: str) -> List[WeightEvolutionRecord]:
        """Получение эволюции весов сессии"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM weight_evolution 
                    WHERE session_id = ? 
                    ORDER BY timestamp
                """, (session_id,))
                
                evolution = []
                for row in cursor.fetchall():
                    evolution.append(WeightEvolutionRecord(
                        evolution_id=row['evolution_id'],
                        session_id=row['session_id'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        trade_count=row['trade_count'],
                        ai_weights=json.loads(row['ai_weights']),
                        win_rate=row['win_rate'],
                        total_pnl=row['total_pnl'],
                        learning_metrics=json.loads(row['learning_metrics']) if row['learning_metrics'] else {},
                        created_at=datetime.fromisoformat(row['created_at'])
                    ))
                
                return evolution
                
        except Exception as e:
            logger.error(f"❌ Ошибка получения эволюции весов {session_id}: {e}")
            return []
    
    def get_recent_sessions(self, limit: int = 10) -> List[SessionRecord]:
        """Получение последних сессий"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM sessions 
                    ORDER BY start_time DESC 
                    LIMIT ?
                """, (limit,))
                
                sessions = []
                for row in cursor.fetchall():
                    sessions.append(SessionRecord(
                        session_id=row['session_id'],
                        session_name=row['session_name'],
                        start_time=datetime.fromisoformat(row['start_time']),
                        end_time=datetime.fromisoformat(row['end_time']) if row['end_time'] else None,
                        profile_name=row['profile_name'],
                        total_trades=row['total_trades'],
                        profitable_trades=row['profitable_trades'],
                        total_pnl=row['total_pnl'],
                        win_rate=row['win_rate'],
                        initial_weights=json.loads(row['initial_weights']),
                        final_weights=json.loads(row['final_weights']) if row['final_weights'] else {},
                        config_snapshot=json.loads(row['config_snapshot']),
                        status=row['status'],
                        created_at=datetime.fromisoformat(row['created_at']),
                        updated_at=datetime.fromisoformat(row['updated_at'])
                    ))
                
                return sessions
                
        except Exception as e:
            logger.error(f"❌ Ошибка получения последних сессий: {e}")
            return []
    
    def get_performance_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Получение статистики производительности"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Дата начала периода
                start_date = datetime.now() - timedelta(days=days)
                
                # Общая статистика
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_sessions,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_sessions,
                        AVG(win_rate) as avg_win_rate,
                        SUM(total_pnl) as total_pnl,
                        AVG(total_pnl) as avg_pnl_per_session
                    FROM sessions 
                    WHERE start_time >= ?
                """, (start_date,))
                
                session_stats = cursor.fetchone()
                
                # Статистика по сделкам
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_trades,
                        COUNT(CASE WHEN is_profitable = 1 THEN 1 END) as profitable_trades,
                        AVG(pnl) as avg_pnl_per_trade,
                        AVG(confidence) as avg_confidence,
                        AVG(duration_minutes) as avg_duration
                    FROM trades t
                    JOIN sessions s ON t.session_id = s.session_id
                    WHERE s.start_time >= ?
                """, (start_date,))
                
                trade_stats = cursor.fetchone()
                
                # Статистика по символам
                cursor.execute("""
                    SELECT 
                        symbol,
                        COUNT(*) as trades_count,
                        COUNT(CASE WHEN is_profitable = 1 THEN 1 END) as profitable_count,
                        AVG(pnl) as avg_pnl,
                        SUM(pnl) as total_pnl
                    FROM trades t
                    JOIN sessions s ON t.session_id = s.session_id
                    WHERE s.start_time >= ?
                    GROUP BY symbol
                    ORDER BY trades_count DESC
                """, (start_date,))
                
                symbol_stats = cursor.fetchall()
                
                return {
                    'period_days': days,
                    'session_statistics': dict(session_stats) if session_stats else {},
                    'trade_statistics': dict(trade_stats) if trade_stats else {},
                    'symbol_statistics': [dict(row) for row in symbol_stats]
                }
                
        except Exception as e:
            logger.error(f"❌ Ошибка получения статистики: {e}")
            return {}
    
    def export_session_data(self, session_id: str, export_path: str) -> bool:
        """Экспорт данных сессии в JSON"""
        try:
            session = self.get_session(session_id)
            if not session:
                logger.error(f"❌ Сессия не найдена: {session_id}")
                return False
            
            trades = self.get_session_trades(session_id)
            weight_evolution = self.get_weight_evolution(session_id)
            
            export_data = {
                'session': asdict(session),
                'trades': [asdict(trade) for trade in trades],
                'weight_evolution': [asdict(evolution) for evolution in weight_evolution],
                'export_timestamp': datetime.now().isoformat()
            }
            
            # Конвертируем datetime объекты в строки
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_datetime(item) for item in obj]
                return obj
            
            export_data = convert_datetime(export_data)
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Данные сессии экспортированы: {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка экспорта данных сессии {session_id}: {e}")
            return False
    
    def create_backup(self) -> str:
        """Создание резервной копии базы данных"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"rl_database_backup_{timestamp}.db"
            backup_path = os.path.join(self.backup_dir, backup_filename)
            
            # Копируем базу данных
            shutil.copy2(self.db_path, backup_path)
            
            # Сжимаем резервную копию
            compressed_path = f"{backup_path}.gz"
            with open(backup_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Удаляем несжатую копию
            os.remove(backup_path)
            
            logger.info(f"✅ Создана резервная копия: {compressed_path}")
            return compressed_path
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания резервной копии: {e}")
            return ""
    
    def cleanup_old_backups(self, max_backups: int = 30):
        """Очистка старых резервных копий"""
        try:
            if not os.path.exists(self.backup_dir):
                return
            
            # Получаем список файлов резервных копий
            backup_files = []
            for filename in os.listdir(self.backup_dir):
                if filename.startswith('rl_database_backup_') and filename.endswith('.db.gz'):
                    filepath = os.path.join(self.backup_dir, filename)
                    backup_files.append((filepath, os.path.getctime(filepath)))
            
            # Сортируем по времени создания (новые первыми)
            backup_files.sort(key=lambda x: x[1], reverse=True)
            
            # Удаляем старые копии
            deleted_count = 0
            for filepath, _ in backup_files[max_backups:]:
                try:
                    os.remove(filepath)
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"⚠️ Не удалось удалить резервную копию {filepath}: {e}")
            
            if deleted_count > 0:
                logger.info(f"✅ Удалено старых резервных копий: {deleted_count}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка очистки резервных копий: {e}")
    
    def get_database_info(self) -> Dict[str, Any]:
        """Получение информации о базе данных"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Размер базы данных
                db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                
                # Количество записей в таблицах
                cursor.execute("SELECT COUNT(*) FROM sessions")
                sessions_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM trades")
                trades_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM weight_evolution")
                evolution_count = cursor.fetchone()[0]
                
                # Последняя активность
                cursor.execute("SELECT MAX(created_at) FROM sessions")
                last_session = cursor.fetchone()[0]
                
                return {
                    'database_path': self.db_path,
                    'database_size_bytes': db_size,
                    'database_size_mb': round(db_size / (1024 * 1024), 2),
                    'sessions_count': sessions_count,
                    'trades_count': trades_count,
                    'weight_evolution_count': evolution_count,
                    'last_session_created': last_session,
                    'backup_directory': self.backup_dir
                }
                
        except Exception as e:
            logger.error(f"❌ Ошибка получения информации о базе данных: {e}")
            return {}

# Пример использования
def main():
    """Пример использования базы данных"""
    db = ReinforcementLearningDatabase()
    
    # Получаем информацию о базе данных
    info = db.get_database_info()
    print(f"Информация о базе данных: {info}")
    
    # Получаем последние сессии
    recent_sessions = db.get_recent_sessions(5)
    print(f"Последние сессии: {len(recent_sessions)}")
    
    # Получаем статистику
    stats = db.get_performance_statistics(30)
    print(f"Статистика за 30 дней: {stats}")

if __name__ == "__main__":
    main()