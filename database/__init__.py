#!/usr/bin/env python3
"""
Database Module for Reinforcement Learning System
Модуль базы данных для системы обучения с подкреплением
"""

from .reinforcement_learning_db import (
    ReinforcementLearningDatabase,
    SessionRecord,
    TradeRecord,
    WeightEvolutionRecord
)

from .persistence_manager import (
    PersistenceManager,
    PersistenceConfig
)

__all__ = [
    'ReinforcementLearningDatabase',
    'SessionRecord',
    'TradeRecord',
    'WeightEvolutionRecord',
    'PersistenceManager',
    'PersistenceConfig'
]

__version__ = "1.0.0"
__author__ = "Peper Binance v4 Team"
__description__ = "Database and persistence system for reinforcement learning"