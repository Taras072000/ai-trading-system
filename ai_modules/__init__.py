"""
AI Modules Package для Peper Binance v4
Оптимизированные AI модули с минимальным потреблением ресурсов
"""

__version__ = "4.0.0"
__author__ = "Peper Binance Team"

# Импорт главного менеджера
from .ai_manager import (
    ai_manager,
    AIManager,
    AIModuleType,
    AIModuleStatus,
    AIResponse,
    get_trading_analysis,
    get_lava_analysis,
    get_mistral_response,
    get_lgbm_prediction
)

# Ленивые импорты для экономии памяти
def get_trading_ai():
    """Ленивый импорт Trading AI"""
    from .trading_ai import TradingAI
    return TradingAI()

def get_lava_ai():
    """Ленивый импорт Lava AI"""
    from .lava_ai import LavaAI
    return LavaAI()

def get_mistral_ai():
    """Ленивый импорт Mistral AI"""
    from .mistral_ai import MistralAI
    return MistralAI()

def get_lgbm_ai():
    """Ленивый импорт LGBM AI"""
    from .lgbm_ai import LGBMAI
    return LGBMAI()

# Экспортируемые модули
__all__ = [
    # Главный менеджер
    'ai_manager',
    'AIManager',
    'AIModuleType',
    'AIModuleStatus', 
    'AIResponse',
    
    # Быстрые функции доступа
    'get_trading_analysis',
    'get_lava_analysis',
    'get_mistral_response',
    'get_lgbm_prediction',
    
    # Ленивые импорты
    'get_trading_ai',
    'get_lava_ai', 
    'get_mistral_ai',
    'get_lgbm_ai'
]