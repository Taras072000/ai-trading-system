"""
Утилиты для работы с часовыми поясами и временем в UTC
"""
from datetime import datetime, timezone, timedelta
import pytz
import pandas as pd
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)

# Константы для часовых поясов
UTC = timezone.utc
MOSCOW_TZ = pytz.timezone('Europe/Moscow')  # UTC+3

def get_utc_now() -> datetime:
    """
    Получить текущее время в UTC
    
    Returns:
        datetime: Текущее время в UTC
    """
    return datetime.now(UTC)

def get_local_now() -> datetime:
    """
    Получить текущее время в локальном часовом поясе (Москва UTC+3)
    
    Returns:
        datetime: Текущее время в московском часовом поясе
    """
    return datetime.now(MOSCOW_TZ)

def convert_to_utc(dt: Union[datetime, str], source_tz: Optional[str] = None) -> datetime:
    """
    Конвертировать время в UTC
    
    Args:
        dt: Время для конвертации (datetime или строка)
        source_tz: Исходный часовой пояс (по умолчанию 'Europe/Moscow')
        
    Returns:
        datetime: Время в UTC
    """
    if source_tz is None:
        source_tz = 'Europe/Moscow'
    
    if isinstance(dt, str):
        dt = pd.to_datetime(dt)
    
    if dt.tzinfo is None:
        # Если время наивное, считаем его в исходном часовом поясе
        source_timezone = pytz.timezone(source_tz)
        dt = source_timezone.localize(dt)
    
    return dt.astimezone(UTC)

def convert_from_utc(dt: datetime, target_tz: str = 'Europe/Moscow') -> datetime:
    """
    Конвертировать время из UTC в целевой часовой пояс
    
    Args:
        dt: Время в UTC
        target_tz: Целевой часовой пояс
        
    Returns:
        datetime: Время в целевом часовом поясе
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    
    target_timezone = pytz.timezone(target_tz)
    return dt.astimezone(target_timezone)

def ensure_utc_timestamp(timestamp: Union[datetime, str, int, float]) -> datetime:
    """
    Убедиться что timestamp в UTC
    
    Args:
        timestamp: Временная метка в различных форматах
        
    Returns:
        datetime: Время в UTC
    """
    if isinstance(timestamp, (int, float)):
        # Unix timestamp
        return datetime.fromtimestamp(timestamp, tz=UTC)
    
    if isinstance(timestamp, str):
        # Строковое представление
        dt = pd.to_datetime(timestamp)
        if dt.tzinfo is None:
            # Если нет информации о часовом поясе, считаем UTC
            return dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    
    if isinstance(timestamp, datetime):
        if timestamp.tzinfo is None:
            # Наивное время считаем UTC
            return timestamp.replace(tzinfo=UTC)
        return timestamp.astimezone(UTC)
    
    raise ValueError(f"Неподдерживаемый тип timestamp: {type(timestamp)}")

def get_binance_time_offset() -> timedelta:
    """
    Получить смещение времени для Binance API (всегда UTC)
    
    Returns:
        timedelta: Смещение (всегда 0 для UTC)
    """
    return timedelta(0)

def format_utc_time(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S UTC") -> str:
    """
    Форматировать время в UTC с указанием часового пояса
    
    Args:
        dt: Время для форматирования
        format_str: Формат строки
        
    Returns:
        str: Отформатированная строка времени
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    
    return dt.strftime(format_str)

def is_market_hours_utc(dt: Optional[datetime] = None) -> bool:
    """
    Проверить, является ли время рыночными часами (в UTC)
    Основные рынки работают примерно с 00:00 до 22:00 UTC
    
    Args:
        dt: Время для проверки (по умолчанию текущее время UTC)
        
    Returns:
        bool: True если рыночные часы
    """
    if dt is None:
        dt = get_utc_now()
    
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    
    hour = dt.hour
    # Основная торговая активность с 00:00 до 22:00 UTC
    return 0 <= hour <= 22

def get_moscow_time_from_utc(utc_dt: datetime) -> datetime:
    """
    Конвертировать UTC время в московское время
    
    Args:
        utc_dt: Время в UTC
        
    Returns:
        datetime: Время в московском часовом поясе
    """
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=UTC)
    
    return utc_dt.astimezone(MOSCOW_TZ)

def log_timezone_info():
    """
    Логировать информацию о текущих часовых поясах
    """
    utc_now = get_utc_now()
    moscow_now = get_local_now()
    
    logger.info(f"Текущее время UTC: {format_utc_time(utc_now)}")
    logger.info(f"Текущее время Москва: {moscow_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    logger.info(f"Смещение Москва-UTC: {moscow_now.utcoffset()}")

# Функции для обратной совместимости
def datetime_now_utc() -> datetime:
    """Алиас для get_utc_now()"""
    return get_utc_now()

def datetime_now_local() -> datetime:
    """Алиас для get_local_now()"""
    return get_local_now()