# Руководство пользователя системы диагностики AI моделей

## 1. Введение

Система диагностики AI моделей предназначена для комплексного анализа стабильности и производительности торговых AI моделей в системе WinRT и PPL. Система позволяет:

- 🔍 **Изолированное тестирование** каждой AI модели отдельно
- 🔗 **Комбинированное тестирование** взаимодействия моделей
- 📊 **Анализ производительности** и потребления ресурсов
- 📈 **Генерацию отчетов** с графиками и статистикой
- 🎯 **Получение рекомендаций** по оптимизации системы

## 2. Быстрый старт

### 2.1 Установка и настройка

```bash
# Переход в директорию проекта
cd "Peper Binance v4 Clean"

# Установка дополнительных зависимостей для диагностики
pip install psutil plotly reportlab openpyxl rich

# Проверка готовности системы
python -c "from ai_modules.lava_ai import LavaAI; print('✅ LavaAI готов')"
python -c "from ai_modules.mistral_ai import MistralAI; print('✅ MistralAI готов')"
```

### 2.2 Первый запуск

```bash
# Базовая диагностика всех моделей (30 минут)
python ai_diagnostic_system/main.py

# Быстрая диагностика одной модели (5 минут)
python ai_diagnostic_system/main.py --models lava --test-type isolated --config quick_test.json
```

## 3. Типы диагностики

### 3.1 Изолированное тестирование

Тестирует каждую AI модель отдельно для выявления индивидуальных проблем:

```bash
# Тестирование всех моделей изолированно
python ai_diagnostic_system/main.py --test-type isolated

# Тестирование конкретных моделей
python ai_diagnostic_system/main.py --test-type isolated --models lava,mistral

# Тестирование с расширенными метриками
python ai_diagnostic_system/main.py --test-type isolated --config detailed_config.json
```

**Что измеряется:**
- ⏱️ Время отклика модели
- 🎯 Точность предсказаний
- 💾 Потребление памяти
- 🖥️ Загрузка CPU/GPU
- 📊 Стабильность работы

### 3.2 Комбинированное тестирование

Анализирует взаимодействие моделей и выявляет конфликты:

```bash
# Тестирование всех комбинаций моделей
python ai_diagnostic_system/main.py --test-type combined

# Тестирование конкретной комбинации
python ai_diagnostic_system/main.py --test-type combined --models lava,mistral,trading

# Нагрузочное тестирование
python ai_diagnostic_system/main.py --test-type combined --config stress_test.json
```

**Что анализируется:**
- 🔗 Синергия между моделями
- ⚡ Конфликты в решениях
- 📈 Влияние одной модели на другие
- 🎭 Поведение под нагрузкой

### 3.3 Полная диагностика

Комплексный анализ всей системы:

```bash
# Полная диагностика (рекомендуется)
python ai_diagnostic_system/main.py --test-type full

# Полная диагностика с кастомными настройками
python ai_diagnostic_system/main.py --test-type full --config production_config.json --output-dir ./reports/$(date +%Y%m%d)
```

## 4. Конфигурация

### 4.1 Базовая конфигурация (basic_config.json)

```json
{
  "test_duration_minutes": 15,
  "test_pairs": ["BTCUSDT", "ETHUSDT"],
  "parallel_tests": true,
  "max_parallel_models": 2,
  
  "isolated_test_enabled": true,
  "combined_test_enabled": true,
  
  "resource_monitoring_interval": 10,
  "performance_sampling_rate": 5,
  
  "generate_charts": true,
  "export_formats": ["html"],
  "detailed_logging": false,
  
  "accuracy_threshold": 0.55,
  "performance_threshold": 2000,
  "memory_threshold": 256,
  "cpu_threshold": 70
}
```

### 4.2 Продакшн конфигурация (production_config.json)

```json
{
  "test_duration_minutes": 60,
  "test_pairs": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"],
  "parallel_tests": true,
  "max_parallel_models": 3,
  
  "isolated_test_enabled": true,
  "combined_test_enabled": true,
  "interaction_analysis": true,
  "conflict_detection": true,
  
  "resource_monitoring_interval": 5,
  "performance_sampling_rate": 10,
  
  "generate_charts": true,
  "export_formats": ["html", "pdf", "csv"],
  "detailed_logging": true,
  
  "accuracy_threshold": 0.6,
  "performance_threshold": 1000,
  "memory_threshold": 512,
  "cpu_threshold": 80
}
```

### 4.3 Быстрый тест (quick_test.json)

```json
{
  "test_duration_minutes": 5,
  "test_pairs": ["BTCUSDT"],
  "parallel_tests": false,
  
  "isolated_test_enabled": true,
  "combined_test_enabled": false,
  
  "resource_monitoring_interval": 30,
  "performance_sampling_rate": 2,
  
  "generate_charts": false,
  "export_formats": ["html"],
  "detailed_logging": false
}
```

## 5. Интерпретация результатов

### 5.1 Метрики производительности

#### Время отклика (Response Time)
- ✅ **< 500ms** - Отличная производительность
- ⚠️ **500-1000ms** - Приемлемая производительность
- ❌ **> 1000ms** - Требует оптимизации

#### Точность (Accuracy)
- ✅ **> 65%** - Высокая точность
- ⚠️ **55-65%** - Средняя точность
- ❌ **< 55%** - Низкая точность, требует переобучения

#### Потребление памяти (Memory Usage)
- ✅ **< 256MB** - Оптимальное потребление
- ⚠️ **256-512MB** - Умеренное потребление
- ❌ **> 512MB** - Высокое потребление, возможны утечки

### 5.2 Анализ взаимодействий

#### Синергия моделей (Synergy Score)
- ✅ **> 0.7** - Высокая синергия, модели дополняют друг друга
- ⚠️ **0.4-0.7** - Умеренная синергия
- ❌ **< 0.4** - Низкая синергия, возможны конфликты

#### Конфликты решений (Decision Conflicts)
- ✅ **< 10%** - Минимальные конфликты
- ⚠️ **10-25%** - Умеренные конфликты
- ❌ **> 25%** - Высокий уровень конфликтов

## 6. Типичные проблемы и решения

### 6.1 Низкая производительность модели

**Симптомы:**
- Время отклика > 1000ms
- Высокое потребление CPU
- Низкая точность

**Решения:**
```bash
# Диагностика конкретной модели
python ai_diagnostic_system/main.py --test-type isolated --models [model_name]

# Анализ с детальным логированием
python ai_diagnostic_system/main.py --config detailed_config.json --models [model_name]
```

**Рекомендации:**
- Проверить качество входных данных
- Рассмотреть переобучение модели
- Оптимизировать параметры модели

### 6.2 Конфликты между моделями

**Симптомы:**
- Высокий процент конфликтов решений
- Низкая синергия
- Нестабильные результаты

**Решения:**
```bash
# Анализ взаимодействий
python ai_diagnostic_system/main.py --test-type combined --models [model1,model2]

# Детальный анализ конфликтов
python ai_diagnostic_system/main.py --config conflict_analysis.json
```

**Рекомендации:**
- Пересмотреть веса моделей в ансамбле
- Изменить логику принятия решений
- Рассмотреть исключение конфликтующих моделей

### 6.3 Утечки памяти

**Симптомы:**
- Постоянный рост потребления памяти
- Замедление работы со временем
- Ошибки OutOfMemory

**Решения:**
```bash
# Мониторинг ресурсов
python ai_diagnostic_system/main.py --config memory_monitoring.json

# Длительный тест для выявления утечек
python ai_diagnostic_system/main.py --config long_term_test.json
```

## 7. Автоматизация диагностики

### 7.1 Ежедневная диагностика

Создайте скрипт для автоматической ежедневной диагностики:

```bash
#!/bin/bash
# daily_diagnostic.sh

DATE=$(date +%Y%m%d)
OUTPUT_DIR="./diagnostic_reports/$DATE"

echo "🚀 Запуск ежедневной диагностики AI моделей..."

# Быстрая проверка всех моделей
python ai_diagnostic_system/main.py \
  --test-type isolated \
  --config quick_test.json \
  --output-dir "$OUTPUT_DIR/quick"

# Полная диагностика критических моделей
python ai_diagnostic_system/main.py \
  --test-type full \
  --models lava,mistral,trading \
  --config production_config.json \
  --output-dir "$OUTPUT_DIR/full"

echo "✅ Диагностика завершена. Отчеты сохранены в $OUTPUT_DIR"
```

### 7.2 Мониторинг в реальном времени

```bash
# Непрерывный мониторинг производительности
python ai_diagnostic_system/main.py \
  --test-type isolated \
  --config continuous_monitoring.json \
  --output-dir ./live_monitoring
```

## 8. Интеграция с существующей системой

### 8.1 Использование с winrate_test_with_results2.py

```python
# Пример интеграции в существующий код
from ai_diagnostic_system.core.diagnostic_controller import DiagnosticController
from ai_diagnostic_system.config.diagnostic_config import DiagnosticConfig

# В вашем существующем коде
async def run_enhanced_testing():
    # Обычное тестирование винрейта
    winrate_results = await run_winrate_test()
    
    # Диагностика AI моделей
    diagnostic_config = DiagnosticConfig(test_duration_minutes=15)
    diagnostic_controller = DiagnosticController(diagnostic_config)
    diagnostic_results = await diagnostic_controller.run_full_diagnostic(['lava', 'mistral'])
    
    # Объединение результатов
    combined_report = merge_reports(winrate_results, diagnostic_results)
    return combined_report
```

### 8.2 Расширение существующих тестов

```python
# Добавление диагностики в RealWinrateTester
class EnhancedWinrateTester(RealWinrateTester):
    def __init__(self, config: TestConfig):
        super().__init__(config)
        self.diagnostic_controller = DiagnosticController()
    
    async def test_symbol_with_diagnostics(self, symbol: str):
        # Обычное тестирование
        winrate_result = await self.test_symbol(symbol)
        
        # Диагностика моделей
        diagnostic_result = await self.diagnostic_controller.run_isolated_tests(['lava', 'mistral'])
        
        return {
            'winrate': winrate_result,
            'diagnostics': diagnostic_result
        }
```

## 9. Часто задаваемые вопросы

### Q: Как часто нужно запускать диагностику?
**A:** Рекомендуется:
- Ежедневно: быстрая диагностика (5-15 минут)
- Еженедельно: полная диагностика (30-60 минут)
- После изменений: целевая диагностика измененных компонентов

### Q: Какие модели критически важны для диагностики?
**A:** Приоритет по важности:
1. **MistralAI** - принятие финальных решений
2. **LavaAI** - технический анализ
3. **TradingAI** - анализ рыночных условий
4. **LGBMAI** - управление рисками
5. **ReinforcementLearning** - адаптация весов

### Q: Что делать, если диагностика показывает критические проблемы?
**A:** Следуйте плану действий:
1. Остановите торговлю на проблемных парах
2. Запустите детальную диагностику проблемной модели
3. Проанализируйте логи и отчеты
4. Примените рекомендации системы
5. Повторите диагностику для подтверждения исправления

### Q: Можно ли запускать диагностику во время торговли?
**A:** Да, но с ограничениями:
- Используйте `--config low_impact.json` для минимального влияния
- Избегайте нагрузочного тестирования в торговые часы
- Мониторьте потребление ресурсов

## 10. Поддержка и устранение неполадок

### 10.1 Логи и отладка

```bash
# Включение детального логирования
python ai_diagnostic_system/main.py --config debug_config.json

# Просмотр логов в реальном времени
tail -f ./diagnostic_reports/latest/diagnostic.log
```

### 10.2 Проверка системы

```bash
# Проверка готовности всех компонентов
python ai_diagnostic_system/utils/system_check.py

# Тест подключения к AI модулям
python ai_diagnostic_system/tests/test_integration.py
```

### 10.3 Восстановление после ошибок

```bash
# Очистка кэшей диагностики
python ai_diagnostic_system/utils/clear_cache.py

# Сброс конфигурации к значениям по умолчанию
python ai_diagnostic_system/utils/reset_config.py
```