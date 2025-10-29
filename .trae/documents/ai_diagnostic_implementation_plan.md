# План реализации системы диагностики AI моделей

## 1. Структура проекта

```
ai_diagnostic_system/
├── main.py                          # Главный файл запуска
├── config/
│   ├── diagnostic_config.py         # Конфигурация диагностики
│   └── model_configs.py             # Конфигурации для каждой модели
├── core/
│   ├── diagnostic_controller.py     # Основной контроллер
│   ├── test_orchestrator.py         # Оркестратор тестов
│   └── session_manager.py           # Управление сессиями
├── testers/
│   ├── isolated_tester.py           # Изолированное тестирование
│   ├── combined_tester.py           # Комбинированное тестирование
│   ├── model_wrappers/
│   │   ├── lava_ai_wrapper.py       # Обертка для LavaAI
│   │   ├── mistral_ai_wrapper.py    # Обертка для MistralAI
│   │   ├── trading_ai_wrapper.py    # Обертка для TradingAI
│   │   ├── lgbm_ai_wrapper.py       # Обертка для LGBMAI
│   │   └── rl_engine_wrapper.py     # Обертка для ReinforcementLearning
│   └── interaction_analyzer.py      # Анализатор взаимодействий
├── monitoring/
│   ├── performance_monitor.py       # Мониторинг производительности
│   ├── resource_tracker.py          # Отслеживание ресурсов
│   └── metrics_collector.py         # Сбор метрик
├── analysis/
│   ├── statistical_analyzer.py     # Статистический анализ
│   ├── performance_analyzer.py     # Анализ производительности
│   └── bottleneck_detector.py      # Детектор узких мест
├── reporting/
│   ├── report_generator.py         # Генератор отчетов
│   ├── chart_generator.py          # Генератор графиков
│   └── export_manager.py           # Менеджер экспорта
├── recommendations/
│   ├── recommendation_engine.py    # Движок рекомендаций
│   ├── decision_maker.py           # Система принятия решений
│   └── action_planner.py           # Планировщик действий
├── utils/
│   ├── data_loader.py              # Загрузчик данных
│   ├── logger.py                   # Система логирования
│   └── helpers.py                  # Вспомогательные функции
└── tests/
    ├── test_isolated.py            # Тесты изолированного тестирования
    ├── test_combined.py            # Тесты комбинированного тестирования
    └── test_integration.py         # Интеграционные тесты
```

## 2. Основные компоненты

### 2.1 Диагностический контроллер (diagnostic_controller.py)

```python
class DiagnosticController:
    """Основной контроллер системы диагностики"""
    
    def __init__(self, config: DiagnosticConfig):
        self.config = config
        self.session_manager = SessionManager()
        self.test_orchestrator = TestOrchestrator()
        self.performance_monitor = PerformanceMonitor()
        
    async def run_full_diagnostic(self, models: List[str]) -> DiagnosticResult:
        """Запуск полной диагностики"""
        
    async def run_isolated_tests(self, models: List[str]) -> Dict[str, ModelTestResult]:
        """Запуск изолированных тестов"""
        
    async def run_combined_tests(self, model_combinations: List[List[str]]) -> CombinedTestResult:
        """Запуск комбинированных тестов"""
```

### 2.2 Изолированный тестер (isolated_tester.py)

```python
class IsolatedTester:
    """Тестер для изолированного тестирования AI моделей"""
    
    def __init__(self):
        self.model_wrappers = {
            'lava': LavaAIWrapper(),
            'mistral': MistralAIWrapper(),
            'trading': TradingAIWrapper(),
            'lgbm': LGBMAIWrapper(),
            'reinforcement': RLEngineWrapper()
        }
        
    async def test_model(self, model_name: str, test_config: TestConfig) -> ModelTestResult:
        """Тестирование отдельной модели"""
        
    async def measure_performance(self, model_name: str, test_data: pd.DataFrame) -> PerformanceMetrics:
        """Измерение производительности модели"""
        
    async def analyze_accuracy(self, model_name: str, predictions: List, actual: List) -> AccuracyMetrics:
        """Анализ точности модели"""
```

### 2.3 Комбинированный тестер (combined_tester.py)

```python
class CombinedTester:
    """Тестер для комбинированного тестирования моделей"""
    
    def __init__(self):
        self.interaction_analyzer = InteractionAnalyzer()
        self.conflict_detector = ConflictDetector()
        
    async def test_model_combination(self, models: List[str], test_config: TestConfig) -> CombinationResult:
        """Тестирование комбинации моделей"""
        
    async def analyze_interactions(self, models: List[str], results: Dict) -> InteractionAnalysis:
        """Анализ взаимодействий между моделями"""
        
    async def detect_conflicts(self, model_decisions: Dict) -> List[Conflict]:
        """Выявление конфликтов между моделями"""
```

### 2.4 Монитор производительности (performance_monitor.py)

```python
class PerformanceMonitor:
    """Мониторинг производительности и ресурсов"""
    
    def __init__(self):
        self.resource_tracker = ResourceTracker()
        self.metrics_collector = MetricsCollector()
        
    async def start_monitoring(self, test_id: str):
        """Начало мониторинга"""
        
    async def stop_monitoring(self, test_id: str) -> ResourceUsageReport:
        """Остановка мониторинга и получение отчета"""
        
    async def collect_metrics(self, model_name: str, operation: str) -> OperationMetrics:
        """Сбор метрик операции"""
```

## 3. Интеграция с существующей системой

### 3.1 Интеграция с winrate_test_with_results2.py

```python
class WinrateIntegration:
    """Интеграция с существующей системой тестирования винрейта"""
    
    def __init__(self, winrate_tester: RealWinrateTester):
        self.winrate_tester = winrate_tester
        
    async def run_diagnostic_with_winrate(self, config: DiagnosticConfig) -> IntegratedResult:
        """Запуск диагностики с использованием винрейт тестера"""
        
    async def extract_model_performance(self) -> Dict[str, ModelPerformance]:
        """Извлечение данных о производительности моделей"""
        
    async def compare_with_baseline(self, diagnostic_results: DiagnosticResult) -> ComparisonReport:
        """Сравнение с базовыми показателями"""
```

### 3.2 Использование существующих AI модулей

```python
# Пример интеграции с LavaAI
class LavaAIWrapper:
    """Обертка для интеграции с LavaAI"""
    
    def __init__(self):
        from ai_modules.lava_ai import LavaAI
        self.lava_ai = LavaAI()
        
    async def run_isolated_test(self, test_data: pd.DataFrame) -> TestResult:
        """Изолированный тест LavaAI"""
        start_time = time.time()
        
        # Тестирование анализа паттернов
        pattern_results = await self.lava_ai.analyze_patterns(test_data)
        
        # Тестирование технического анализа
        technical_results = await self.lava_ai.technical_analysis(test_data)
        
        execution_time = time.time() - start_time
        
        return TestResult(
            model_name='lava',
            execution_time=execution_time,
            pattern_accuracy=self._calculate_pattern_accuracy(pattern_results),
            technical_accuracy=self._calculate_technical_accuracy(technical_results),
            memory_usage=self._get_memory_usage(),
            cpu_usage=self._get_cpu_usage()
        )
```

## 4. Система рекомендаций

### 4.1 Движок рекомендаций (recommendation_engine.py)

```python
class RecommendationEngine:
    """Движок для генерации рекомендаций на основе результатов диагностики"""
    
    def __init__(self):
        self.decision_maker = DecisionMaker()
        self.action_planner = ActionPlanner()
        
    async def generate_recommendations(self, diagnostic_results: DiagnosticResult) -> List[Recommendation]:
        """Генерация рекомендаций"""
        
    async def analyze_retraining_needs(self, model_performance: Dict) -> RetrainingAnalysis:
        """Анализ потребности в переобучении"""
        
    async def suggest_architecture_changes(self, interaction_results: InteractionAnalysis) -> List[ArchitectureChange]:
        """Предложения по изменению архитектуры"""
```

### 4.2 Система принятия решений (decision_maker.py)

```python
class DecisionMaker:
    """Система принятия решений на основе аналитических данных"""
    
    def __init__(self):
        self.thresholds = DecisionThresholds()
        
    def should_retrain_model(self, model_name: str, performance_metrics: PerformanceMetrics) -> bool:
        """Определение необходимости переобучения модели"""
        
    def prioritize_actions(self, recommendations: List[Recommendation]) -> List[PrioritizedAction]:
        """Приоритизация рекомендуемых действий"""
        
    def assess_system_health(self, diagnostic_results: DiagnosticResult) -> SystemHealthStatus:
        """Оценка общего состояния системы"""
```

## 5. Отчетность и визуализация

### 5.1 Генератор отчетов (report_generator.py)

```python
class ReportGenerator:
    """Генератор детализированных отчетов"""
    
    def __init__(self):
        self.chart_generator = ChartGenerator()
        self.export_manager = ExportManager()
        
    async def generate_full_report(self, diagnostic_results: DiagnosticResult) -> Report:
        """Генерация полного отчета"""
        
    async def generate_model_report(self, model_name: str, test_results: ModelTestResult) -> ModelReport:
        """Генерация отчета по отдельной модели"""
        
    async def generate_comparison_report(self, results: List[DiagnosticResult]) -> ComparisonReport:
        """Генерация сравнительного отчета"""
```

### 5.2 Генератор графиков (chart_generator.py)

```python
class ChartGenerator:
    """Генератор графиков и визуализаций"""
    
    def __init__(self):
        self.style_config = ChartStyleConfig()
        
    def create_performance_chart(self, performance_data: Dict) -> Chart:
        """Создание графика производительности"""
        
    def create_resource_usage_chart(self, resource_data: List[ResourceUsage]) -> Chart:
        """Создание графика использования ресурсов"""
        
    def create_interaction_heatmap(self, interaction_matrix: np.ndarray) -> Chart:
        """Создание тепловой карты взаимодействий"""
```

## 6. Конфигурация и настройки

### 6.1 Конфигурация диагностики (diagnostic_config.py)

```python
@dataclass
class DiagnosticConfig:
    """Конфигурация системы диагностики"""
    
    # Общие настройки
    test_duration_minutes: int = 30
    test_pairs: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    parallel_tests: bool = True
    max_parallel_models: int = 3
    
    # Настройки изолированного тестирования
    isolated_test_enabled: bool = True
    measure_accuracy: bool = True
    measure_performance: bool = True
    measure_resources: bool = True
    
    # Настройки комбинированного тестирования
    combined_test_enabled: bool = True
    test_all_combinations: bool = True
    interaction_analysis: bool = True
    conflict_detection: bool = True
    
    # Настройки мониторинга
    resource_monitoring_interval: int = 5  # секунды
    performance_sampling_rate: int = 10  # измерений в секунду
    
    # Настройки отчетности
    generate_charts: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["html", "pdf"])
    detailed_logging: bool = True
    
    # Пороги для рекомендаций
    accuracy_threshold: float = 0.6
    performance_threshold: float = 1000  # мс
    memory_threshold: float = 512  # MB
    cpu_threshold: float = 80  # %
```

## 7. Запуск и использование

### 7.1 Главный файл (main.py)

```python
#!/usr/bin/env python3
"""
Система диагностики AI моделей торговой системы
Запуск: python main.py --config config.json --models lava,mistral,trading
"""

import asyncio
import argparse
from pathlib import Path
from core.diagnostic_controller import DiagnosticController
from config.diagnostic_config import DiagnosticConfig

async def main():
    parser = argparse.ArgumentParser(description='AI Models Diagnostic System')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--models', type=str, help='Comma-separated list of models to test')
    parser.add_argument('--test-type', choices=['isolated', 'combined', 'full'], default='full')
    parser.add_argument('--output-dir', type=str, default='./diagnostic_reports')
    
    args = parser.parse_args()
    
    # Загрузка конфигурации
    config = DiagnosticConfig.load_from_file(args.config) if args.config else DiagnosticConfig()
    
    # Инициализация контроллера
    controller = DiagnosticController(config)
    
    # Определение моделей для тестирования
    models = args.models.split(',') if args.models else ['lava', 'mistral', 'trading', 'lgbm']
    
    # Запуск диагностики
    print(f"🚀 Запуск диагностики AI моделей: {', '.join(models)}")
    
    if args.test_type == 'isolated':
        results = await controller.run_isolated_tests(models)
    elif args.test_type == 'combined':
        results = await controller.run_combined_tests([models])
    else:
        results = await controller.run_full_diagnostic(models)
    
    # Генерация отчета
    report_path = await controller.generate_report(results, args.output_dir)
    print(f"📊 Отчет сохранен: {report_path}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 7.2 Примеры использования

```bash
# Полная диагностика всех моделей
python main.py --test-type full

# Изолированное тестирование конкретных моделей
python main.py --test-type isolated --models lava,mistral

# Комбинированное тестирование с кастомной конфигурацией
python main.py --test-type combined --config custom_config.json --output-dir ./reports

# Быстрая диагностика с минимальными настройками
python main.py --models trading --test-type isolated
```

## 8. Интеграция с существующей системой

Диагностическая система будет интегрирована с существующим `winrate_test_with_results2.py` через:

1. **Импорт существующих AI модулей** - использование уже созданных классов LavaAI, MistralAI, TradingAI, LGBMAI
2. **Расширение TestConfig** - добавление параметров диагностики в существующую конфигурацию
3. **Интеграция с RealWinrateTester** - использование существующей логики тестирования как базы
4. **Совместное использование данных** - использование тех же исторических данных и кэшей
5. **Единая система логирования** - интеграция с существующей системой логов

Это обеспечит совместимость и позволит использовать диагностическую систему как дополнение к существующему функционалу.