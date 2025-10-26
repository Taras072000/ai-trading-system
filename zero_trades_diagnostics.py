"""
🔍 ДИАГНОСТИКА ПРОБЛЕМЫ ОТСУТСТВИЯ СДЕЛОК
Специализированная система для выявления причин, почему система не генерирует сделки

Автор: AI Trading System
Дата: 2024
"""

import asyncio
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Импортируем существующие компоненты
from winrate_test_with_results2 import RealWinrateTester, TestConfig

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ZeroTradesAnalysis:
    """Результаты анализа отсутствия сделок"""
    timestamp: str
    
    # Анализ AI сигналов
    ai_signals_generated: Dict[str, int]
    ai_signals_quality: Dict[str, float]
    ai_models_active: List[str]
    
    # Анализ фильтров
    filters_blocking: Dict[str, int]
    consensus_failures: int
    confidence_failures: int
    
    # Анализ данных
    data_availability: Dict[str, bool]
    market_conditions: Dict[str, Any]
    
    # Рекомендации
    immediate_fixes: List[str]
    parameter_adjustments: List[str]
    
    # Диагноз
    root_cause: str
    severity: str

class ZeroTradesDiagnostics:
    """
    🔍 СПЕЦИАЛИЗИРОВАННАЯ ДИАГНОСТИКА ОТСУТСТВИЯ СДЕЛОК
    
    Анализирует каждый этап процесса принятия торговых решений:
    1. Загрузка данных
    2. Генерация AI сигналов
    3. Применение фильтров
    4. Формирование консенсуса
    5. Принятие торгового решения
    """
    
    def __init__(self):
        self.tester = None
        self.analysis_results = {}
        
        logger.info("🔍 Инициализирована диагностика отсутствия сделок")
    
    async def run_zero_trades_analysis(self, symbols: List[str] = None) -> ZeroTradesAnalysis:
        """
        🚀 ЗАПУСК АНАЛИЗА ОТСУТСТВИЯ СДЕЛОК
        
        Этапы:
        1. Проверка доступности данных
        2. Анализ генерации AI сигналов
        3. Анализ работы фильтров
        4. Анализ консенсуса
        5. Выявление корневой причины
        """
        logger.info("🚀 Запуск анализа отсутствия сделок...")
        
        if symbols is None:
            symbols = ['BTCUSDT', 'ETHUSDT']
        
        try:
            # Создаем тестер с минимальными фильтрами
            config = TestConfig(
                symbols=symbols,
                test_period_days=1,  # Короткий период для быстрой диагностики
                position_size_percent=0.02,
                min_confidence=0.01,  # Очень низкий порог
                min_consensus_models=1,  # Минимальный консенсус
                enabled_ai_models=['trading_ai', 'lava_ai'],
                use_strict_filters=False,  # Отключаем строгие фильтры
                min_volatility=0.0,  # Отключаем фильтр волатильности
                min_volume_ratio=0.0,  # Отключаем фильтр объема
                use_time_filter=False,  # Отключаем временной фильтр
                require_volume_confirmation=False  # Отключаем подтверждение объемом
            )
            
            self.tester = RealWinrateTester(config)
            await self.tester.initialize()
            
            # Этап 1: Проверка доступности данных
            logger.info("📊 Этап 1: Проверка доступности данных...")
            data_analysis = await self._analyze_data_availability(symbols)
            
            # Этап 2: Анализ генерации AI сигналов
            logger.info("🤖 Этап 2: Анализ генерации AI сигналов...")
            ai_analysis = await self._analyze_ai_signal_generation(symbols)
            
            # Этап 3: Анализ работы фильтров
            logger.info("🔍 Этап 3: Анализ работы фильтров...")
            filter_analysis = await self._analyze_filter_blocking(symbols)
            
            # Этап 4: Анализ консенсуса
            logger.info("🤝 Этап 4: Анализ консенсуса...")
            consensus_analysis = await self._analyze_consensus_failures(symbols)
            
            # Этап 5: Выявление корневой причины
            logger.info("🎯 Этап 5: Выявление корневой причины...")
            root_cause_analysis = await self._identify_root_cause(
                data_analysis, ai_analysis, filter_analysis, consensus_analysis
            )
            
            # Формируем итоговый анализ
            analysis = ZeroTradesAnalysis(
                timestamp=datetime.now().isoformat(),
                ai_signals_generated=ai_analysis['signals_generated'],
                ai_signals_quality=ai_analysis['signals_quality'],
                ai_models_active=ai_analysis['models_active'],
                filters_blocking=filter_analysis['blocking_filters'],
                consensus_failures=consensus_analysis['failures'],
                confidence_failures=filter_analysis['confidence_failures'],
                data_availability=data_analysis['availability'],
                market_conditions=data_analysis['market_conditions'],
                immediate_fixes=root_cause_analysis['immediate_fixes'],
                parameter_adjustments=root_cause_analysis['parameter_adjustments'],
                root_cause=root_cause_analysis['root_cause'],
                severity=root_cause_analysis['severity']
            )
            
            # Создаем детальный отчет
            await self._create_zero_trades_report(analysis)
            
            logger.info("✅ Анализ отсутствия сделок завершен!")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Ошибка при анализе: {e}")
            raise
    
    async def _analyze_data_availability(self, symbols: List[str]) -> Dict[str, Any]:
        """Анализ доступности данных"""
        logger.info("📊 Анализ доступности данных...")
        
        data_availability = {}
        market_conditions = {}
        
        for symbol in symbols:
            try:
                # Загружаем данные
                data = await self.tester.load_historical_data(symbol)
                
                if data is not None and len(data) > 0:
                    data_availability[symbol] = True
                    
                    # Анализируем рыночные условия
                    latest_price = data['close'].iloc[-1]
                    price_change_24h = ((latest_price - data['close'].iloc[-24]) / data['close'].iloc[-24]) * 100 if len(data) >= 24 else 0
                    volume_avg = data['volume'].tail(24).mean() if len(data) >= 24 else 0
                    volatility = data['close'].pct_change().std() * 100
                    
                    market_conditions[symbol] = {
                        'latest_price': latest_price,
                        'price_change_24h': price_change_24h,
                        'volume_avg': volume_avg,
                        'volatility': volatility,
                        'data_points': len(data)
                    }
                    
                    logger.info(f"📊 {symbol}: {len(data)} точек данных, волатильность {volatility:.2f}%")
                else:
                    data_availability[symbol] = False
                    market_conditions[symbol] = {'error': 'Нет данных'}
                    logger.warning(f"⚠️ {symbol}: Данные недоступны")
                    
            except Exception as e:
                data_availability[symbol] = False
                market_conditions[symbol] = {'error': str(e)}
                logger.error(f"❌ {symbol}: Ошибка загрузки данных - {e}")
        
        return {
            'availability': data_availability,
            'market_conditions': market_conditions
        }
    
    async def _analyze_ai_signal_generation(self, symbols: List[str]) -> Dict[str, Any]:
        """Анализ генерации AI сигналов"""
        logger.info("🤖 Анализ генерации AI сигналов...")
        
        signals_generated = {}
        signals_quality = {}
        models_active = []
        
        # Проверяем каждую AI модель
        ai_models = ['trading_ai', 'lava_ai', 'gemini_ai', 'claude_ai']
        
        for model_name in ai_models:
            signals_generated[model_name] = 0
            signals_quality[model_name] = 0.0
            
            try:
                # Проверяем доступность модели
                if hasattr(self.tester, 'ai_orchestrator') and self.tester.ai_orchestrator:
                    model_available = await self._check_model_availability(model_name)
                    if model_available:
                        models_active.append(model_name)
                        
                        # Тестируем генерацию сигналов
                        for symbol in symbols:
                            data = await self.tester.load_historical_data(symbol)
                            if data is not None and len(data) > 0:
                                try:
                                    signal = await self.tester.get_individual_ai_signal(model_name, symbol, data)
                                    if signal:
                                        signals_generated[model_name] += 1
                                        signals_quality[model_name] = max(signals_quality[model_name], signal.confidence)
                                        logger.info(f"🤖 {model_name} -> {symbol}: {signal.action} (confidence: {signal.confidence:.2f})")
                                    else:
                                        logger.warning(f"⚠️ {model_name} -> {symbol}: Нет сигнала")
                                except Exception as e:
                                    logger.error(f"❌ {model_name} -> {symbol}: Ошибка генерации сигнала - {e}")
                    else:
                        logger.warning(f"⚠️ {model_name}: Модель недоступна")
                        
            except Exception as e:
                logger.error(f"❌ {model_name}: Ошибка проверки модели - {e}")
        
        logger.info(f"🤖 Активных моделей: {len(models_active)}, Сигналов сгенерировано: {sum(signals_generated.values())}")
        
        return {
            'signals_generated': signals_generated,
            'signals_quality': signals_quality,
            'models_active': models_active
        }
    
    async def _check_model_availability(self, model_name: str) -> bool:
        """Проверка доступности AI модели"""
        try:
            # Простая проверка доступности модели
            if model_name == 'trading_ai':
                return hasattr(self.tester, 'trading_ai') and self.tester.trading_ai is not None
            elif model_name == 'lava_ai':
                return hasattr(self.tester, 'lava_ai') and self.tester.lava_ai is not None
            elif model_name == 'gemini_ai':
                return hasattr(self.tester, 'gemini_ai') and self.tester.gemini_ai is not None
            elif model_name == 'claude_ai':
                return hasattr(self.tester, 'claude_ai') and self.tester.claude_ai is not None
            else:
                return False
        except:
            return False
    
    async def _analyze_filter_blocking(self, symbols: List[str]) -> Dict[str, Any]:
        """Анализ блокировки фильтрами"""
        logger.info("🔍 Анализ работы фильтров...")
        
        blocking_filters = {
            'confidence_filter': 0,
            'volatility_filter': 0,
            'volume_filter': 0,
            'time_filter': 0,
            'technical_filter': 0
        }
        
        confidence_failures = 0
        
        # Симулируем проверку фильтров
        for symbol in symbols:
            data = await self.tester.load_historical_data(symbol)
            if data is not None and len(data) > 0:
                
                # Проверяем фильтр уверенности
                test_confidence = 0.15  # Тестовая уверенность
                if test_confidence < self.tester.config.min_confidence:
                    blocking_filters['confidence_filter'] += 1
                    confidence_failures += 1
                
                # Проверяем фильтр волатильности
                volatility = data['close'].pct_change().std()
                if volatility < self.tester.config.min_volatility:
                    blocking_filters['volatility_filter'] += 1
                
                # Проверяем фильтр объема
                volume_ratio = self.tester._calculate_volume_ratio(data)
                if volume_ratio < self.tester.config.min_volume_ratio:
                    blocking_filters['volume_filter'] += 1
                
                # Проверяем временной фильтр
                if self.tester.config.use_time_filter:
                    current_hour = datetime.now().hour
                    if not self.tester.is_trading_hour_allowed(datetime.now()):
                        blocking_filters['time_filter'] += 1
        
        logger.info(f"🔍 Блокировок фильтрами: {sum(blocking_filters.values())}")
        
        return {
            'blocking_filters': blocking_filters,
            'confidence_failures': confidence_failures
        }
    
    async def _analyze_consensus_failures(self, symbols: List[str]) -> Dict[str, Any]:
        """Анализ неудач консенсуса"""
        logger.info("🤝 Анализ консенсуса...")
        
        consensus_failures = 0
        
        # Симулируем проверку консенсуса
        for symbol in symbols:
            # Создаем тестовые решения моделей
            test_decisions = [
                {'model': 'trading_ai', 'action': 'BUY', 'confidence': 0.3},
                {'model': 'lava_ai', 'action': 'HOLD', 'confidence': 0.2}
            ]
            
            # Проверяем достаточность для консенсуса
            buy_votes = sum(1 for d in test_decisions if d['action'] == 'BUY')
            sell_votes = sum(1 for d in test_decisions if d['action'] == 'SELL')
            
            if max(buy_votes, sell_votes) < self.tester.config.min_consensus_models:
                consensus_failures += 1
        
        logger.info(f"🤝 Неудач консенсуса: {consensus_failures}")
        
        return {
            'failures': consensus_failures
        }
    
    async def _identify_root_cause(self, data_analysis: Dict, ai_analysis: Dict, 
                                 filter_analysis: Dict, consensus_analysis: Dict) -> Dict[str, Any]:
        """Выявление корневой причины"""
        logger.info("🎯 Выявление корневой причины...")
        
        immediate_fixes = []
        parameter_adjustments = []
        root_cause = "Неизвестная причина"
        severity = "MEDIUM"
        
        # Анализируем данные
        if not all(data_analysis['availability'].values()):
            root_cause = "Отсутствие исторических данных"
            severity = "CRITICAL"
            immediate_fixes.append("Проверить подключение к источнику данных")
            immediate_fixes.append("Убедиться в корректности API ключей")
        
        # Анализируем AI модели
        elif len(ai_analysis['models_active']) == 0:
            root_cause = "Ни одна AI модель не активна"
            severity = "CRITICAL"
            immediate_fixes.append("Проверить инициализацию AI моделей")
            immediate_fixes.append("Убедиться в доступности API ключей для AI")
        
        elif sum(ai_analysis['signals_generated'].values()) == 0:
            root_cause = "AI модели не генерируют сигналы"
            severity = "HIGH"
            immediate_fixes.append("Проверить логику генерации сигналов в AI моделях")
            parameter_adjustments.append("Снизить пороги уверенности AI моделей")
        
        # Анализируем фильтры
        elif filter_analysis['confidence_failures'] > 0:
            root_cause = "Слишком высокий порог уверенности"
            severity = "MEDIUM"
            parameter_adjustments.append(f"Снизить min_confidence с {self.tester.config.min_confidence} до 0.15")
        
        elif sum(filter_analysis['blocking_filters'].values()) > 0:
            root_cause = "Фильтры блокируют все сигналы"
            severity = "MEDIUM"
            parameter_adjustments.append("Ослабить фильтры волатильности и объема")
            parameter_adjustments.append("Отключить временные фильтры")
        
        # Анализируем консенсус
        elif consensus_analysis['failures'] > 0:
            root_cause = "Недостаточно моделей для консенсуса"
            severity = "MEDIUM"
            parameter_adjustments.append(f"Снизить min_consensus_models с {self.tester.config.min_consensus_models} до 1")
        
        logger.info(f"🎯 Корневая причина: {root_cause} (серьезность: {severity})")
        
        return {
            'root_cause': root_cause,
            'severity': severity,
            'immediate_fixes': immediate_fixes,
            'parameter_adjustments': parameter_adjustments
        }
    
    async def _create_zero_trades_report(self, analysis: ZeroTradesAnalysis):
        """Создание детального отчета по отсутствию сделок"""
        report_lines = [
            "=" * 80,
            "🔍 ДЕТАЛЬНЫЙ ОТЧЕТ: ДИАГНОСТИКА ОТСУТСТВИЯ СДЕЛОК",
            "=" * 80,
            f"📅 Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "🎯 КОРНЕВАЯ ПРИЧИНА:",
            "-" * 50,
            f"   {analysis.root_cause}",
            f"   Серьезность: {analysis.severity}",
            "",
            "🤖 АНАЛИЗ AI МОДЕЛЕЙ:",
            "-" * 50,
            f"   Активных моделей: {len(analysis.ai_models_active)}",
            f"   Модели: {', '.join(analysis.ai_models_active)}",
            ""
        ]
        
        # Добавляем статистику по сигналам
        report_lines.append("   Сигналы по моделям:")
        for model, count in analysis.ai_signals_generated.items():
            quality = analysis.ai_signals_quality.get(model, 0)
            report_lines.append(f"     • {model}: {count} сигналов (качество: {quality:.2f})")
        
        report_lines.extend([
            "",
            "🔍 АНАЛИЗ ФИЛЬТРОВ:",
            "-" * 50,
            f"   Блокировок фильтром уверенности: {analysis.confidence_failures}",
            f"   Всего блокировок фильтрами: {sum(analysis.filters_blocking.values())}",
            ""
        ])
        
        # Добавляем детали по фильтрам
        report_lines.append("   Блокировки по типам:")
        for filter_name, count in analysis.filters_blocking.items():
            if count > 0:
                report_lines.append(f"     • {filter_name}: {count}")
        
        report_lines.extend([
            "",
            "🤝 АНАЛИЗ КОНСЕНСУСА:",
            "-" * 50,
            f"   Неудач консенсуса: {analysis.consensus_failures}",
            "",
            "📊 РЫНОЧНЫЕ УСЛОВИЯ:",
            "-" * 50
        ])
        
        # Добавляем рыночные условия
        for symbol, conditions in analysis.market_conditions.items():
            if 'error' not in conditions:
                report_lines.append(f"   {symbol}:")
                report_lines.append(f"     • Цена: {conditions.get('latest_price', 0):.2f}")
                report_lines.append(f"     • Изменение 24ч: {conditions.get('price_change_24h', 0):.2f}%")
                report_lines.append(f"     • Волатильность: {conditions.get('volatility', 0):.2f}%")
                report_lines.append(f"     • Точек данных: {conditions.get('data_points', 0)}")
            else:
                report_lines.append(f"   {symbol}: ОШИБКА - {conditions['error']}")
        
        report_lines.extend([
            "",
            "🚨 НЕМЕДЛЕННЫЕ ИСПРАВЛЕНИЯ:",
            "-" * 50
        ])
        
        for fix in analysis.immediate_fixes:
            report_lines.append(f"   • {fix}")
        
        if not analysis.immediate_fixes:
            report_lines.append("   Немедленных исправлений не требуется")
        
        report_lines.extend([
            "",
            "🎛️ РЕКОМЕНДУЕМЫЕ НАСТРОЙКИ:",
            "-" * 50
        ])
        
        for adjustment in analysis.parameter_adjustments:
            report_lines.append(f"   • {adjustment}")
        
        if not analysis.parameter_adjustments:
            report_lines.append("   Настройки параметров не требуются")
        
        report_lines.extend([
            "",
            "=" * 80,
            "✅ ДИАГНОСТИКА ЗАВЕРШЕНА",
            "=" * 80
        ])
        
        report_content = "\n".join(report_lines)
        
        # Сохраняем отчет
        report_filename = f"zero_trades_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"📋 Создан отчет диагностики: {report_filename}")


async def main():
    """Основная функция для запуска диагностики отсутствия сделок"""
    print("🔍 Запуск диагностики отсутствия сделок...")
    
    # Создаем систему диагностики
    diagnostics = ZeroTradesDiagnostics()
    
    # Запускаем анализ
    try:
        analysis = await diagnostics.run_zero_trades_analysis(['BTCUSDT', 'ETHUSDT'])
        
        print("\n" + "="*60)
        print("✅ ДИАГНОСТИКА ОТСУТСТВИЯ СДЕЛОК ЗАВЕРШЕНА!")
        print("="*60)
        print(f"🎯 Корневая причина: {analysis.root_cause}")
        print(f"⚠️ Серьезность: {analysis.severity}")
        print(f"🤖 Активных AI моделей: {len(analysis.ai_models_active)}")
        print(f"📊 Сигналов сгенерировано: {sum(analysis.ai_signals_generated.values())}")
        print(f"🔍 Блокировок фильтрами: {sum(analysis.filters_blocking.values())}")
        print(f"🤝 Неудач консенсуса: {analysis.consensus_failures}")
        print("="*60)
        
        if analysis.immediate_fixes:
            print("\n🚨 НЕМЕДЛЕННЫЕ ИСПРАВЛЕНИЯ:")
            for fix in analysis.immediate_fixes:
                print(f"   • {fix}")
        
        if analysis.parameter_adjustments:
            print("\n🎛️ РЕКОМЕНДУЕМЫЕ НАСТРОЙКИ:")
            for adjustment in analysis.parameter_adjustments:
                print(f"   • {adjustment}")
        
        print(f"\n📋 Детальный отчет создан: zero_trades_analysis_*.txt")
        
    except Exception as e:
        print(f"❌ Ошибка при диагностике: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())