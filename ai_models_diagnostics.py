"""
🤖 ДИАГНОСТИКА AI МОДЕЛЕЙ
Специализированная система для проверки доступности и работоспособности AI моделей

Автор: AI Trading System
Дата: 2024
"""

import asyncio
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIModelsDiagnostics:
    """
    🤖 ДИАГНОСТИКА AI МОДЕЛЕЙ
    
    Проверяет:
    1. Наличие API ключей
    2. Инициализацию моделей
    3. Доступность сервисов
    4. Тестовые запросы
    """
    
    def __init__(self):
        self.results = {}
        logger.info("🤖 Инициализирована диагностика AI моделей")
    
    async def run_full_ai_diagnostics(self) -> Dict[str, Any]:
        """
        🚀 ПОЛНАЯ ДИАГНОСТИКА AI МОДЕЛЕЙ
        
        Этапы:
        1. Проверка переменных окружения
        2. Проверка конфигурации
        3. Тестирование инициализации
        4. Тестовые запросы
        """
        logger.info("🚀 Запуск полной диагностики AI моделей...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'environment_check': {},
            'config_check': {},
            'initialization_check': {},
            'api_test': {},
            'summary': {},
            'recommendations': []
        }
        
        try:
            # Этап 1: Проверка переменных окружения
            logger.info("🔑 Этап 1: Проверка переменных окружения...")
            results['environment_check'] = await self._check_environment_variables()
            
            # Этап 2: Проверка конфигурации
            logger.info("⚙️ Этап 2: Проверка конфигурации...")
            results['config_check'] = await self._check_configuration_files()
            
            # Этап 3: Тестирование инициализации
            logger.info("🔧 Этап 3: Тестирование инициализации...")
            results['initialization_check'] = await self._test_model_initialization()
            
            # Этап 4: Тестовые запросы
            logger.info("📡 Этап 4: Тестовые запросы...")
            results['api_test'] = await self._test_api_requests()
            
            # Формируем сводку и рекомендации
            results['summary'] = self._generate_summary(results)
            results['recommendations'] = self._generate_recommendations(results)
            
            # Создаем отчет
            await self._create_ai_diagnostics_report(results)
            
            logger.info("✅ Диагностика AI моделей завершена!")
            return results
            
        except Exception as e:
            logger.error(f"❌ Ошибка при диагностике AI моделей: {e}")
            raise
    
    async def _check_environment_variables(self) -> Dict[str, Any]:
        """Проверка переменных окружения для AI моделей"""
        logger.info("🔑 Проверка переменных окружения...")
        
        required_env_vars = {
            'OPENAI_API_KEY': 'OpenAI (для trading_ai)',
            'ANTHROPIC_API_KEY': 'Anthropic Claude (для claude_ai)',
            'GOOGLE_API_KEY': 'Google Gemini (для gemini_ai)',
            'LAVA_API_KEY': 'Lava AI (для lava_ai)',
            'LAVA_API_URL': 'Lava AI URL (для lava_ai)'
        }
        
        env_status = {}
        
        for var_name, description in required_env_vars.items():
            value = os.getenv(var_name)
            if value:
                # Проверяем, что это не placeholder
                if value.startswith('sk-') or len(value) > 20:
                    env_status[var_name] = {
                        'status': 'OK',
                        'description': description,
                        'length': len(value),
                        'preview': value[:10] + '...' if len(value) > 10 else value
                    }
                    logger.info(f"✅ {var_name}: OK ({len(value)} символов)")
                else:
                    env_status[var_name] = {
                        'status': 'PLACEHOLDER',
                        'description': description,
                        'value': value,
                        'issue': 'Похоже на placeholder'
                    }
                    logger.warning(f"⚠️ {var_name}: Placeholder значение")
            else:
                env_status[var_name] = {
                    'status': 'MISSING',
                    'description': description,
                    'issue': 'Переменная не установлена'
                }
                logger.error(f"❌ {var_name}: Отсутствует")
        
        return env_status
    
    async def _check_configuration_files(self) -> Dict[str, Any]:
        """Проверка конфигурационных файлов"""
        logger.info("⚙️ Проверка конфигурационных файлов...")
        
        config_files = [
            '.env',
            'config.json',
            'ai_config.json'
        ]
        
        config_status = {}
        
        for config_file in config_files:
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        if config_file.endswith('.json'):
                            content = json.load(f)
                            config_status[config_file] = {
                                'status': 'OK',
                                'size': os.path.getsize(config_file),
                                'keys': list(content.keys()) if isinstance(content, dict) else 'Not a dict'
                            }
                        else:
                            content = f.read()
                            config_status[config_file] = {
                                'status': 'OK',
                                'size': os.path.getsize(config_file),
                                'lines': len(content.splitlines())
                            }
                    logger.info(f"✅ {config_file}: OK")
                except Exception as e:
                    config_status[config_file] = {
                        'status': 'ERROR',
                        'error': str(e)
                    }
                    logger.error(f"❌ {config_file}: Ошибка чтения - {e}")
            else:
                config_status[config_file] = {
                    'status': 'MISSING'
                }
                logger.warning(f"⚠️ {config_file}: Файл не найден")
        
        return config_status
    
    async def _test_model_initialization(self) -> Dict[str, Any]:
        """Тестирование инициализации моделей"""
        logger.info("🔧 Тестирование инициализации моделей...")
        
        models_to_test = [
            'trading_ai',
            'lava_ai', 
            'gemini_ai',
            'claude_ai'
        ]
        
        init_results = {}
        
        for model_name in models_to_test:
            try:
                logger.info(f"🔧 Тестирование {model_name}...")
                
                # Пытаемся импортировать и инициализировать модель
                if model_name == 'trading_ai':
                    result = await self._test_openai_initialization()
                elif model_name == 'lava_ai':
                    result = await self._test_lava_initialization()
                elif model_name == 'gemini_ai':
                    result = await self._test_gemini_initialization()
                elif model_name == 'claude_ai':
                    result = await self._test_claude_initialization()
                else:
                    result = {'status': 'UNKNOWN', 'error': 'Неизвестная модель'}
                
                init_results[model_name] = result
                
                if result['status'] == 'OK':
                    logger.info(f"✅ {model_name}: Инициализация успешна")
                else:
                    logger.error(f"❌ {model_name}: {result.get('error', 'Неизвестная ошибка')}")
                    
            except Exception as e:
                init_results[model_name] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
                logger.error(f"❌ {model_name}: Ошибка инициализации - {e}")
        
        return init_results
    
    async def _test_openai_initialization(self) -> Dict[str, Any]:
        """Тестирование инициализации OpenAI"""
        try:
            import openai
            
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                return {'status': 'ERROR', 'error': 'OPENAI_API_KEY не установлен'}
            
            # Создаем клиент
            client = openai.OpenAI(api_key=api_key)
            
            return {
                'status': 'OK',
                'client_type': str(type(client)),
                'api_key_length': len(api_key)
            }
            
        except ImportError:
            return {'status': 'ERROR', 'error': 'Модуль openai не установлен'}
        except Exception as e:
            return {'status': 'ERROR', 'error': str(e)}
    
    async def _test_lava_initialization(self) -> Dict[str, Any]:
        """Тестирование инициализации Lava AI"""
        try:
            import aiohttp
            
            api_key = os.getenv('LAVA_API_KEY')
            api_url = os.getenv('LAVA_API_URL')
            
            if not api_key:
                return {'status': 'ERROR', 'error': 'LAVA_API_KEY не установлен'}
            
            if not api_url:
                return {'status': 'ERROR', 'error': 'LAVA_API_URL не установлен'}
            
            return {
                'status': 'OK',
                'api_key_length': len(api_key),
                'api_url': api_url
            }
            
        except ImportError:
            return {'status': 'ERROR', 'error': 'Модуль aiohttp не установлен'}
        except Exception as e:
            return {'status': 'ERROR', 'error': str(e)}
    
    async def _test_gemini_initialization(self) -> Dict[str, Any]:
        """Тестирование инициализации Google Gemini"""
        try:
            import google.generativeai as genai
            
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                return {'status': 'ERROR', 'error': 'GOOGLE_API_KEY не установлен'}
            
            # Конфигурируем API
            genai.configure(api_key=api_key)
            
            return {
                'status': 'OK',
                'api_key_length': len(api_key)
            }
            
        except ImportError:
            return {'status': 'ERROR', 'error': 'Модуль google-generativeai не установлен'}
        except Exception as e:
            return {'status': 'ERROR', 'error': str(e)}
    
    async def _test_claude_initialization(self) -> Dict[str, Any]:
        """Тестирование инициализации Anthropic Claude"""
        try:
            import anthropic
            
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                return {'status': 'ERROR', 'error': 'ANTHROPIC_API_KEY не установлен'}
            
            # Создаем клиент
            client = anthropic.Anthropic(api_key=api_key)
            
            return {
                'status': 'OK',
                'client_type': str(type(client)),
                'api_key_length': len(api_key)
            }
            
        except ImportError:
            return {'status': 'ERROR', 'error': 'Модуль anthropic не установлен'}
        except Exception as e:
            return {'status': 'ERROR', 'error': str(e)}
    
    async def _test_api_requests(self) -> Dict[str, Any]:
        """Тестирование API запросов"""
        logger.info("📡 Тестирование API запросов...")
        
        api_results = {}
        
        # Простой тестовый промпт
        test_prompt = "Ответь одним словом: работает ли API?"
        
        # Тестируем каждую модель
        models = ['trading_ai', 'lava_ai', 'gemini_ai', 'claude_ai']
        
        for model_name in models:
            try:
                logger.info(f"📡 Тестирование API {model_name}...")
                
                if model_name == 'trading_ai':
                    result = await self._test_openai_api(test_prompt)
                elif model_name == 'lava_ai':
                    result = await self._test_lava_api(test_prompt)
                elif model_name == 'gemini_ai':
                    result = await self._test_gemini_api(test_prompt)
                elif model_name == 'claude_ai':
                    result = await self._test_claude_api(test_prompt)
                else:
                    result = {'status': 'SKIPPED', 'reason': 'Неизвестная модель'}
                
                api_results[model_name] = result
                
                if result['status'] == 'OK':
                    logger.info(f"✅ {model_name}: API работает")
                else:
                    logger.error(f"❌ {model_name}: {result.get('error', 'API не работает')}")
                    
            except Exception as e:
                api_results[model_name] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
                logger.error(f"❌ {model_name}: Ошибка API - {e}")
        
        return api_results
    
    async def _test_openai_api(self, prompt: str) -> Dict[str, Any]:
        """Тестирование OpenAI API"""
        try:
            import openai
            
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                return {'status': 'ERROR', 'error': 'API ключ отсутствует'}
            
            client = openai.OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10
            )
            
            return {
                'status': 'OK',
                'response': response.choices[0].message.content,
                'model': response.model
            }
            
        except Exception as e:
            return {'status': 'ERROR', 'error': str(e)}
    
    async def _test_lava_api(self, prompt: str) -> Dict[str, Any]:
        """Тестирование Lava AI API"""
        try:
            import aiohttp
            
            api_key = os.getenv('LAVA_API_KEY')
            api_url = os.getenv('LAVA_API_URL')
            
            if not api_key or not api_url:
                return {'status': 'ERROR', 'error': 'API ключ или URL отсутствует'}
            
            async with aiohttp.ClientSession() as session:
                headers = {'Authorization': f'Bearer {api_key}'}
                data = {'prompt': prompt, 'max_tokens': 10}
                
                async with session.post(api_url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            'status': 'OK',
                            'response': result.get('response', 'Нет ответа'),
                            'status_code': response.status
                        }
                    else:
                        return {
                            'status': 'ERROR',
                            'error': f'HTTP {response.status}',
                            'status_code': response.status
                        }
                        
        except Exception as e:
            return {'status': 'ERROR', 'error': str(e)}
    
    async def _test_gemini_api(self, prompt: str) -> Dict[str, Any]:
        """Тестирование Google Gemini API"""
        try:
            import google.generativeai as genai
            
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                return {'status': 'ERROR', 'error': 'API ключ отсутствует'}
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            
            response = model.generate_content(prompt)
            
            return {
                'status': 'OK',
                'response': response.text,
                'model': 'gemini-pro'
            }
            
        except Exception as e:
            return {'status': 'ERROR', 'error': str(e)}
    
    async def _test_claude_api(self, prompt: str) -> Dict[str, Any]:
        """Тестирование Anthropic Claude API"""
        try:
            import anthropic
            
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                return {'status': 'ERROR', 'error': 'API ключ отсутствует'}
            
            client = anthropic.Anthropic(api_key=api_key)
            
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return {
                'status': 'OK',
                'response': response.content[0].text,
                'model': response.model
            }
            
        except Exception as e:
            return {'status': 'ERROR', 'error': str(e)}
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация сводки результатов"""
        summary = {
            'total_models': 4,
            'env_vars_ok': 0,
            'models_initialized': 0,
            'apis_working': 0,
            'critical_issues': [],
            'warnings': []
        }
        
        # Подсчитываем переменные окружения
        for var, status in results['environment_check'].items():
            if status['status'] == 'OK':
                summary['env_vars_ok'] += 1
            elif status['status'] == 'MISSING':
                summary['critical_issues'].append(f"Отсутствует {var}")
            elif status['status'] == 'PLACEHOLDER':
                summary['warnings'].append(f"Placeholder в {var}")
        
        # Подсчитываем инициализацию
        for model, status in results['initialization_check'].items():
            if status['status'] == 'OK':
                summary['models_initialized'] += 1
            else:
                summary['critical_issues'].append(f"Не инициализируется {model}")
        
        # Подсчитываем API
        for model, status in results['api_test'].items():
            if status['status'] == 'OK':
                summary['apis_working'] += 1
            else:
                summary['critical_issues'].append(f"API не работает {model}")
        
        return summary
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций"""
        recommendations = []
        
        # Проверяем переменные окружения
        for var, status in results['environment_check'].items():
            if status['status'] == 'MISSING':
                recommendations.append(f"Установить переменную окружения {var}")
            elif status['status'] == 'PLACEHOLDER':
                recommendations.append(f"Заменить placeholder в {var} на реальный API ключ")
        
        # Проверяем инициализацию
        for model, status in results['initialization_check'].items():
            if status['status'] == 'ERROR' and 'не установлен' in status.get('error', ''):
                recommendations.append(f"Установить зависимости для {model}")
        
        # Общие рекомендации
        if not recommendations:
            recommendations.append("Все AI модели настроены корректно")
        else:
            recommendations.append("После исправления проблем перезапустить систему")
        
        return recommendations
    
    async def _create_ai_diagnostics_report(self, results: Dict[str, Any]):
        """Создание отчета диагностики AI моделей"""
        report_lines = [
            "=" * 80,
            "🤖 ДЕТАЛЬНЫЙ ОТЧЕТ: ДИАГНОСТИКА AI МОДЕЛЕЙ",
            "=" * 80,
            f"📅 Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "📊 СВОДКА:",
            "-" * 50,
            f"   Всего моделей: {results['summary']['total_models']}",
            f"   Переменных окружения OK: {results['summary']['env_vars_ok']}/5",
            f"   Моделей инициализировано: {results['summary']['models_initialized']}/4",
            f"   API работают: {results['summary']['apis_working']}/4",
            "",
            "🔑 ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ:",
            "-" * 50
        ]
        
        for var, status in results['environment_check'].items():
            status_icon = "✅" if status['status'] == 'OK' else "⚠️" if status['status'] == 'PLACEHOLDER' else "❌"
            report_lines.append(f"   {status_icon} {var}: {status['status']}")
            if 'issue' in status:
                report_lines.append(f"      Проблема: {status['issue']}")
        
        report_lines.extend([
            "",
            "🔧 ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ:",
            "-" * 50
        ])
        
        for model, status in results['initialization_check'].items():
            status_icon = "✅" if status['status'] == 'OK' else "❌"
            report_lines.append(f"   {status_icon} {model}: {status['status']}")
            if 'error' in status:
                report_lines.append(f"      Ошибка: {status['error']}")
        
        report_lines.extend([
            "",
            "📡 ТЕСТИРОВАНИЕ API:",
            "-" * 50
        ])
        
        for model, status in results['api_test'].items():
            status_icon = "✅" if status['status'] == 'OK' else "❌"
            report_lines.append(f"   {status_icon} {model}: {status['status']}")
            if status['status'] == 'OK' and 'response' in status:
                report_lines.append(f"      Ответ: {status['response']}")
            elif 'error' in status:
                report_lines.append(f"      Ошибка: {status['error']}")
        
        report_lines.extend([
            "",
            "🚨 КРИТИЧЕСКИЕ ПРОБЛЕМЫ:",
            "-" * 50
        ])
        
        for issue in results['summary']['critical_issues']:
            report_lines.append(f"   • {issue}")
        
        if not results['summary']['critical_issues']:
            report_lines.append("   Критических проблем не обнаружено")
        
        report_lines.extend([
            "",
            "💡 РЕКОМЕНДАЦИИ:",
            "-" * 50
        ])
        
        for rec in results['recommendations']:
            report_lines.append(f"   • {rec}")
        
        report_lines.extend([
            "",
            "=" * 80,
            "✅ ДИАГНОСТИКА AI МОДЕЛЕЙ ЗАВЕРШЕНА",
            "=" * 80
        ])
        
        report_content = "\n".join(report_lines)
        
        # Сохраняем отчет
        report_filename = f"ai_models_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"📋 Создан отчет диагностики AI моделей: {report_filename}")


async def main():
    """Основная функция для запуска диагностики AI моделей"""
    print("🤖 Запуск диагностики AI моделей...")
    
    # Создаем систему диагностики
    diagnostics = AIModelsDiagnostics()
    
    # Запускаем анализ
    try:
        results = await diagnostics.run_full_ai_diagnostics()
        
        print("\n" + "="*60)
        print("✅ ДИАГНОСТИКА AI МОДЕЛЕЙ ЗАВЕРШЕНА!")
        print("="*60)
        print(f"🔑 Переменных окружения OK: {results['summary']['env_vars_ok']}/5")
        print(f"🔧 Моделей инициализировано: {results['summary']['models_initialized']}/4")
        print(f"📡 API работают: {results['summary']['apis_working']}/4")
        print("="*60)
        
        if results['summary']['critical_issues']:
            print("\n🚨 КРИТИЧЕСКИЕ ПРОБЛЕМЫ:")
            for issue in results['summary']['critical_issues']:
                print(f"   • {issue}")
        
        if results['summary']['warnings']:
            print("\n⚠️ ПРЕДУПРЕЖДЕНИЯ:")
            for warning in results['summary']['warnings']:
                print(f"   • {warning}")
        
        if results['recommendations']:
            print("\n💡 РЕКОМЕНДАЦИИ:")
            for rec in results['recommendations']:
                print(f"   • {rec}")
        
        print(f"\n📋 Детальный отчет создан: ai_models_diagnostics_*.txt")
        
    except Exception as e:
        print(f"❌ Ошибка при диагностике: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())