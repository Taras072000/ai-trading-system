#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работы локальной модели Mistral
"""

import asyncio
import sys
import os
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.append(str(Path(__file__).parent))

from ai_modules.mistral_ai import MistralAI

async def test_mistral_local():
    """Тестирование локальной модели Mistral"""
    print("🔧 Тестирование локальной модели Mistral...")
    
    try:
        # Инициализация Mistral AI
        mistral = MistralAI()
        print("✅ MistralAI инициализирован")
        
        # Проверка информации о модели
        model_info = await mistral.get_model_info()
        print(f"📊 Информация о модели:")
        for key, value in model_info.items():
            print(f"   {key}: {value}")
        
        # Проверка существования файла модели
        if not model_info['model_exists']:
            print("❌ Файл модели не найден!")
            return False
        
        print("✅ Файл модели найден")
        
        # Инициализация модели
        await mistral.initialize()
        print("✅ Модель инициализирована")
        
        # Тестовый запрос
        test_prompt = "Привет! Как дела?"
        print(f"🤖 Тестовый запрос: {test_prompt}")
        
        response = await mistral.generate_text(test_prompt, max_tokens=50)
        print(f"📝 Ответ модели: {response.text}")
        print(f"⏱️ Время обработки: {response.processing_time:.2f}s")
        print(f"🎯 Уверенность: {response.confidence:.2f}")
        print(f"🔢 Токенов использовано: {response.tokens_used}")
        
        # Тестирование торгового анализа
        print("\n📈 Тестирование торгового анализа...")
        trading_analysis = await mistral.analyze_trading_opportunity(
            symbol="BTCUSDT",
            current_price=45000.0,
            price_data=[
                {"price": 44000, "timestamp": "2024-01-01"},
                {"price": 45000, "timestamp": "2024-01-02"}
            ]
        )
        print(f"📊 Торговый анализ: {trading_analysis}")
        
        # Очистка
        await mistral.cleanup()
        print("✅ Очистка завершена")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_mistral_local())
    if success:
        print("\n🎉 Тестирование завершено успешно!")
    else:
        print("\n💥 Тестирование завершилось с ошибками!")
        sys.exit(1)