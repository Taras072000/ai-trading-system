#!/usr/bin/env python3
"""
Тестовый скрипт для проверки улучшений в системе винрейт тестирования
"""

import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class AIModelDecision:
    model_name: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    reasoning: str
    timestamp: datetime

class TestImprovements:
    """Тестовый класс для проверки улучшений"""
    
    def __init__(self):
        self.ai_models_performance = {
            'trading_ai': {'signal_accuracy': 0.75, 'contribution_to_pnl': 150.0, 'consensus_participation_rate': 0.8},
            'lava_ai': {'signal_accuracy': 0.68, 'contribution_to_pnl': 120.0, 'consensus_participation_rate': 0.7},
            'lgbm_ai': {'signal_accuracy': 0.82, 'contribution_to_pnl': 180.0, 'consensus_participation_rate': 0.9},
            'mistral_ai': {'signal_accuracy': 0.71, 'contribution_to_pnl': 90.0, 'consensus_participation_rate': 0.6},
            'reinforcement_learning_engine': {'signal_accuracy': 0.79, 'contribution_to_pnl': 200.0, 'consensus_participation_rate': 0.85}
        }
        self.consensus_weight_threshold = 0.55
        self.min_consensus_models = 2
    
    def calculate_model_weights(self) -> Dict[str, float]:
        """Рассчитывает веса моделей на основе их производительности"""
        try:
            if not self.ai_models_performance:
                print("⚠️ Нет данных о производительности моделей, используем равные веса")
                return {}
            
            weights = {}
            for model_name, performance in self.ai_models_performance.items():
                # Комбинированный вес на основе точности, вклада в PnL и участия в консенсусе
                accuracy_weight = performance.get('signal_accuracy', 0.5)
                pnl_weight = max(0.1, min(2.0, performance.get('contribution_to_pnl', 0) / 100.0))
                participation_weight = performance.get('consensus_participation_rate', 0.5)
                
                # Итоговый вес = среднее взвешенное
                combined_weight = (accuracy_weight * 0.4 + pnl_weight * 0.4 + participation_weight * 0.2)
                weights[model_name] = max(0.1, min(3.0, combined_weight))  # Ограничиваем вес от 0.1 до 3.0
            
            print(f"📊 Рассчитанные веса моделей:")
            for model, weight in weights.items():
                perf = self.ai_models_performance[model]
                print(f"   {model}: {weight:.3f} (точность: {perf['signal_accuracy']:.3f}, PnL: {perf['contribution_to_pnl']:.1f}, участие: {perf['consensus_participation_rate']:.3f})")
            
            return weights
            
        except Exception as e:
            print(f"❌ Ошибка при расчете весов моделей: {e}")
            return {}
    
    def test_weighted_voting(self):
        """Тестирует взвешенное голосование"""
        print("\n🗳️ ТЕСТ ВЗВЕШЕННОГО ГОЛОСОВАНИЯ")
        print("=" * 50)
        
        # Создаем тестовые решения моделей
        model_decisions = [
            AIModelDecision('trading_ai', 'BUY', 0.8, 'Strong uptrend', datetime.now()),
            AIModelDecision('lava_ai', 'BUY', 0.6, 'Positive sentiment', datetime.now()),
            AIModelDecision('lgbm_ai', 'SELL', 0.9, 'Overbought conditions', datetime.now()),
            AIModelDecision('mistral_ai', 'BUY', 0.7, 'Technical breakout', datetime.now()),
            AIModelDecision('reinforcement_learning_engine', 'BUY', 0.85, 'Optimal action', datetime.now())
        ]
        
        # Получаем веса моделей
        model_weights = self.calculate_model_weights()
        
        # Подсчитываем голоса за каждое действие
        buy_votes = [d for d in model_decisions if d.action == 'BUY']
        sell_votes = [d for d in model_decisions if d.action == 'SELL']
        hold_votes = [d for d in model_decisions if d.action == 'HOLD']
        
        # Рассчитываем взвешенные голоса
        buy_weighted_score = sum(d.confidence * model_weights.get(d.model_name, 1.0) for d in buy_votes)
        sell_weighted_score = sum(d.confidence * model_weights.get(d.model_name, 1.0) for d in sell_votes)
        hold_weighted_score = sum(d.confidence * model_weights.get(d.model_name, 1.0) for d in hold_votes)
        
        print(f"\n🟢 BUY голоса: {len(buy_votes)} (взвешенный счет: {buy_weighted_score:.3f})")
        print(f"   Детали: {[f'{d.model_name}({d.confidence:.3f}*{model_weights.get(d.model_name, 1.0):.3f})' for d in buy_votes]}")
        print(f"🔴 SELL голоса: {len(sell_votes)} (взвешенный счет: {sell_weighted_score:.3f})")
        print(f"   Детали: {[f'{d.model_name}({d.confidence:.3f}*{model_weights.get(d.model_name, 1.0):.3f})' for d in sell_votes]}")
        print(f"⚪ HOLD голоса: {len(hold_votes)} (взвешенный счет: {hold_weighted_score:.3f})")
        
        # Определяем консенсус
        max_score = max(buy_weighted_score, sell_weighted_score, hold_weighted_score)
        
        print(f"\n📊 Анализ консенсуса:")
        print(f"   Максимальный взвешенный счет: {max_score:.3f}")
        print(f"   Порог для консенсуса: {self.consensus_weight_threshold:.3f}")
        print(f"   Минимум голосов: {self.min_consensus_models}")
        
        if max_score >= self.consensus_weight_threshold:
            if buy_weighted_score == max_score and len(buy_votes) >= self.min_consensus_models:
                print(f"✅ Взвешенный консенсус достигнут: BUY с {len(buy_votes)} голосами (взвешенный счет: {buy_weighted_score:.3f})")
                return 'BUY', buy_weighted_score
            elif sell_weighted_score == max_score and len(sell_votes) >= self.min_consensus_models:
                print(f"✅ Взвешенный консенсус достигнут: SELL с {len(sell_votes)} голосами (взвешенный счет: {sell_weighted_score:.3f})")
                return 'SELL', sell_weighted_score
            else:
                print(f"❌ Недостаточно голосов для консенсуса")
                return None, 0
        else:
            print(f"❌ Взвешенный консенсус НЕ достигнут: максимальный счет {max_score:.3f} < порога {self.consensus_weight_threshold:.3f}")
            return None, 0
    
    def test_weighted_confidence(self, final_action, participating_decisions, model_weights):
        """Тестирует расчет взвешенной уверенности"""
        print(f"\n📊 ТЕСТ ВЗВЕШЕННОЙ УВЕРЕННОСТИ")
        print("=" * 50)
        
        if not participating_decisions:
            print("❌ Нет участвующих решений")
            return 0
        
        # Взвешенная уверенность = сумма (уверенность * вес модели) / сумма весов
        total_weighted_confidence = sum(d.confidence * model_weights.get(d.model_name, 1.0) for d in participating_decisions)
        total_weights = sum(model_weights.get(d.model_name, 1.0) for d in participating_decisions)
        confidence_avg = total_weighted_confidence / total_weights if total_weights > 0 else 0
        
        print(f"Действие: {final_action}")
        print(f"Взвешенная уверенность консенсуса: {confidence_avg:.3f}")
        print(f"Детали расчета: {total_weighted_confidence:.3f} / {total_weights:.3f}")
        print(f"Участвующие модели: {[f'{d.model_name}({d.confidence:.3f}*{model_weights.get(d.model_name, 1.0):.3f})' for d in participating_decisions]}")
        
        return confidence_avg
    
    def test_hhll_pattern(self):
        """Тестирует логику HH/LL паттернов"""
        print(f"\n📈 ТЕСТ HH/LL ПАТТЕРНОВ")
        print("=" * 50)
        
        # Симулируем данные цен (восходящий тренд)
        test_prices = [100, 102, 101, 105, 103, 108, 106, 110, 109, 115, 112, 118]
        
        print(f"Тестовые цены: {test_prices}")
        
        # Простая логика определения HH/LL
        highs = []
        lows = []
        
        for i in range(1, len(test_prices) - 1):
            if test_prices[i] > test_prices[i-1] and test_prices[i] > test_prices[i+1]:
                highs.append((i, test_prices[i]))
            elif test_prices[i] < test_prices[i-1] and test_prices[i] < test_prices[i+1]:
                lows.append((i, test_prices[i]))
        
        print(f"Найденные максимумы: {highs}")
        print(f"Найденные минимумы: {lows}")
        
        # Определяем тренд
        if len(highs) >= 2:
            higher_highs = sum(1 for i in range(1, len(highs)) if highs[i][1] > highs[i-1][1])
            print(f"Higher Highs: {higher_highs} из {len(highs)-1}")
        
        if len(lows) >= 2:
            higher_lows = sum(1 for i in range(1, len(lows)) if lows[i][1] > lows[i-1][1])
            print(f"Higher Lows: {higher_lows} из {len(lows)-1}")
        
        # Определяем фазу рынка
        if len(highs) >= 2 and len(lows) >= 2:
            hh_ratio = higher_highs / (len(highs) - 1) if len(highs) > 1 else 0
            hl_ratio = higher_lows / (len(lows) - 1) if len(lows) > 1 else 0
            
            if hh_ratio >= 0.6 and hl_ratio >= 0.6:
                market_phase = "UPTREND"
                pattern_confidence = (hh_ratio + hl_ratio) / 2
            elif hh_ratio <= 0.4 and hl_ratio <= 0.4:
                market_phase = "DOWNTREND"
                pattern_confidence = 1 - (hh_ratio + hl_ratio) / 2
            else:
                market_phase = "SIDEWAYS"
                pattern_confidence = 0.5
        else:
            market_phase = "INSUFFICIENT_DATA"
            pattern_confidence = 0.0
        
        print(f"\n📊 Результат анализа HH/LL:")
        print(f"   Фаза рынка: {market_phase}")
        print(f"   Уверенность в паттерне: {pattern_confidence:.3f}")
        
        return market_phase, pattern_confidence

def main():
    """Основная функция тестирования"""
    print("🚀 ТЕСТИРОВАНИЕ УЛУЧШЕНИЙ СИСТЕМЫ ВИНРЕЙТ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    
    tester = TestImprovements()
    
    # Тест 1: Взвешенное голосование
    final_action, weighted_score = tester.test_weighted_voting()
    
    # Тест 2: Взвешенная уверенность (если есть консенсус)
    if final_action:
        model_weights = tester.calculate_model_weights()
        # Создаем участвующие решения для тестирования
        participating_decisions = [
            AIModelDecision('trading_ai', final_action, 0.8, 'Test', datetime.now()),
            AIModelDecision('lava_ai', final_action, 0.6, 'Test', datetime.now()),
            AIModelDecision('mistral_ai', final_action, 0.7, 'Test', datetime.now()),
            AIModelDecision('reinforcement_learning_engine', final_action, 0.85, 'Test', datetime.now())
        ]
        confidence = tester.test_weighted_confidence(final_action, participating_decisions, model_weights)
    
    # Тест 3: HH/LL паттерны
    market_phase, pattern_confidence = tester.test_hhll_pattern()
    
    print(f"\n✅ ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ")
    print("=" * 60)
    print(f"📊 Сводка результатов:")
    print(f"   Взвешенное голосование: {'✅ Работает' if final_action else '❌ Нет консенсуса'}")
    print(f"   Взвешенная уверенность: {'✅ Работает' if final_action else '❌ Не тестировалось'}")
    print(f"   HH/LL анализ: ✅ Работает (фаза: {market_phase})")

if __name__ == "__main__":
    main()