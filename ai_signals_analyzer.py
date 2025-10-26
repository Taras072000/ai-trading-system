#!/usr/bin/env python3
"""
AI Signals Quality Analyzer
Анализатор качества AI сигналов для оценки производительности каждой модели

Автор: AI Assistant
Дата: 2025-01-22
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any
import os
from collections import defaultdict

class AISignalsAnalyzer:
    """Класс для анализа качества AI сигналов"""
    
    def __init__(self, csv_file_path: str):
        """
        Инициализация анализатора
        
        Args:
            csv_file_path: Путь к CSV файлу с данными торгов
        """
        self.csv_file_path = csv_file_path
        self.df = None
        self.ai_models = []
        self.symbols = []
        self.analysis_results = {}
        
    def load_data(self) -> bool:
        """
        Загрузка данных из CSV файла
        
        Returns:
            bool: True если данные успешно загружены
        """
        try:
            if not os.path.exists(self.csv_file_path):
                print(f"❌ Файл не найден: {self.csv_file_path}")
                return False
                
            self.df = pd.read_csv(self.csv_file_path)
            print(f"✅ Данные загружены: {len(self.df)} записей")
            
            # Переименовываем колонки для удобства работы
            column_mapping = {
                'Символ': 'symbol',
                'Направление': 'direction', 
                'Время входа': 'entry_time',
                'Время выхода': 'exit_time',
                'Цена входа': 'entry_price',
                'Цена выхода': 'exit_price',
                'P&L ($)': 'pnl',
                'P&L (%)': 'pnl_percent',
                'AI модель': 'ai_model',
                'Уверенность (%)': 'confidence',
                'Сила консенсуса': 'consensus_strength',
                'Время удержания (ч)': 'holding_time',
                'Размер позиции': 'position_size',
                'Комиссия': 'commission',
                'Результат': 'result'
            }
            
            self.df = self.df.rename(columns=column_mapping)
            
            # Преобразуем результат в английский формат
            self.df['result'] = self.df['result'].map({'Прибыль': 'profit', 'Убыток': 'loss'})
            
            # Получаем уникальные AI модели и символы
            self.ai_models = sorted(self.df['ai_model'].unique())
            self.symbols = sorted(self.df['symbol'].unique())
            
            print(f"📊 AI модели: {self.ai_models}")
            print(f"💱 Символы: {self.symbols}")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка загрузки данных: {e}")
            return False
    
    def analyze_model_performance(self, model_name: str) -> Dict[str, Any]:
        """
        Анализ производительности конкретной AI модели
        
        Args:
            model_name: Название AI модели
            
        Returns:
            Dict с результатами анализа
        """
        model_data = self.df[self.df['ai_model'] == model_name].copy()
        
        if len(model_data) == 0:
            return {"error": f"Нет данных для модели {model_name}"}
        
        # Основные метрики
        total_trades = len(model_data)
        profitable_trades = len(model_data[model_data['result'] == 'profit'])
        losing_trades = len(model_data[model_data['result'] == 'loss'])
        
        win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # P&L анализ
        total_pnl = model_data['pnl'].sum()
        avg_pnl = model_data['pnl'].mean()
        avg_profit = model_data[model_data['result'] == 'profit']['pnl'].mean() if profitable_trades > 0 else 0
        avg_loss = model_data[model_data['result'] == 'loss']['pnl'].mean() if losing_trades > 0 else 0
        
        # Лучшие и худшие сделки
        best_trade = model_data.loc[model_data['pnl'].idxmax()] if len(model_data) > 0 else None
        worst_trade = model_data.loc[model_data['pnl'].idxmin()] if len(model_data) > 0 else None
        
        # Анализ по символам
        symbol_analysis = {}
        for symbol in self.symbols:
            symbol_data = model_data[model_data['symbol'] == symbol]
            if len(symbol_data) > 0:
                symbol_analysis[symbol] = {
                    'trades': len(symbol_data),
                    'win_rate': (len(symbol_data[symbol_data['result'] == 'profit']) / len(symbol_data)) * 100,
                    'total_pnl': symbol_data['pnl'].sum(),
                    'avg_pnl': symbol_data['pnl'].mean()
                }
        
        # Анализ по направлениям
        direction_analysis = {}
        for direction in ['LONG', 'SHORT']:
            dir_data = model_data[model_data['direction'] == direction]
            if len(dir_data) > 0:
                direction_analysis[direction] = {
                    'trades': len(dir_data),
                    'win_rate': (len(dir_data[dir_data['result'] == 'profit']) / len(dir_data)) * 100,
                    'total_pnl': dir_data['pnl'].sum(),
                    'avg_pnl': dir_data['pnl'].mean()
                }
        
        # Анализ уверенности (confidence)
        avg_confidence = model_data['confidence'].mean()
        confidence_profitable = model_data[model_data['result'] == 'profit']['confidence'].mean() if profitable_trades > 0 else 0
        confidence_losing = model_data[model_data['result'] == 'loss']['confidence'].mean() if losing_trades > 0 else 0
        
        return {
            'model_name': model_name,
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'best_trade': {
                'symbol': best_trade['symbol'] if best_trade is not None else None,
                'pnl': best_trade['pnl'] if best_trade is not None else 0,
                'direction': best_trade['direction'] if best_trade is not None else None
            },
            'worst_trade': {
                'symbol': worst_trade['symbol'] if worst_trade is not None else None,
                'pnl': worst_trade['pnl'] if worst_trade is not None else 0,
                'direction': worst_trade['direction'] if worst_trade is not None else None
            },
            'symbol_analysis': symbol_analysis,
            'direction_analysis': direction_analysis,
            'avg_confidence': avg_confidence,
            'confidence_profitable': confidence_profitable,
            'confidence_losing': confidence_losing
        }
    
    def analyze_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Анализ всех AI моделей
        
        Returns:
            Dict с результатами анализа всех моделей
        """
        results = {}
        
        print("\n🔍 АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ AI МОДЕЛЕЙ")
        print("=" * 60)
        
        for model in self.ai_models:
            print(f"\n📈 Анализ модели: {model}")
            results[model] = self.analyze_model_performance(model)
            
        self.analysis_results = results
        return results
    
    def print_model_report(self, model_name: str, analysis: Dict[str, Any]):
        """
        Вывод детального отчета по модели
        
        Args:
            model_name: Название модели
            analysis: Результаты анализа модели
        """
        print(f"\n{'='*20} {model_name.upper()} {'='*20}")
        
        # Основные метрики
        print(f"📊 Общая статистика:")
        print(f"   • Всего сделок: {analysis['total_trades']}")
        print(f"   • Прибыльных: {analysis['profitable_trades']} ({analysis['win_rate']:.1f}%)")
        print(f"   • Убыточных: {analysis['losing_trades']} ({100-analysis['win_rate']:.1f}%)")
        print(f"   • Общий P&L: ${analysis['total_pnl']:.2f}")
        print(f"   • Средний P&L: ${analysis['avg_pnl']:.2f}")
        print(f"   • Средняя прибыль: ${analysis['avg_profit']:.2f}")
        print(f"   • Средний убыток: ${analysis['avg_loss']:.2f}")
        
        # Уверенность
        print(f"\n🎯 Анализ уверенности:")
        print(f"   • Средняя уверенность: {analysis['avg_confidence']:.1f}%")
        print(f"   • Уверенность прибыльных: {analysis['confidence_profitable']:.1f}%")
        print(f"   • Уверенность убыточных: {analysis['confidence_losing']:.1f}%")
        
        # Лучшие и худшие сделки
        print(f"\n🏆 Экстремальные сделки:")
        print(f"   • Лучшая: {analysis['best_trade']['symbol']} {analysis['best_trade']['direction']} = ${analysis['best_trade']['pnl']:.2f}")
        print(f"   • Худшая: {analysis['worst_trade']['symbol']} {analysis['worst_trade']['direction']} = ${analysis['worst_trade']['pnl']:.2f}")
        
        # Анализ по символам
        print(f"\n💱 Анализ по символам:")
        for symbol, data in analysis['symbol_analysis'].items():
            print(f"   • {symbol}: {data['trades']} сделок, WR: {data['win_rate']:.1f}%, P&L: ${data['total_pnl']:.2f}")
        
        # Анализ по направлениям
        print(f"\n📈📉 Анализ по направлениям:")
        for direction, data in analysis['direction_analysis'].items():
            print(f"   • {direction}: {data['trades']} сделок, WR: {data['win_rate']:.1f}%, P&L: ${data['total_pnl']:.2f}")
    
    def create_comparison_table(self) -> pd.DataFrame:
        """
        Создание сравнительной таблицы всех моделей
        
        Returns:
            DataFrame с сравнительной таблицей
        """
        comparison_data = []
        
        for model_name, analysis in self.analysis_results.items():
            comparison_data.append({
                'Модель': model_name,
                'Сделок': analysis['total_trades'],
                'Win Rate (%)': f"{analysis['win_rate']:.1f}%",
                'Общий P&L ($)': f"{analysis['total_pnl']:.2f}",
                'Средний P&L ($)': f"{analysis['avg_pnl']:.2f}",
                'Средняя уверенность (%)': f"{analysis['avg_confidence']:.1f}%",
                'Лучшая сделка ($)': f"{analysis['best_trade']['pnl']:.2f}",
                'Худшая сделка ($)': f"{analysis['worst_trade']['pnl']:.2f}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Сортировка по общему P&L (по убыванию)
        df_comparison = df_comparison.sort_values('Общий P&L ($)', key=lambda x: x.str.replace('$', '').astype(float), ascending=False)
        
        return df_comparison
    
    def print_comparison_table(self):
        """Вывод сравнительной таблицы"""
        print(f"\n{'='*80}")
        print("📊 СРАВНИТЕЛЬНАЯ ТАБЛИЦА AI МОДЕЛЕЙ")
        print(f"{'='*80}")
        
        df_comparison = self.create_comparison_table()
        
        # Красивый вывод таблицы
        print(df_comparison.to_string(index=False))
    
    def generate_recommendations(self) -> List[str]:
        """
        Генерация рекомендаций по оптимизации
        
        Returns:
            List рекомендаций
        """
        recommendations = []
        
        # Анализ лучших и худших моделей
        models_by_pnl = sorted(self.analysis_results.items(), key=lambda x: x[1]['total_pnl'], reverse=True)
        models_by_winrate = sorted(self.analysis_results.items(), key=lambda x: x[1]['win_rate'], reverse=True)
        
        best_pnl_model = models_by_pnl[0]
        worst_pnl_model = models_by_pnl[-1]
        best_wr_model = models_by_winrate[0]
        worst_wr_model = models_by_winrate[-1]
        
        recommendations.append(f"🏆 Лучшая модель по P&L: {best_pnl_model[0]} (${best_pnl_model[1]['total_pnl']:.2f})")
        recommendations.append(f"🎯 Лучшая модель по Win Rate: {best_wr_model[0]} ({best_wr_model[1]['win_rate']:.1f}%)")
        recommendations.append(f"⚠️ Худшая модель по P&L: {worst_pnl_model[0]} (${worst_pnl_model[1]['total_pnl']:.2f})")
        recommendations.append(f"⚠️ Худшая модель по Win Rate: {worst_wr_model[0]} ({worst_wr_model[1]['win_rate']:.1f}%)")
        
        # Анализ уверенности
        high_confidence_models = [name for name, data in self.analysis_results.items() if data['avg_confidence'] > 70]
        if high_confidence_models:
            recommendations.append(f"💪 Модели с высокой уверенностью (>70%): {', '.join(high_confidence_models)}")
        
        # Рекомендации по оптимизации
        recommendations.append("\n🔧 РЕКОМЕНДАЦИИ ПО ОПТИМИЗАЦИИ:")
        
        for model_name, analysis in self.analysis_results.items():
            if analysis['win_rate'] < 50:
                recommendations.append(f"   • {model_name}: Низкий Win Rate ({analysis['win_rate']:.1f}%) - требует пересмотра логики")
            
            if analysis['total_pnl'] < 0:
                recommendations.append(f"   • {model_name}: Отрицательный P&L (${analysis['total_pnl']:.2f}) - рассмотреть отключение")
            
            if analysis['avg_confidence'] < 50:
                recommendations.append(f"   • {model_name}: Низкая уверенность ({analysis['avg_confidence']:.1f}%) - улучшить калибровку")
        
        return recommendations
    
    def run_full_analysis(self):
        """Запуск полного анализа"""
        print("🚀 ЗАПУСК АНАЛИЗА КАЧЕСТВА AI СИГНАЛОВ")
        print("=" * 60)
        
        # Загрузка данных
        if not self.load_data():
            return
        
        # Анализ всех моделей
        self.analyze_all_models()
        
        # Детальные отчеты по каждой модели
        print(f"\n{'='*60}")
        print("📋 ДЕТАЛЬНЫЕ ОТЧЕТЫ ПО МОДЕЛЯМ")
        print(f"{'='*60}")
        
        for model_name, analysis in self.analysis_results.items():
            self.print_model_report(model_name, analysis)
        
        # Сравнительная таблица
        self.print_comparison_table()
        
        # Рекомендации
        print(f"\n{'='*60}")
        print("💡 РЕКОМЕНДАЦИИ")
        print(f"{'='*60}")
        
        recommendations = self.generate_recommendations()
        for rec in recommendations:
            print(rec)
        
        print(f"\n{'='*60}")
        print("✅ АНАЛИЗ ЗАВЕРШЕН")
        print(f"{'='*60}")


def main():
    """Основная функция"""
    # Путь к CSV файлу с результатами торгов
    csv_file_path = "/Users/mac/Documents/Peper Binance v4/reports/csv_reports/all_trades_20251022_104150.csv"
    
    # Создание и запуск анализатора
    analyzer = AISignalsAnalyzer(csv_file_path)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()