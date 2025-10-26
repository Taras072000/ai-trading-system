#!/usr/bin/env python3
import pandas as pd
import numpy as np

# Загружаем данные
df = pd.read_csv('reports/csv_reports/all_trades_20251022_125859.csv')

# Основная статистика
total_trades = len(df)
profitable_trades = len(df[df['Результат'] == 'Прибыль'])
losing_trades = len(df[df['Результат'] == 'Убыток'])
winrate = (profitable_trades / total_trades) * 100

# Прибыль/убыток
total_pnl = df['P&L ($)'].sum()
avg_profit = df[df['Результат'] == 'Прибыль']['P&L ($)'].mean()
avg_loss = df[df['Результат'] == 'Убыток']['P&L ($)'].mean()

# Статистика по символам
symbol_stats = df.groupby('Символ').agg({
    'Результат': lambda x: (x == 'Прибыль').sum() / len(x) * 100,
    'P&L ($)': 'sum'
}).round(2)

print('🎯 РЕЗУЛЬТАТЫ ОПТИМИЗИРОВАННОЙ СИСТЕМЫ')
print('=' * 50)
print(f'📊 Общая статистика:')
print(f'   Всего сделок: {total_trades}')
print(f'   Прибыльных: {profitable_trades}')
print(f'   Убыточных: {losing_trades}')
print(f'   ВИНРЕЙТ: {winrate:.1f}%')
print()
print(f'💰 Финансовые результаты:')
print(f'   Общая прибыль: ${total_pnl:.2f}')
print(f'   Средняя прибыль: ${avg_profit:.2f}')
print(f'   Средний убыток: ${avg_loss:.2f}')
print()
print(f'📈 Статистика по символам:')
for symbol, row in symbol_stats.iterrows():
    print(f'   {symbol}: Винрейт {row["Результат"]:.1f}%, P&L ${row["P&L ($)"]:.2f}')

print()
print('🔍 ДЕТАЛЬНЫЙ АНАЛИЗ:')
print('=' * 50)

# Анализ по направлениям
direction_stats = df.groupby('Направление').agg({
    'Результат': lambda x: (x == 'Прибыль').sum() / len(x) * 100,
    'P&L ($)': 'sum'
}).round(2)

print(f'📊 По направлениям:')
for direction, row in direction_stats.iterrows():
    print(f'   {direction}: Винрейт {row["Результат"]:.1f}%, P&L ${row["P&L ($)"]:.2f}')

# Анализ консенсуса
consensus_stats = df.groupby('Сила консенсуса').agg({
    'Результат': lambda x: (x == 'Прибыль').sum() / len(x) * 100,
    'P&L ($)': 'sum'
}).round(2)

print(f'📊 По силе консенсуса:')
for consensus, row in consensus_stats.iterrows():
    print(f'   {consensus} моделей: Винрейт {row["Результат"]:.1f}%, P&L ${row["P&L ($)"]:.2f}')

# Максимальная просадка
df['Cumulative_PnL'] = df['P&L ($)'].cumsum()
running_max = df['Cumulative_PnL'].expanding().max()
drawdown = df['Cumulative_PnL'] - running_max
max_drawdown = drawdown.min()

print()
print(f'📉 Максимальная просадка: ${max_drawdown:.2f}')

# Сравнение с предыдущими результатами
print()
print('📊 СРАВНЕНИЕ С ПРЕДЫДУЩИМИ РЕЗУЛЬТАТАМИ:')
print('=' * 50)
print(f'   Предыдущий винрейт: 45.9%')
print(f'   Новый винрейт: {winrate:.1f}%')
print(f'   Улучшение: {winrate - 45.9:.1f} процентных пунктов')
print()
if winrate >= 70:
    print('✅ ЦЕЛЬ ДОСТИГНУТА! Винрейт 70%+ достигнут!')
elif winrate >= 60:
    print('🟡 ХОРОШИЙ РЕЗУЛЬТАТ! Близко к цели 70%')
else:
    print('🔴 ЦЕЛЬ НЕ ДОСТИГНУТА. Требуется дополнительная оптимизация')