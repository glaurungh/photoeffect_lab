"""
Обработка данных лабораторной работы №8
"Изучение внешнего фотоэффекта"
Автор: [Ваше имя]
Дата: 2026-01-30
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# =============================================================================
# 1. ИСХОДНЫЕ ДАННЫЕ И КОНСТАНТЫ УСТАНОВКИ
# =============================================================================

# Параметры установки (из методических указаний)
I_sv = 1.0          # Сила света источника, кд
dI_sv = 0.1         # Абсолютная погрешность силы света, кд (10%)
S = 23.0            # Площадь катода, см²
dS = 0.5            # Абсолютная погрешность площади, см²
delta_l_abs = 0.5   # Абсолютная погрешность измерения расстояния, см
                    # (цена деления линейки или установочная погрешность)

# Данные измерений
distances = [5, 10, 20]  # см

# Данные для l = 5 см
U_5 = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])  # В
I_5 = np.array([0.4, 1.3, 1.9, 2.6, 2.9, 3.3, 3.5, 3.6, 3.7, 3.7, 3.7, 3.7])  # мкА

# Данные для l = 10 см
U_10 = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])  # В
I_10 = np.array([0.02, 0.4, 0.6, 0.9, 1.1, 1.2, 1.3, 1.35, 1.4, 1.4, 1.4, 1.4])  # мкА

# Данные для l = 20 см
U_20 = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])  # В
I_20 = np.array([0.02, 0.1, 0.2, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])  # мкА

data_dict = {
    5: {'U': U_5, 'I': I_5},
    10: {'U': U_10, 'I': I_10},
    20: {'U': U_20, 'I': I_20}
}

# =============================================================================
# 2. ОБРАБОТКА ДАННЫХ
# =============================================================================

def calculate_flux(l_cm):
    """
    Расчёт светового потока Φ = I_sv * S / l²
    """
    return I_sv * S / (l_cm ** 2)

def calculate_flux_error(l_cm):
    """
    Относительная погрешность светового потока (в долях единицы)
    δΦ/Φ = √[(δI/I)² + (δS/S)² + (2·δl/l)²]
    """
    term1 = (dI_sv / I_sv) ** 2
    term2 = (dS / S) ** 2
    term3 = (2 * delta_l_abs / l_cm) ** 2
    return np.sqrt(term1 + term2 + term3)

def get_saturation_current(I_array, threshold=0.001):
    """
    Определение тока насыщения как среднего значения на плато.
    Берётся среднее последних 3-4 точек, где изменение меньше threshold.
    """
    # Берём последние 3 значения как плато
    return np.mean(I_array[-3:])

results = []

print("=" * 70)
print("ЛАБОРАТОРНАЯ РАБОТА №8: ИЗУЧЕНИЕ ВНЕШНЕГО ФОТОЭФФЕКТА")
print("=" * 70)
print("\nТаблица обработки результатов:\n")
print(f"{'l, см':<10} {'Φ, лм':<12} {'ΔΦ, лм':<10} {'I_н, мкА':<12} {'γ, мкА/лм':<12} {'Δγ, мкА/лм':<10}")
print("-" * 70)

gamma_values = []
gamma_errors = []

for l in distances:
    # Расчёт светового потока
    flux = calculate_flux(l)
    delta_flux_rel = calculate_flux_error(l)
    delta_flux_abs = flux * delta_flux_rel
    
    # Определение тока насыщения (последние 3 точки)
    I_data = data_dict[l]['I']
    I_sat = get_saturation_current(I_data)
    
    # Оценка погрешности тока (примерно 5% от прибора + разброс)
    delta_I_sat = I_sat * 0.05  # 5% инструментальная погрешность
    
    # Расчёт чувствительности γ = I_sat / Φ
    gamma = I_sat / flux
    delta_gamma = gamma * np.sqrt((delta_I_sat/I_sat)**2 + (delta_flux_rel)**2)
    
    gamma_values.append(gamma)
    gamma_errors.append(delta_gamma)
    
    results.append({
        'l': l,
        'Phi': flux,
        'dPhi': delta_flux_abs,
        'I_sat': I_sat,
        'dI_sat': delta_I_sat,
        'gamma': gamma,
        'dgamma': delta_gamma
    })
    
    print(f"{l:<10} {flux:<12.3f} {delta_flux_abs:<10.3f} {I_sat:<12.2f} {gamma:<12.2f} {delta_gamma:<10.2f}")

# Средневзвешенное значение чувствительности
gamma_array = np.array(gamma_values)
# Веса обратны квадратам погрешностей
weights = 1 / np.array(gamma_errors)**2
gamma_weighted = np.sum(gamma_array * weights) / np.sum(weights)
gamma_weighted_err = np.sqrt(1 / np.sum(weights))

# Простое среднее
gamma_mean = np.mean(gamma_values)
gamma_mean_err = np.std(gamma_values, ddof=1) / np.sqrt(len(gamma_values))

print("-" * 70)
print(f"\nСреднее значение чувствительности: γ = {gamma_mean:.2f} ± {gamma_mean_err:.2f} мкА/лм")
print(f"Средневзвешенное значение: γ = {gamma_weighted:.2f} ± {gamma_weighted_err:.2f} мкА/лм")
print(f"Относительная погрешность: {gamma_weighted_err/gamma_weighted*100:.1f}%")

# =============================================================================
# 3. ПОСТРОЕНИЕ ГРАФИКОВ
# =============================================================================

# График 1: Вольтамперные характеристики
fig1, ax1 = plt.subplots(figsize=(10, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
markers = ['o', 's', '^']

for i, l in enumerate(distances):
    U = data_dict[l]['U']
    I = data_dict[l]['I']
    ax1.plot(U, I, marker=markers[i], markersize=6, linewidth=1.5, 
             label=f'l = {l} см', color=colors[i], markerfacecolor='white', 
             markeredgewidth=1.5)

ax1.set_xlabel('Напряжение U, В', fontsize=12)
ax1.set_ylabel('Фототок I, мкА', fontsize=12)
ax1.set_title('Вольтамперные характеристики фотоэлемента', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, title='Расстояние до источника')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(0, 130)
ax1.set_ylim(0, None)
plt.tight_layout()
plt.savefig('photoeffect_IV_curves.png', dpi=300, bbox_inches='tight')
print("\n[✓] График I(U) сохранён как 'photoeffect_IV_curves.png'")

# График 2: Световая характеристика (линейная аппроксимация)
fig2, ax2 = plt.subplots(figsize=(10, 6))

phi_values = [r['Phi'] for r in results]
phi_errors = [r['dPhi'] for r in results]
I_sat_values = [r['I_sat'] for r in results]
I_sat_errors = [r['dI_sat'] for r in results]

ax2.errorbar(phi_values, I_sat_values, xerr=phi_errors, yerr=I_sat_errors, 
             fmt='ko', markersize=8, capsize=5, capthick=1, elinewidth=1, 
             label='Экспериментальные точки')

# Линейная аппроксимация (метод наименьших квадратов)
slope, intercept, r_value, p_value, std_err = linregress(phi_values, I_sat_values)
x_fit = np.linspace(0, max(phi_values)*1.2, 100)
y_fit = slope * x_fit + intercept

ax2.plot(x_fit, y_fit, 'r--', linewidth=2, 
         label=f'Линейная аппроксимация\nγ = {slope:.2f} ± {std_err:.2f} мкА/лм\nR² = {r_value**2:.4f}')

ax2.set_xlabel('Световой поток Φ, лм', fontsize=12)
ax2.set_ylabel('Ток насыщения I_н, мкА', fontsize=12)
ax2.set_title('Световая характеристика фотоэлемента', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlim(0, max(phi_values)*1.1)
ax2.set_ylim(0, max(I_sat_values)*1.2)
plt.tight_layout()
plt.savefig('photoeffect_light_curve.png', dpi=300, bbox_inches='tight')
print("[✓] График I_н(Φ) сохранён как 'photoeffect_light_curve.png'")

# =============================================================================
# 4. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ В ФАЙЛ
# =============================================================================

# Создаём сводную таблицу
df = pd.DataFrame({
    'l (см)': distances,
    'Phi (лм)': [f"{r['Phi']:.3f} ± {r['dPhi']:.3f}" for r in results],
    'I_sat (мкА)': [f"{r['I_sat']:.2f} ± {r['dI_sat']:.2f}" for r in results],
    'gamma (мкА/лм)': [f"{r['gamma']:.2f} ± {r['dgamma']:.2f}" for r in results]
})

print("\n" + "=" * 70)
print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("=" * 70)
print(df.to_string(index=False))

# Сохранение в CSV
df.to_csv('photoeffect_results.csv', index=False, encoding='utf-8-sig')
print("\n[✓] Таблица результатов сохранена в 'photoeffect_results.csv'")

# Вывод промежуточных расчётов для протокола
print("\n" + "=" * 70)
print("ПРОМЕЖУТОЧНЫЕ ВЫЧИСЛЕНИЯ (для записи в тетрадь):")
print("=" * 70)

for i, l in enumerate(distances):
    r = results[i]
    print(f"\nДля l = {l} см:")
    print(f"  S = {S} см² = {S*1e-4:.4f} м²")
    print(f"  Ω = S/l² = {S}/{l}² = {r['Phi']/I_sv:.4f} стерадиан")
    print(f"  Φ = I_sv·Ω = {I_sv}·{r['Phi']/I_sv:.4f} = {r['Phi']:.4f} лм")
    print(f"  I_н = {r['I_sat']:.2f} мкА (среднее последних 3 точек)")
    print(f"  γ = I_н/Φ = {r['I_sat']:.2f}/{r['Phi']:.4f} = {r['gamma']:.2f} мкА/лм")
    print(f"  Δγ/γ = √[(ΔI/I)² + (ΔΦ/Φ)²] = √[({r['dI_sat']:.2f}/{r['I_sat']:.2f})² + ({r['dPhi']:.3f}/{r['Phi']:.3f})²] = {r['dgamma']/r['gamma']:.2%}")

plt.show()
print("\nГотово!")
