import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Константы
I_sv = 1.0  # сила света лампочки, кд
S_cm2 = 23  # площадь катода, см²
S_m2 = S_cm2 * 1e-4  # площадь катода в м²

# Данные измерений (напряжение В, ток мкА)
data_l5 = np.array([
    [10, 0.4], [20, 1.3], [30, 1.9], [40, 2.6], [50, 2.9],
    [60, 3.3], [70, 3.5], [80, 3.6], [90, 3.7], [100, 3.7],
    [110, 3.7], [120, 3.7]
])

data_l10 = np.array([
    [10, 0.02], [20, 0.4], [30, 0.6], [40, 0.9], [50, 1.1],
    [60, 1.2], [70, 1.3], [80, 1.35], [90, 1.4], [100, 1.4],
    [110, 1.4], [120, 1.4]
])

data_l20 = np.array([
    [10, 0.02], [20, 0.1], [30, 0.2], [40, 0.3], [50, 0.4],
    [60, 0.4], [70, 0.4], [80, 0.4], [90, 0.4], [100, 0.4],
    [110, 0.4], [120, 0.4]
])

# Расстояния в метрах
distances = np.array([0.05, 0.10, 0.20])  # 5 см, 10 см, 20 см

# Функция для определения тока насыщения (среднее значение последних 4 измерений)
def get_saturation_current(data):
    # Берем последние 4 значения тока, где он стабилизировался
    saturation_values = data[-4:, 1]
    return np.mean(saturation_values)

# Определяем токи насыщения
I_sat_5 = get_saturation_current(data_l5)
I_sat_10 = get_saturation_current(data_l10)
I_sat_20 = get_saturation_current(data_l20)
I_sat = np.array([I_sat_5, I_sat_10, I_sat_20])  # в мкА

print("Токи насыщения:")
print(f"l = 5 см:  I_нас = {I_sat_5:.2f} мкА")
print(f"l = 10 см: I_нас = {I_sat_10:.2f} мкА")
print(f"l = 20 см: I_нас = {I_sat_20:.2f} мкА")
print()

# Расчет световых потоков (люмены)
# Формула: Φ = Iсв * S / l²
flux = I_sv * S_m2 / (distances ** 2)

print("Световые потоки и чувствительность:")
print(f"{'Расстояние (см)':<20} {'Световой поток (лм)':<25} {'Ток насыщения (мкА)':<25} {'Чувствительность γ (мкА/лм)':<30}")
print("-" * 100)

gamma_values = []
for i, (l, phi, i_sat) in enumerate(zip(distances * 100, flux, I_sat)):
    gamma = i_sat / phi
    gamma_values.append(gamma)
    print(f"{l:<20.0f} {phi:<25.4f} {i_sat:<25.2f} {gamma:<30.2f}")

gamma_values = np.array(gamma_values)
gamma_avg = np.mean(gamma_values)
gamma_std = np.std(gamma_values)

print("-" * 100)
print(f"Средняя чувствительность: γ_ср = {gamma_avg:.2f} ± {gamma_std:.2f} мкА/лм")
print(f"Относительная погрешность: {(gamma_std/gamma_avg)*100:.1f}%")
print()

# Построение графиков
plt.figure(figsize=(14, 6))

# График 1: Вольт-амперные характеристики
plt.subplot(1, 2, 1)
plt.plot(data_l5[:, 0], data_l5[:, 1], 'o-', label='l = 5 см', linewidth=2, markersize=6)
plt.plot(data_l10[:, 0], data_l10[:, 1], 's-', label='l = 10 см', linewidth=2, markersize=6)
plt.plot(data_l20[:, 0], data_l20[:, 1], '^-', label='l = 20 см', linewidth=2, markersize=6)
plt.xlabel('Напряжение U, В', fontsize=12)
plt.ylabel('Ток I, мкА', fontsize=12)
plt.title('Вольт-амперные характеристики фотоэлемента', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.xlim(0, 130)
plt.ylim(0, max(data_l5[:, 1]) * 1.1)

# График 2: Световая характеристика (I_нас = f(Φ))
plt.subplot(1, 2, 2)

# Функция для аппроксимации (линейная зависимость)
def linear_func(x, a, b):
    return a * x + b

# Аппроксимация данных
popt, pcov = curve_fit(linear_func, flux, I_sat)
a, b = popt
flux_fit = np.linspace(0, max(flux) * 1.1, 100)
I_fit = linear_func(flux_fit, a, b)

plt.plot(flux, I_sat, 'ro', markersize=10, label='Экспериментальные данные')
plt.plot(flux_fit, I_fit, 'b--', linewidth=2, label=f'Аппроксимация: I = {a:.2f}Φ + {b:.2f}')
plt.xlabel('Световой поток Φ, лм', fontsize=12)
plt.ylabel('Ток насыщения I_нас, мкА', fontsize=12)
plt.title('Световая характеристика фотоэлемента', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.xlim(0, max(flux) * 1.15)
plt.ylim(0, max(I_sat) * 1.15)

# Добавление аннотаций с координатами точек
for i, (phi, i_sat) in enumerate(zip(flux, I_sat)):
    plt.annotate(f'l={int(distances[i]*100)} см', 
                xy=(phi, i_sat), 
                xytext=(10, 10), 
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))

plt.tight_layout()
plt.savefig('photoeffect_results.png', dpi=300, bbox_inches='tight')
print("Графики сохранены в файл 'photoeffect_results.png'")
plt.show()

# Расчет погрешностей
print("\nРасчет погрешностей:")
print("=" * 70)

# Погрешность измерения расстояния (примерно ±0.5 см)
delta_l = 0.005  # м

# Погрешность площади катода
delta_S = 0.5 * 1e-4  # м² (из условия: 23 ± 0.5 см²)

# Погрешность силы света
delta_Isv = 0.1  # кд (из условия: 1.0 ± 0.1 кд)

# Расчет относительной погрешности для светового потока
print("\nОтносительные погрешности для светового потока Φ = Iсв·S/l²:")
for i, l in enumerate(distances):
    rel_err_phi = np.sqrt(
        (delta_Isv / I_sv)**2 + 
        (delta_S / S_m2)**2 + 
        (2 * delta_l / l)**2
    )
    abs_err_phi = flux[i] * rel_err_phi
    print(f"l = {l*100:.0f} см: δΦ/Φ = {rel_err_phi*100:.2f}%  →  δΦ = {abs_err_phi:.5f} лм")

# Погрешность измерения тока (предположим ±0.05 мкА для малых токов и ±0.1 мкА для больших)
delta_I = np.array([0.1, 0.05, 0.02])  # мкА

print("\nОтносительные погрешности для чувствительности γ = I_нас/Φ:")
gamma_errors = []
for i in range(3):
    rel_err_gamma = np.sqrt((delta_I[i] / I_sat[i])**2 + (delta_l * 2 / distances[i])**2)
    abs_err_gamma = gamma_values[i] * rel_err_gamma
    gamma_errors.append(abs_err_gamma)
    print(f"l = {distances[i]*100:.0f} см: δγ/γ = {rel_err_gamma*100:.2f}%  →  δγ = {abs_err_gamma:.2f} мкА/лм")

gamma_errors = np.array(gamma_errors)
print(f"\nИтоговый результат: γ = {gamma_avg:.2f} ± {np.mean(gamma_errors):.2f} мкА/лм")
print(f"Относительная погрешность: {(np.mean(gamma_errors)/gamma_avg)*100:.1f}%")

# Сохранение результатов в файл
with open('photoeffect_results.txt', 'w', encoding='utf-8') as f:
    f.write("ЛАБОРАТОРНАЯ РАБОТА: ИЗУЧЕНИЕ ВНЕШНЕГО ФОТОЭФФЕКТА\n")
    f.write("=" * 70 + "\n\n")
    f.write("Исходные данные:\n")
    f.write(f"  Сила света лампочки: Iсв = {I_sv} ± {delta_Isv} кд\n")
    f.write(f"  Площадь катода: S = {S_cm2} ± 0.5 см² = {S_m2:.4f} м²\n\n")
    
    f.write("Результаты измерений:\n")
    f.write(f"{'l, см':<10} {'Φ, лм':<15} {'I_нас, мкА':<15} {'γ, мкА/лм':<15}\n")
    f.write("-" * 55 + "\n")
    for i in range(3):
        f.write(f"{distances[i]*100:<10.0f} {flux[i]:<15.4f} {I_sat[i]:<15.2f} {gamma_values[i]:<15.2f}\n")
    f.write("-" * 55 + "\n")
    f.write(f"{'Среднее:':<10} {'':<15} {'':<15} {gamma_avg:<15.2f}\n\n")
    
    f.write(f"Итоговый результат: γ = {gamma_avg:.2f} ± {np.mean(gamma_errors):.2f} мкА/лм\n")
    f.write(f"Относительная погрешность: {(np.mean(gamma_errors)/gamma_avg)*100:.1f}%\n")

print("\nРезультаты расчетов сохранены в файл 'photoeffect_results.txt'")
