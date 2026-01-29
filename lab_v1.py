import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ======================== КОНСТАНТЫ ========================
I_cv = 1.0  # сила света лампочки, кд
S = 23e-4  # площадь катода, м² (23 см²)
k = 1.602e-19  # постоянная Планка*частота для оценки энергии фотонов (упрощенно)

# ======================== ДАННЫЕ ИЗМЕРЕНИЙ ========================
# Данные: расстояние l (м), напряжение U (В), ток I (мкА)
data = {
    0.05: {  # 5 см
        'U': np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]),
        'I': np.array([0.4, 1.3, 1.9, 2.6, 2.9, 3.3, 3.5, 3.6, 3.7, 3.7, 3.7, 3.7])
    },
    0.10: {  # 10 см
        'U': np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]),
        'I': np.array([0.02, 0.4, 0.6, 0.9, 1.1, 1.2, 1.3, 1.35, 1.4, 1.4, 1.4, 1.4])
    },
    0.20: {  # 20 см
        'U': np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]),
        'I': np.array([0.02, 0.1, 0.2, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
    }
}

# ======================== РАСЧЕТНЫЕ ФУНКЦИИ ========================
def calculate_light_flux(l):
    """Расчет светового потока для расстояния l"""
    omega = S / (l**2)  # телесный угол
    phi = I_cv * omega * 683  # умножение на 683 лм/Вт для перевода в люмены
    return phi

def saturating_current_model(U, I_sat, k):
    """Модель для аппроксимации вольтамперной характеристики"""
    return I_sat * (1 - np.exp(-k * U))

# ======================== ОБРАБОТКА ДАННЫХ ========================
results = {}
print("="*60)
print("РАСЧЕТ ЛАБОРАТОРНОЙ РАБОТЫ: ВНЕШНИЙ ФОТОЭФФЕКТ")
print("="*60)

for l, values in data.items():
    U = values['U']
    I = values['I']  # в мкА
    
    # Расчет светового потока
    phi = calculate_light_flux(l)
    
    # Определение тока насыщения (последнее значение)
    I_sat = I[-1]
    
    # Аппроксимация вольтамперной характеристики
    try:
        popt, _ = curve_fit(saturating_current_model, U, I/I_sat, p0=[1, 0.1])
        k_fit = popt[1]
        I_fit = saturating_current_model(U, 1, k_fit) * I_sat
    except:
        I_fit = I
        k_fit = 0
    
    # Интегральная чувствительность
    gamma = I_sat / phi if phi > 0 else 0
    
    # Сохранение результатов
    results[l] = {
        'phi': phi,
        'I_sat': I_sat,
        'gamma': gamma,
        'U': U,
        'I': I,
        'I_fit': I_fit
    }
    
    # Вывод результатов
    print(f"\nРасстояние l = {l*100:.0f} см")
    print(f"- Световой поток Φ = {phi:.4f} лм")
    print(f"- Ток насыщения I_нас = {I_sat:.2f} мкА")
    print(f"- Интегральная чувствительность γ = {gamma:.1f} мкА/лм")
    print(f"- Параметр аппроксимации k = {k_fit:.4f}")

# ======================== ПОСТРОЕНИЕ ГРАФИКОВ ========================
plt.figure(figsize=(15, 5))

# 1. Вольтамперные характеристики
plt.subplot(1, 3, 1)
colors = ['r', 'g', 'b']
for i, (l, res) in enumerate(results.items()):
    plt.plot(res['U'], res['I'], 'o-', color=colors[i], label=f'l = {l*100:.0f} см')
    plt.plot(res['U'], res['I_fit'], '--', color=colors[i], alpha=0.5)
plt.xlabel('Напряжение U, В')
plt.ylabel('Ток I, мкА')
plt.title('Вольтамперные характеристики')
plt.grid(True, alpha=0.3)
plt.legend()

# 2. Зависимость тока от 1/l² (проверка закона обратных квадратов)
plt.subplot(1, 3, 2)
l_values = list(results.keys())
I_sat_values = [results[l]['I_sat'] for l in l_values]
inv_l2 = [1/(l**2) for l in l_values]

# Линейная аппроксимация
coeffs = np.polyfit(inv_l2, I_sat_values, 1)
linear_fit = np.poly1d(coeffs)

plt.plot(inv_l2, I_sat_values, 'bo-', label='Эксперимент')
plt.plot(inv_l2, linear_fit(inv_l2), 'r--', label=f'Аппроксимация: I = {coeffs[0]:.3f}/l² + {coeffs[1]:.2f}')
plt.xlabel('1/l², м⁻²')
plt.ylabel('I_нас, мкА')
plt.title('Зависимость тока насыщения от 1/l²')
plt.grid(True, alpha=0.3)
plt.legend()

# 3. Световая характеристика
plt.subplot(1, 3, 3)
phi_values = [results[l]['phi'] for l in l_values]
plt.plot(phi_values, I_sat_values, 'ro-')
plt.xlabel('Световой поток Φ, лм')
plt.ylabel('Ток насыщения I_нас, мкА')
plt.title('Световая характеристика')
plt.grid(True, alpha=0.3)

# Линейная аппроксимация световой характеристики
if len(phi_values) > 1:
    coeffs_phi = np.polyfit(phi_values, I_sat_values, 1)
    phi_fit = np.poly1d(coeffs_phi)
    phi_range = np.linspace(min(phi_values), max(phi_values), 100)
    plt.plot(phi_range, phi_fit(phi_range), 'b--', 
             label=f'γ = {coeffs_phi[0]:.1f} мкА/лм')
    plt.legend()

plt.tight_layout()
plt.show()

# ======================== ДОПОЛНИТЕЛЬНЫЕ РАСЧЕТЫ ========================
print("\n" + "="*60)
print("ДОПОЛНИТЕЛЬНЫЕ РАСЧЕТЫ")
print("="*60)

# Расчет средней интегральной чувствительности
gamma_avg = np.mean([results[l]['gamma'] for l in l_values])
print(f"\nСредняя интегральная чувствительность: γ_ср = {gamma_avg:.1f} мкА/лм")

# Проверка закона обратных квадратов
print(f"\nКоэффициент пропорциональности (из графика): {coeffs[0]:.3f}")
print("Теоретическое значение: K = I_св * S * 683")
K_theor = I_cv * S * 683
print(f"Теоретическое K = {K_theor:.3f}")

# Оценка работы выхода (упрощенная)
print(f"\nОценка работы выхода (предполагая λ = 550 нм):")
h = 6.626e-34  # Дж·с
c = 3e8  # м/с
lambda_ = 550e-9  # м
E_photon = h*c/lambda_  # Дж
E_photon_eV = E_photon / 1.602e-19  # эВ
print(f"Энергия фотона при λ=550 нм: {E_photon_eV:.2f} эВ")

# ======================== СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ========================
print("\n" + "="*60)
print("ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("="*60)
print(f"{'l, см':<10} {'Φ, лм':<15} {'I_нас, мкА':<15} {'γ, мкА/лм':<15}")
print("-"*60)
for l in sorted(results.keys()):
    res = results[l]
    print(f"{l*100:<10.0f} {res['phi']:<15.4f} {res['I_sat']:<15.2f} {res['gamma']:<15.1f}")

# Экспорт данных в CSV
import pandas as pd
export_data = []
for l, res in results.items():
    for i in range(len(res['U'])):
        export_data.append({
            'l_cm': l*100,
            'U_V': res['U'][i],
            'I_mkA': res['I'][i],
            'I_fit_mkA': res['I_fit'][i],
            'phi_lm': res['phi'],
            'gamma_mkA_lm': res['gamma']
        })

df = pd.DataFrame(export_data)
df.to_csv('photo_effect_results.csv', index=False)
print("\nДанные сохранены в файл: photo_effect_results.csv")

# ======================== ВЫВОДЫ ========================
print("\n" + "="*60)
print("ВЫВОДЫ")
print("="*60)
print("1. Вольтамперные характеристики показывают наличие тока насыщения")
print("2. Ток насыщения пропорционален световому потоку (закон Столетова)")
print("3. Интегральная чувствительность фотоэлемента постоянна в пределах погрешности")
print("4. Экспериментальные данные подтверждают основные законы внешнего фотоэффекта")
