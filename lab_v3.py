import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Ввод экспериментальных данных
# -----------------------------

U = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120], dtype=float)

I_5cm  = np.array([0.4, 1.3, 1.9, 2.6, 2.9, 3.3, 3.5, 3.6, 3.7, 3.7, 3.7, 3.7], dtype=float)
I_10cm = np.array([0.02, 0.4, 0.6, 0.9, 1.1, 1.2, 1.3, 1.35, 1.4, 1.4, 1.4, 1.4], dtype=float)
I_20cm = np.array([0.02, 0.1, 0.2, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], dtype=float)

# расстояния источника света (см)
l = np.array([5.0, 10.0, 20.0])

# -----------------------------
# 2. Поиск токов насыщения
# -----------------------------

def saturation_current(I, tail_points=4):
    """
    Оценка тока насыщения как среднее по последним tail_points точкам.
    """
    return I[-tail_points:].mean()

I_sat_5  = saturation_current(I_5cm)
I_sat_10 = saturation_current(I_10cm)
I_sat_20 = saturation_current(I_20cm)

print("Токи насыщения (мкА):")
print(f"l =  5 см:  I_sat = {I_sat_5:.3f}")
print(f"l = 10 см:  I_sat = {I_sat_10:.3f}")
print(f"l = 20 см:  I_sat = {I_sat_20:.3f}")

# -----------------------------
# 3. Относительные световые потоки и чувствительность
#    Φ ~ 1 / l^2
# -----------------------------

# задаём относительный поток так, чтобы при 5 см было 100 усл. ед.
Phi_5 = 100.0
Phi  = Phi_5 * (l[0] / l)**2  # [Φ_5, Φ_10, Φ_20]

I_sat = np.array([I_sat_5, I_sat_10, I_sat_20])

gamma = I_sat / Phi  # интегральная чувствительность в мкА/(усл.ед. потока)
gamma_mean = gamma.mean()

print("\nОтносительные световые потоки (усл. ед.):")
for li, phi in zip(l, Phi):
    print(f"l = {li:4.1f} см: Φ_отн = {phi:.2f}")

print("\nИнтегральная световая чувствительность γ = I_sat / Φ:")
for li, Is, phi, g in zip(l, I_sat, Phi, gamma):
    print(f"l = {li:4.1f} см: I_sat = {Is:.3f} мкА, Φ = {phi:6.2f}, γ = {g:.4f} мкА/усл.ед.")

print(f"\nСредняя чувствительность: γ̄ = {gamma_mean:.4f} мкА/усл.ед.")

# -----------------------------
# 4. Построение ВА‑характеристик
# -----------------------------

plt.figure(figsize=(7, 5))
plt.plot(U, I_5cm,  "o-", label="l = 5 см")
plt.plot(U, I_10cm, "s-", label="l = 10 см")
plt.plot(U, I_20cm, "^-", label="l = 20 см")
plt.xlabel("Напряжение U, В")
plt.ylabel("Ток I, мкА")
plt.title("Вольт-амперные характеристики фотоэлемента")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 5. Световая характеристика: I_sat vs 1/l^2
# -----------------------------

inv_l2 = 1.0 / l**2

plt.figure(figsize=(6, 4))
plt.plot(inv_l2, I_sat, "o-")
plt.xlabel("1 / l², 1/см² (отн. интенсивность)")
plt.ylabel("I_sat, мкА")
plt.title("Световая характеристика (ток насыщения)")
plt.grid(True)
plt.tight_layout()
plt.show()

