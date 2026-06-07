import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parametry wejściowe Modelu Standardowego przy masie bozonu Z (M_Z = 91.1876 GeV)
MZ = 91.1876
alpha_1_MZ = 0.016923  # Znormalizowane wg GUT U(1)_Y
alpha_2_MZ = 0.03374   # SU(2)_L
alpha_3_MZ = 0.1184    # SU(3)_c

# Konwersja na g_i = sqrt(4 * pi * alpha_i)
g1_MZ = np.sqrt(4 * np.pi * alpha_1_MZ)
g2_MZ = np.sqrt(4 * np.pi * alpha_2_MZ)
g3_MZ = np.sqrt(4 * np.pi * alpha_3_MZ)

# Współczynniki Beta 1-pętlowe (z pracy SHZ-BCC, sekcja 9)
b = np.array([41.0/10.0, -19.0/6.0, -7.0])

# Macierz współczynników 2-pętlowych B_ij (z pracy SHZ-BCC)
B = np.array([
    [199./50., 27./10., 44./5.],
    [9./10., 35./6., 12.],
    [11./10., 9./2., -26.]
])

def rge_derivatives(t, g):
    # t = ln(mu), gdzie mu to skala energii
    g_arr = np.array(g)
    g3_pow = g_arr**3

    # Udział 1-pętlowy
    beta_1loop = (g3_pow / (16 * np.pi**2)) * b

    # Udział 2-pętlowy (zaniedbujemy mniejsze wkłady kwarku top/yukawa dla uproszczenia wykresu)
    g2_arr = g_arr**2
    beta_2loop = (g3_pow / (16 * np.pi**2)**2) * np.dot(B, g2_arr)

    return beta_1loop + beta_2loop

# Zakres całkowania: od ln(M_Z) do ln(10^17 GeV)
t_start = np.log(MZ)
t_end = np.log(1e17)
t_eval = np.linspace(t_start, t_end, 500)

print("Rozwiązuję równania różniczkowe RGE...")
sol = solve_ivp(rge_derivatives, [t_start, t_end], [g1_MZ, g2_MZ, g3_MZ], t_eval=t_eval, method='RK45')

# Konwersja z powrotem na odwrotności alfa_i (1/alpha_i) - standardowy sposób rysowania
mu_vals = np.exp(sol.t)
log10_mu = np.log10(mu_vals)
alpha_inv = (4 * np.pi) / (sol.y**2)

# RYSOWANIE WYKRESU
plt.figure(figsize=(10, 6))
plt.plot(log10_mu, alpha_inv[0], label=r'$\alpha_1^{-1}$ (Hiperładunek U(1))', color='#0055ff', linewidth=2)
plt.plot(log10_mu, alpha_inv[1], label=r'$\alpha_2^{-1}$ (Słabe SU(2))', color='#ff0055', linewidth=2)
plt.plot(log10_mu, alpha_inv[2], label=r'$\alpha_3^{-1}$ (Silne SU(3))', color='#00aa55', linewidth=2)

# Zaznaczenie docelowej skali Pati-Salama / SHZ-BCC (5 * 10^15 GeV)
MBCC = 5e15
plt.axvline(np.log10(MBCC), color='black', linestyle='--', label=r'Skala $M_{BCC}$ ($5 \times 10^{15}$ GeV)')

plt.title('Bieg stałych sprzężenia (RGE - 2 pętle)', fontsize=14)
plt.xlabel(r'$\log_{10}(\mu)$ [Energia w GeV]', fontsize=12)
plt.ylabel(r'$\alpha_i^{-1}(\mu)$', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()

# Zapis pliku
plt.savefig('/home/user/rge_plot.png', dpi=300)
print("Wykres został zapisany do rge_plot.png")

# Wypisanie wartości na skali docelowej
idx_BCC = np.argmin(np.abs(mu_vals - MBCC))
print(f"\nWartości 1/alpha na skali M_BCC = {MBCC:.1e} GeV:")
print(f"alpha_1^-1 = {alpha_inv[0][idx_BCC]:.2f}")
print(f"alpha_2^-1 = {alpha_inv[1][idx_BCC]:.2f}")
print(f"alpha_3^-1 = {alpha_inv[2][idx_BCC]:.2f}")
