import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Warunki początkowe
MZ = 91.1876
alpha_1_MZ = 0.016923
alpha_2_MZ = 0.03374
alpha_3_MZ = 0.1184

g1_MZ = np.sqrt(4 * np.pi * alpha_1_MZ)
g2_MZ = np.sqrt(4 * np.pi * alpha_2_MZ)
g3_MZ = np.sqrt(4 * np.pi * alpha_3_MZ)

# === Współczynniki dla Modelu Standardowego (SM) ===
b_SM = np.array([41.0/10.0, -19.0/6.0, -7.0])
B_SM = np.array([
    [199./50., 27./10., 44./5.],
    [9./10., 35./6., 12.],
    [11./10., 9./2., -26.]
])

# === Współczynniki dla Supersymetrii (MSSM) ===
b_MSSM = np.array([33.0/5.0, 1.0, -3.0])
B_MSSM = np.array([
    [199./25., 27./5., 88./5.],
    [9./5., 25.0, 24.0],
    [11./5., 9.0, 14.0]
])

def get_derivatives(b_vec, B_mat):
    def rge_derivatives(t, g):
        g_arr = np.array(g)
        g3_pow = g_arr**3
        beta_1loop = (g3_pow / (16 * np.pi**2)) * b_vec
        g2_arr = g_arr**2
        beta_2loop = (g3_pow / (16 * np.pi**2)**2) * np.dot(B_mat, g2_arr)
        return beta_1loop + beta_2loop
    return rge_derivatives

# Skale energii (logarytmicznie)
t_MZ = np.log(MZ)
t_SUSY = np.log(1000.0) # Skala SUSY: 1 TeV
t_GUT = np.log(1e17)

# 1. Obliczenie dla czystego SM na całym zakresie (jako tło referencyjne)
t_eval_SM = np.linspace(t_MZ, t_GUT, 200)
sol_SM = solve_ivp(get_derivatives(b_SM, B_SM), [t_MZ, t_GUT], [g1_MZ, g2_MZ, g3_MZ], t_eval=t_eval_SM, method='RK45')
alpha_inv_SM = (4 * np.pi) / (sol_SM.y**2)

# 2. Obliczenie dla MSSM (Najpierw SM do 1 TeV, potem przejście na współczynniki SUSY)
# Etap 1: Od M_Z do M_SUSY
t_eval_MSSM_1 = np.linspace(t_MZ, t_SUSY, 50)
sol_MSSM_1 = solve_ivp(get_derivatives(b_SM, B_SM), [t_MZ, t_SUSY], [g1_MZ, g2_MZ, g3_MZ], t_eval=t_eval_MSSM_1, method='RK45')

# Pobranie wartości g na skali 1 TeV
g_SUSY_start = sol_MSSM_1.y[:, -1]

# Etap 2: Od M_SUSY do M_GUT (używając współczynników MSSM)
t_eval_MSSM_2 = np.linspace(t_SUSY, t_GUT, 150)
sol_MSSM_2 = solve_ivp(get_derivatives(b_MSSM, B_MSSM), [t_SUSY, t_GUT], g_SUSY_start, t_eval=t_eval_MSSM_2, method='RK45')

# Łączenie tablic
t_eval_MSSM = np.concatenate((sol_MSSM_1.t, sol_MSSM_2.t))
g_MSSM = np.concatenate((sol_MSSM_1.y, sol_MSSM_2.y), axis=1)
alpha_inv_MSSM = (4 * np.pi) / (g_MSSM**2)

# RYSOWANIE WYKRESU
plt.figure(figsize=(11, 7))

# Rysowanie tła Modelu Standardowego (linie przerywane)
log10_mu_SM = np.log10(np.exp(t_eval_SM))
plt.plot(log10_mu_SM, alpha_inv_SM[0], linestyle='--', color='#88aaff', label='SM: U(1)')
plt.plot(log10_mu_SM, alpha_inv_SM[1], linestyle='--', color='#ff88aa', label='SM: SU(2)')
plt.plot(log10_mu_SM, alpha_inv_SM[2], linestyle='--', color='#88ddaa', label='SM: SU(3)')

# Rysowanie linii Supersymetrii MSSM (linie ciągłe)
log10_mu_MSSM = np.log10(np.exp(t_eval_MSSM))
plt.plot(log10_mu_MSSM, alpha_inv_MSSM[0], linestyle='-', color='#0055ff', linewidth=2.5, label='SUSY (MSSM): U(1)')
plt.plot(log10_mu_MSSM, alpha_inv_MSSM[1], linestyle='-', color='#ff0055', linewidth=2.5, label='SUSY (MSSM): SU(2)')
plt.plot(log10_mu_MSSM, alpha_inv_MSSM[2], linestyle='-', color='#00aa55', linewidth=2.5, label='SUSY (MSSM): SU(3)')

# Oznaczenie M_SUSY
plt.axvline(np.log10(1000), color='gray', linestyle=':', label='Skala wzbudzenia SUSY (1 TeV)')

plt.title('Unifikacja sprzężeń: Model Standardowy vs Supersymetria (MSSM)', fontsize=15)
plt.xlabel(r'$\log_{10}(\mu)$ [Energia w GeV]', fontsize=13)
plt.ylabel(r'$\alpha_i^{-1}(\mu)$', fontsize=13)
plt.legend(fontsize=10, loc='lower right', ncol=2)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()

# Zapis pliku
plt.savefig('/home/user/rge_susy_plot.png', dpi=300)
print("Wykres wygenerowany pomyślnie. Zapisano jako rge_susy_plot.png")
