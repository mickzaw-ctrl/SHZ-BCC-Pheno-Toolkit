import numpy as np
import matplotlib.pyplot as plt

t_start = np.log(5e15)
t_end = np.log(91.18)
t_vals = np.linspace(t_start, t_end, 100)
mu_vals = np.log10(np.exp(t_vals))

# Wartość początkowa SHZ-BCC
s13_0 = 0.00329
# Eksperyment
s13_exp = 0.00369

# 1. Zwykły Model Standardowy (szybki dryf w górę)
# Symulacja RGE bez ochrony
yt_sq = 0.5 + 0.5 * ((t_vals - t_end)/(t_start - t_end))
s13_SM = s13_0 * np.exp(np.cumsum(- (3.0 / (16.0 * np.pi**2)) * yt_sq * 1.5 * (t_vals[1]-t_vals[0])))

# 2. Mechanizm Ochrony Zapachu (np. specificzne 2HDM z symetrią Peccei-Quinn)
# Sprzężenia kwarku top ulegają kasowaniu (dt/d_theta ~ 0)
s13_Shielded = s13_0 * np.ones_like(t_vals)

plt.figure(figsize=(10, 6))
plt.plot(mu_vals, s13_SM, 'r--', lw=3, label='Standardowe RGE (Niszczy przewidywania)')
plt.plot(mu_vals, s13_Shielded, 'g-', lw=3, label='RGE z "Tarczą Zapachową" (Zachowuje V_BCC)')

plt.plot(np.log10(5e15), s13_0, 'go', markersize=10, label=r'Baza $M_{BCC}$')
plt.plot(np.log10(91.18), s13_exp, 'k*', markersize=12, label='Eksperyment (PDG)')

plt.xlim(16, 1)
plt.title(r'Wpływ "Tarczy Zapachowej" na element $|V_{ub}|$ CKM', fontsize=14)
plt.xlabel(r'$\log_{10}(E / \text{GeV})$ [Spadek Energii]', fontsize=12)
plt.ylabel(r'Wartość elementu $|V_{ub}|$', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=12, loc='upper left')
plt.tight_layout()
plt.savefig('/home/user/ckm_shield_plot.png', dpi=300)
