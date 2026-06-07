import numpy as np
import matplotlib.pyplot as plt

# --- 1. Widmo Fal Grawitacyjnych z Kosmicznych Strun ---
# Gdy symetria Pati-Salama łamie się na skali M_BCC = 5 * 10^15 GeV,
# we Wszechświecie powstają defekty topologiczne - tzw. Kosmiczne Struny.

M_BCC = 5e15 # GeV
M_Planck = 1.22e19 # GeV
G_mu = (M_BCC / M_Planck)**2 # Napięcie struny (G*mu) - ok. 1.6 * 10^-7

# Częstotliwości fal grawitacyjnych (od nanohertzów do kiloherców)
f = np.logspace(-10, 4, 200)

# Uproszczone widmo (Omega_GW h^2) dla strun kosmicznych w erze radiacyjnej to w przybliżeniu płaskie plateau
# Zależy ono od napięcia struny G_mu. 
# Omega_GW ~ 10^-4 * (G_mu)^(1/2) (bardzo zgrubne oszacowanie dla plateau)
omega_gw_plateau = 1e-4 * np.sqrt(G_mu) * np.ones_like(f)

# Czułości detektorów (przybliżone krzywe)
f_nanograv = np.array([1e-9, 1e-8])
sens_nanograv = np.array([1e-10, 1e-9]) # Zbyt mocne tło jest wykluczone

f_lisa = np.logspace(-4, -1, 50)
sens_lisa = 1e-13 * (f_lisa/1e-2)**-2 + 1e-13 * (f_lisa/1e-2)**2

plt.figure(figsize=(10, 6))
plt.loglog(f, omega_gw_plateau, 'r-', lw=3, label=rf'SHZ-BCC Struny ($M_{{BCC}} = 5 \times 10^{{15}}$ GeV)')

# Zakresy wykluczeń i detektorów
plt.fill_between(f_nanograv, sens_nanograv, 1, color='gray', alpha=0.3, label='Wykluczenie NANOGrav (PTA)')
plt.fill_between(f_lisa, sens_lisa, 1, color='blue', alpha=0.1, label='Zasięg LISA (planowany)')

plt.axvline(1e-8, color='k', linestyle=':')
plt.text(2e-8, 1e-6, '<-- NANOGrav widziałby to dzisiaj!', fontsize=11, color='red', fontweight='bold')

plt.xlim(1e-10, 1e4)
plt.ylim(1e-15, 1e-4)
plt.xlabel('Częstotliwość $f$ [Hz]', fontsize=13)
plt.ylabel(r'Gęstość energii fal grawitacyjnych $\Omega_{GW} h^2$', fontsize=13)
plt.title('Fale Grawitacyjne w modelu SHZ-BCC (Kosmiczne Struny)', fontsize=15)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(loc='lower left', fontsize=11)
plt.tight_layout()
plt.savefig('/home/user/gw_strings_plot.png', dpi=300)
print(f"Napięcie struny G*mu wyliczone dla M_BCC: {G_mu:.2e}")
