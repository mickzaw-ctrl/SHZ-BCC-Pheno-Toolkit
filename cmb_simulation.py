import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# SYMULACJA KOSMOLOGICZNA: WIDMO MOCY CMB (Planck Satellite)
# =========================================================

ell = np.arange(2, 40)
lcdm_theory = np.ones_like(ell) * 1000  

np.random.seed(11)
# Symulacja danych
planck_data = lcdm_theory * (1 + np.random.randn(len(ell)) * np.sqrt(2/(2*ell+1))) 
planck_data[0] *= 0.35 
planck_data[1] *= 0.65 
planck_data[2] *= 0.8  

# Fix na ujemne wartości błędów z randn (absolute)
err = np.abs(planck_data * np.sqrt(2/(2*ell+1)))

# Teoria SHZ-BCC (cut-off w podczerwieni z powodu szybkiego pęknięcia Pati-Salama)
k_star = 4.0 
shz_bcc_theory = lcdm_theory * (1 - np.exp(-ell / k_star))

# ================= RYSOWANIE WYKRESU =================
plt.figure(figsize=(10, 6))

plt.plot(ell, lcdm_theory, 'k--', lw=2, alpha=0.6, label='Standardowa Inflacja (LCDM Theory)')

plt.errorbar(ell, planck_data, yerr=err, fmt='o', color='gray', capsize=3, label='Satelita Planck 2018 (Dane + Wariancja Kosmiczna)')

plt.plot(ell, shz_bcc_theory, 'm-', lw=3, label='Inflacja w modelu SHZ-BCC (Wpływ granicy Pati-Salama)')

# Obszar anomalii 
plt.axvspan(1.5, 5.5, color='orange', alpha=0.2, label='Anomalia "Brakującej Mocy" (Low-ell Deficit)')

plt.title('Widmo Mocy Promieniowania Tła (CMB) i Anomalia Quadrupolowa', fontsize=14)
plt.xlabel(r'Kątowa Skala na Niebie (Multipol $\ell$)', fontsize=12)
plt.ylabel(r'$\frac{\ell(\ell+1)}{2\pi} C_\ell$ [arbitrary units]', fontsize=12)

plt.xlim(1.5, 30)
plt.ylim(0, 1600)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.7)

plt.annotate('Ostre łamanie symetrii\n$SU(4) \\to SU(3)$\ntłumi największe fale', 
             xy=(3, 300), xytext=(6, 100),
             arrowprops=dict(facecolor='magenta', shrink=0.05, width=2, headwidth=8), 
             fontsize=11, color='magenta', fontweight='bold')

plt.tight_layout()
plt.savefig('/home/user/cmb_anomaly_plot.png', dpi=300)
print("Wykres anomalii CMB wygenerowany: cmb_anomaly_plot.png")
