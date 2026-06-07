import numpy as np
import matplotlib.pyplot as plt

# --- 1. DANE Z MODELU SHZ-BCC ---
# Zgodnie z sekcją 7.3 i 7.4 pracy:
lam_bcc = 1.0 / (np.pi * np.sqrt(2))  # ok. 0.225079
s12 = lam_bcc
s23 = 0.04136
s13 = 0.00329
delta_rad = np.arccos(1.0 / 3.0)  # ok. 70.53 stopnia

# Obliczenie współrzędnych wierzchołka Trójkąta Unitarności (rho_bar, eta_bar)
# W standardowej parametryzacji Wolfensteina: R_u = |V_ub / (lam * V_cb)| = s13 / (s12 * s23)
Ru = s13 / (s12 * s23)
rho_bcc = Ru * np.cos(delta_rad)
eta_bcc = Ru * np.sin(delta_rad)

# --- 2. DANE EKSPERYMENTALNE (Particle Data Group - PDG 2024) ---
# Zmierzone parametry wierzchołka trójkąta z rozpadów mezonów B
rho_pdg = 0.141
rho_err = 0.017
eta_pdg = 0.340
eta_err = 0.011

# --- 3. RYSOWANIE WYKRESU ---
plt.figure(figsize=(10, 6))

# Rysowanie bazowej linii (0,0) do (1,0)
plt.plot([0, 1], [0, 0], 'k-', lw=2)

# Rysowanie trójkąta PDG (eksperyment)
plt.plot([0, rho_pdg], [0, eta_pdg], color='gray', linestyle='--', alpha=0.7)
plt.plot([1, rho_pdg], [0, eta_pdg], color='gray', linestyle='--', alpha=0.7)
# Zaznaczenie obszaru błędu PDG
plt.errorbar(rho_pdg, eta_pdg, xerr=rho_err, yerr=eta_err, fmt='o', color='gray', capsize=5, label='Eksperyment (LHCb/Belle) ±1σ')

# Rysowanie trójkąta SHZ-BCC (teoria)
plt.plot([0, rho_bcc], [0, eta_bcc], color='#ff0055', lw=2)
plt.plot([1, rho_bcc], [0, eta_bcc], color='#ff0055', lw=2)
plt.plot(rho_bcc, eta_bcc, 'o', color='#ff0055', markersize=8, label='Teoria SHZ-BCC (Czysta matematyka)')

# Upiększanie
plt.xlim(-0.1, 1.1)
plt.ylim(-0.05, 0.5)
plt.title('Trójkąt Unitarności CKM: Model SHZ-BCC vs Eksperyment', fontsize=14)
plt.xlabel(r'$\bar{\rho}$', fontsize=14)
plt.ylabel(r'$\bar{\eta}$', fontsize=14)
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)

# Dodanie adnotacji tekstowych
plt.text(rho_bcc - 0.02, eta_bcc + 0.02, f'({rho_bcc:.3f}, {eta_bcc:.3f})', color='#ff0055', fontweight='bold')
plt.text(rho_pdg + 0.02, eta_pdg - 0.03, f'({rho_pdg:.3f}, {eta_pdg:.3f})', color='gray', fontweight='bold')

plt.tight_layout()
plt.savefig('/home/user/ckm_triangle.png', dpi=300)
print(f"Teoretyczny wierzchołek SHZ-BCC: (rho = {rho_bcc:.3f}, eta = {eta_bcc:.3f})")
print(f"Eksperymentalny wierzchołek PDG: (rho = {rho_pdg:.3f}, eta = {eta_pdg:.3f})")
print("Wykres zapisany jako ckm_triangle.png")
