import numpy as np
import matplotlib.pyplot as plt

# --- Parametry modelu ---
M_Pl = 1.22e19 # Masa Plancka [GeV]
M_BCC = 5e15   # Skala łamania Pati-Salama (Oczekiwana wartość pola VEV) [GeV]
lambda_coupling = 1e-4 # Stała sprzężenia dla potencjału

# 1. POTENCJAŁ POLA SKALARNEGO (Fałszywa Próżnia)
# V(phi) = lambda/4 * (phi^2 - v^2)^2
phi = np.linspace(0, 1.5 * M_BCC, 500)
V_phi = (lambda_coupling / 4.0) * (phi**2 - M_BCC**2)**2

# 2. EWOLUCJA SKALI WSZECHŚWIATA I ROZRZEDZANIE STRUN
# Podczas gdy pole utknęło w phi=0 (fałszywa próżnia), energia V_0 napędza inflację
V_0 = (lambda_coupling / 4.0) * M_BCC**4
H_inf = np.sqrt(V_0 / (3 * M_Pl**2)) # Parametr Hubble'a podczas inflacji

# Czas trwania inflacji (w jednostkach 1/H_inf)
t = np.linspace(0, 70, 500) 
a_t = np.exp(t) # Skalowanie wszechświata (Czynnik skali) a(t) ~ e^(H*t)

# Gęstość strun kosmicznych (skaluje się jako 1/a^2)
string_density = 1.0 / (a_t**2)

# --- TWORZENIE WYKRESÓW ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Wykres 1: Potencjał Inflatonu (Meksykański Kapelusz)
ax1.plot(phi / M_BCC, V_phi, 'b-', lw=3)
ax1.axvline(1.0, color='g', linestyle='--', label=r'Prawdziwa Próżnia ($M_{BCC}$)')
ax1.plot(0, V_0, 'ro', markersize=10, label='Fałszywa Próżnia (Start Inflacji)')
ax1.annotate('Kwantowe\nstoczenie', xy=(0.5, 0.5*V_0), xytext=(0.2, 0.2*V_0),
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=11)
ax1.set_title(r'Potencjał Łamania Symetrii Pati-Salama $V(\phi)$', fontsize=14)
ax1.set_xlabel(r'Pole skalarne $\phi / M_{BCC}$', fontsize=12)
ax1.set_ylabel(r'Energia Potencjalna $V(\phi)$', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.5)

# Wykres 2: Rozrzedzanie Strun (E-folds)
ax2.plot(t, string_density, 'r-', lw=3, label='Gęstość Kosmicznych Strun')
ax2.set_yscale('log')
ax2.axvline(60, color='k', linestyle=':', label='Wymagane 60 e-składzeń (e-folds)')
ax2.set_title('Zabójca Strun: Rozszerzanie Inflacyjne', fontsize=14)
ax2.set_xlabel(r'Liczba e-składzeń $N_e = H_{inf} \times t$', fontsize=12)
ax2.set_ylabel(r'Względna gęstość strun $(\rho / \rho_0)$', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.5, which='both')

plt.tight_layout()
plt.savefig('/home/user/inflation_dynamics.png', dpi=300)
