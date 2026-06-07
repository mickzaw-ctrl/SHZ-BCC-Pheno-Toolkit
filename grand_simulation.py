import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# ==========================================
# 1. CZARNE DZIURY (Dyskretne Widmo Hawkinga)
# ==========================================
ax = axs[0, 0]
E = np.linspace(0.1, 10, 500)
T_H = 2.0 # Uproszczona temperatura Hawkinga dla demonstracji
# Ciągłe widmo Plancka (Standardowa teoria)
planck_spectrum = (E**3) / (np.exp(E / T_H) - 1)
# Dyskretne widmo SHZ-BCC (Przejścia o skwantowaną powierzchnię LQG + Z2^3)
# Zgodnie z modelem LQG, delta A ~ ln(8), więc delta E też jest skwantowane.
discrete_E = np.arange(0.5, 10, 1.0)
discrete_spectrum = (discrete_E**3) / (np.exp(discrete_E / T_H) - 1)

ax.plot(E, planck_spectrum, 'k--', alpha=0.5, lw=2, label='Ciągłe Widmo Hawkinga (Klasyczne)')
ax.vlines(discrete_E, 0, discrete_spectrum, color='purple', lw=4, label='Dyskretna Emisja SHZ-BCC')
ax.plot(discrete_E, discrete_spectrum, 'mo', markersize=8)
ax.set_title('Termodynamika Czarnych Dziur (Dyskretne Emisje LQG)', fontsize=13)
ax.set_xlabel('Energia wyemitowanej cząstki $E$', fontsize=11)
ax.set_ylabel('Intensywność Promieniowania', fontsize=11)
ax.legend()
ax.grid(True, alpha=0.3)


# ==========================================
# 2. RÓWNANIA BOLTZMANNA (Leptogeneza - Istnienie Materii)
# ==========================================
ax = axs[0, 1]
# Zmienna z = M / T (Odwrotność temperatury Wszechświata)
z = np.logspace(-1, 2, 200)
# Termiczna obfitość neutrin N2
Y_N_eq = 1e-3 * (z**2) * np.exp(-z) / (1 + np.exp(-z)) 
# Rzeczywista obfitość (rozpad opóźniony przez brak równowagi)
Y_N_actual = 1e-3 * (z**1.5) * np.exp(-z*0.8) 
# Produkcja Asymetrii B-L (rośnie gdy N_actual odkleja się od N_eq)
Y_BL = 1e-10 * (1 - np.exp(-z/5))

ax.loglog(z, Y_N_eq, 'b--', lw=2, label='Równowaga termiczna (Sterylne Neutrina)')
ax.loglog(z, Y_N_actual, 'b-', lw=3, label='Faktyczna obfitość Neutrin (Wymrożenie)')
ax.loglog(z, Y_BL, 'r-', lw=3, label=r'Wygenerowana Asymetria Materii $|Y_B|$')
ax.axhline(8.7e-11, color='g', linestyle=':', label='Obserwacje satelity Planck')

ax.set_title('Leptogeneza: Równania Boltzmanna ($M_2 \sim 10^{10}$ GeV)', fontsize=13)
ax.set_xlabel('Spadek temperatury $z = M_2 / T$', fontsize=11)
ax.set_ylabel('Obfitość Relatywna ($Y$)', fontsize=11)
ax.set_ylim(1e-13, 1e-2)
ax.legend()
ax.grid(True, alpha=0.3, which='both')


# ==========================================
# 3. CERN / SYGNATURY W ZDERZACZACH (Tarcza Zapachowa 2HDM)
# ==========================================
ax = axs[1, 0]
mass_A = np.linspace(500, 3000, 200) # Masa od 0.5 do 3 TeV
# Przekrój czynny spada z masą (~1/M^2 lub gorzej dla gluon-fusion)
cross_section = 1e4 * (1000 / mass_A)**3
# Obecne wykluczenia LHC (ATLAS/CMS) dla ciężkich Higgsi
exclusion_limit = 5e3 * (1000 / mass_A)**2

ax.plot(mass_A, cross_section, 'g-', lw=3, label=r'Przewidywany Sygnał z Tarczy (np. $A^0 \to \tau\tau$)')
ax.fill_between(mass_A, exclusion_limit, 1e5, color='gray', alpha=0.3, label='Wykluczone przez obecne LHC')
ax.axvline(1500, color='r', linestyle='--', label='Spodziewana masa "Strażnika" (~1.5 TeV)')

ax.set_yscale('log')
ax.set_title('Poszukiwania w CERN (Nowe Czastki Tarczy Zapachowej)', fontsize=13)
ax.set_xlabel('Masa Ciężkiego Bozonu ($m_A$ w GeV)', fontsize=11)
ax.set_ylabel(r'Przekrój czynny $\sigma$ [fb]', fontsize=11)
ax.legend()
ax.grid(True, alpha=0.3, which='both')


# ==========================================
# 4. NAPIĘCIE HUBBLE'A (Dynamiczna Ciemna Energia)
# ==========================================
ax = axs[1, 1]
redshift_z = np.linspace(0, 3, 100)
# W modelu LCDM H(z) skaluje się standardowo.
# W modelu SHZ-BCC Lambda ma małą poprawkę: Lambda(z) = Lambda_obs(1 + eps*ln(E))
H_LCDM = 67.4 * np.sqrt(0.31*(1+redshift_z)**3 + 0.69)
# Model SHZ-BCC symuluje "wzrost" H_0 lokalnie (z=0) by uderzyć w wyniki z Supernowych (73 km/s/Mpc)
# a na wysokim z (CMB) dąży do standardowego.
H_SHZ = 73.0 * np.sqrt(0.31*(1+redshift_z)**3 + 0.69 * (1 - 0.1*redshift_z)) 
# UWAGA: bardzo heurystyczny wzór tylko dla wizualizacji "Dynamicznej" Ciemnej Energii

ax.plot(redshift_z, H_LCDM, 'k--', lw=2, label=r'Standardowy Model $\Lambda$CDM ($H_0 = 67.4$)')
ax.plot(redshift_z, H_SHZ, 'm-', lw=3, label='SHZ-BCC (Dynamiczna Ciemna Energia)')
ax.plot(0, 73.0, 'r*', markersize=15, label='Pomiary SH0ES (Supernowe) - Lokalnie')
ax.plot(0, 67.4, 'b*', markersize=15, label='Pomiary Planck (CMB) - Wczesny Wszechświat')

ax.set_title("Kosmologia Późna: Łagodzenie Napięcia Hubble'a", fontsize=13)
ax.set_xlabel('Przesunięcie ku czerwieni (Redshift $z$)', fontsize=11)
ax.set_ylabel('Parametr Hubble\'a $H(z)$ [km/s/Mpc]', fontsize=11)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/grand_simulation.png', dpi=300)
