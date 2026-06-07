import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# SYMULACJA MONTE CARLO: KINEMATYKA ZDERZEŃ LHC (gg -> A0 -> tau tau)
# =========================================================
# Symulujemy zderzenie dwóch protonów (fuzja gluonów) przy sqrt(s) = 14 TeV
# (High-Luminosity LHC). Nowa cząstka "Strażnik Zapachu" (np. pseudo-skalar A^0
# z modelu 2HDM) ma masę ~1.5 TeV. Cząstka ta najchętniej rozpada się na 
# fermiony trzeciej generacji, np. parę taonów (tau+ tau-).

np.random.seed(42)

# 1. PARAMETRY SYMULACJI
mass_A = 1500.0 # Masa nowej cząstki [GeV]
width_A = 45.0  # Szerokość rozpadu (Szerokość rezonansu) [GeV]
n_bkg = 50000   # Liczba zdarzeń tła (Standard Model Drell-Yan Z/gamma* -> tau tau)
n_sig = 1200    # Liczba zdarzeń sygnału (Nasza nowa fizyka SHZ-BCC Shield)

# 2. GENEROWANIE TŁA (Tło maleje wykładniczo z masą)
# Używamy prostego rozkładu potęgowo-wykładniczego typowego dla tła Drell-Yan
mass_bkg = np.random.exponential(scale=300, size=n_bkg) + 200 # Przesunięcie
mass_bkg = mass_bkg[(mass_bkg > 400) & (mass_bkg < 3000)] # Zakres detektora

# 3. GENEROWANIE SYGNAŁU (Rezonans Breit-Wigner rozmyty przez rozdzielczość detektora)
# Prawdziwy rozkład Breit-Wignera
mass_sig_true = np.random.standard_cauchy(n_sig) * (width_A/2.0) + mass_A
# Rozdzielczość detektora CMS/ATLAS dla taonów (ok. 10-15% przy tej masie)
detector_resolution = 0.12 * mass_A
mass_sig_reco = mass_sig_true + np.random.normal(0, detector_resolution, n_sig)
mass_sig_reco = mass_sig_reco[(mass_sig_reco > 400) & (mass_sig_reco < 3000)]

# 4. ŁĄCZENIE DANYCH (PSEUDO-DATA)
data_total = np.concatenate([mass_bkg, mass_sig_reco])

# ================= RYSOWANIE WYKRESU (Styl ATLAS/CMS) =================
plt.figure(figsize=(10, 7))

# Definicja binów (przedziałów histogramu)
bins = np.linspace(500, 2500, 40)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
bin_width = bins[1] - bins[0]

# Tworzenie histogramów
counts_bkg, _ = np.histogram(mass_bkg, bins=bins)
counts_sig, _ = np.histogram(mass_sig_reco, bins=bins)
counts_data, _ = np.histogram(data_total, bins=bins)

# Błędy statystyczne (Poisson)
err_data = np.sqrt(counts_data)

# Rysowanie złączonego histogramu (Stack)
plt.bar(bin_centers, counts_bkg, width=bin_width, color='#33ccff', edgecolor='black', alpha=0.8, label='Tło Modelu Standardowego (Drell-Yan)')
plt.bar(bin_centers, counts_sig, width=bin_width, bottom=counts_bkg, color='#ff3333', edgecolor='black', alpha=0.9, label=r'Sygnał SHZ-BCC ($A^0 \to \tau^+\tau^-$)')

# Rysowanie punktów "Danych Eksperymentalnych"
plt.errorbar(bin_centers, counts_data, yerr=err_data, fmt='ko', markersize=5, capsize=3, label='Pseudo-dane (High-Luminosity LHC)')

# Upiększanie w stylu publikacji CERN
plt.yscale('log')
plt.xlim(500, 2500)
plt.ylim(0.5, 1e4)

plt.xlabel(r'Masa Niezmiennicza pary taonów $m_{\tau\tau}$ [GeV]', fontsize=13)
plt.ylabel(r'Liczba zdarzeń / 50 GeV', fontsize=13)
plt.title(r'Wizualizacja Odkrycia w CERN: Rezonans "Strażnika Zapachu"', fontsize=15)

# Napisy jak na wykresach ATLAS/CMS
plt.text(550, 3e3, r'\textbf{CMS/ATLAS} Symulacja Monte Carlo', fontsize=12)
plt.text(550, 1.5e3, r'$\sqrt{s} = 14$ TeV, $L_{int} = 3000$ fb$^{-1}$', fontsize=11)
plt.text(1500, 1e2, r'Pik odkrycia na 1.5 TeV!', fontsize=12, color='#cc0000', fontweight='bold', ha='center')

plt.legend(loc='upper right', fontsize=11, framealpha=0.9)
plt.grid(True, which='both', linestyle=':', alpha=0.4)

plt.tight_layout()
plt.savefig('/home/user/lhc_collision_kinematics.png', dpi=300)
print("Wykres symulacji zderzeń (Monte Carlo) wygenerowany: lhc_collision_kinematics.png")
