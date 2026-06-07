import numpy as np
import matplotlib.pyplot as plt

# ====================================================================
# SYMULACJA: ŁAMANIE ZAPACHU LEPTONÓW (LFV) - PROCES mu -> e gamma
# ====================================================================
# W modelu Pati-Salama / SHZ-BCC lepton jest 4. kolorem.
# Mieszanie kwarków (CKM) naturalnie indukuje mieszanie leptonów (PMNS),
# co prowadzi do rzadkich rozpadów mionu na elektron i foton (mu -> e + gamma).
# Tempo tego rozpadu silnie zależy od masy bozonu X (M_BCC) oraz mas sterylnych neutrin.

# Stałe
m_mu = 105.658e-3 # GeV (Masa mionu)
alpha_em = 1.0 / 137.0 # Stała struktury subtelnej
G_F = 1.166e-5 # Stała Fermiego (GeV^-2)

# Przeskanujemy przestrzeń mas ciężkich neutrin (Right-Handed Neutrinos)
# To one w pętlach kwantowych napędzają ten rozpad
M_R_scale = np.logspace(3, 14, 200) # Masa ciężkiego neutrina M_R w GeV

# Formuła przybliżona na Branching Ratio (BR) w modelach Left-Right Symmetric / Pati-Salam
# BR(mu -> e gamma) ~ (3 * alpha_em / (32 * pi)) * |Sum_i (V_mu_i * V_e_i * f(M_R_i/M_W))|^2
# W uproszczeniu dla dominującej wymiany na skali M_R:
# BR jest proporcjonalne do (m_nu_Dirac / M_R)^4, ale tu wchodzą też mieszania.
# Dla czytelności symulujemy heurystyczną relację typową dla modeli z masą M_R i kątami CKM-like

# Oczekiwane mieszanie (zakładamy powiązanie z kątem Cabibbo w SHZ-BCC, theta ~ lambda_BCC^3)
mix_angle_term = (0.22508**3)**2 

# Przybliżone Branching Ratio jako funkcja skali M_R
BR_mu_e_gamma = 1e-6 * mix_angle_term * (M_R_scale / 1e14)**2
# W modelach supersymetrycznych / SU(4) często funkcja rośnie ze skalą lub zależy od mas s-leptonów. 
# Zastosujemy tu wariant, w którym ciężkie neutrina indukują wyższy BR.

# --- LIMITY EKSPERYMENTALNE ---
# Obecny limit z eksperymentu MEG (Paul Scherrer Institute, PSI)
meg_limit_current = 4.2e-13
# Planowany limit ulepszonego detektora MEG II (zbiera dane)
meg2_limit_future = 6e-14
# Limit z nadchodzącego eksperymentu Mu3e (Fermilab) i COMET
mu2e_limit = 1e-16 # (Konwersja mionu, ale silnie skorelowana z mu->e gamma)

# ================= RYSOWANIE WYKRESU =================

plt.figure(figsize=(10, 6))

plt.plot(M_R_scale, BR_mu_e_gamma, 'b-', lw=3, label=r'Przewidywanie modelu SHZ-BCC (Wkład $\nu_R$)')

# Wypełnienie wykluczonej strefy (Obecny stan)
plt.axhline(meg_limit_current, color='red', linestyle='--', lw=2, label='Wykluczone przez MEG (2016)')
plt.fill_between(M_R_scale, meg_limit_current, 1e-9, color='red', alpha=0.1)

# Przyszłe eksperymenty (Zasięg odkrycia)
plt.axhline(meg2_limit_future, color='green', linestyle='--', lw=2, label='Czułość MEG II (W toku)')
plt.fill_between(M_R_scale, meg2_limit_future, meg_limit_current, color='green', alpha=0.1)

plt.axhline(mu2e_limit, color='purple', linestyle=':', lw=2, label='Zasięg Mu2e / COMET (Przyszłość)')

# Skale masy z pracy
plt.axvline(1e10, color='gray', linestyle=':', lw=2, label=r'Skala Leptogenezy $M_2 \sim 10^{10}$ GeV')
plt.axvline(1e14, color='black', linestyle=':', lw=2, label=r'Skala Seesaw $M_3 \sim 10^{14}$ GeV')

plt.xscale('log')
plt.yscale('log')
plt.xlim(1e8, 1e15)
plt.ylim(1e-17, 1e-10)

plt.title(r'Łamanie Zapachu Leptonów (LFV): Rozpad $\mu \to e \gamma$', fontsize=14)
plt.xlabel(r'Masa Prawoskrętnego Neutrina $M_R$ [GeV]', fontsize=12)
plt.ylabel(r'Współczynnik rozgałęzienia $\mathcal{B}(\mu \to e \gamma)$', fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(loc='lower right', fontsize=10)
plt.tight_layout()

plt.savefig('/home/user/lfv_muon_decay.png', dpi=300)
print("Wykres LFV wygenerowany jako lfv_muon_decay.png")
