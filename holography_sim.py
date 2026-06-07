import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# SYMULACJA HOLOGRAFICZNA: ENTROPIA SPLĄTANIA W MODELU SHZ-BCC
# =========================================================
# W kwantowej teorii informacji i grawitacji (np. MERA, AdS/CFT),
# objętość (bulk) jest opisywana przez sieć tensorową, a granica
# (boundary/horyzont) to zredukowany wektor stanu, który powstaje
# po wyśledzeniu (tracing out) stopni swobody po jednej stronie horyzontu.

# Zdefiniujmy 8 kanałów bazowych na złączu Horyzontu (Z_2^3)
# To jest C^8 Hilbert space z pracy.

# Operacja BCC (Boundary Consistency Condition) zmusza ten stan
# do stania się uśrednionym stanem Singletowym (tzw. stan GHZ-podobny dla 3 qubitów):
# |u> = 1/sqrt(8) * (|000> + |001> + |010> + ... + |111>)

# Entropia Von Neumanna (S_VN = -Tr(rho * ln(rho)))
# symuluje jak bardzo jedna strona horyzontu jest "nieświadoma"
# o tym, co dzieje się po drugiej stronie, co w holografii bezpośrednio
# tworzy pole powierzchni (Area Law) - czyli stałą kosmologiczną.

N_channels = 8

# Tworzymy wektor stanu idealnego singletu z modelu SHZ-BCC
state_u = np.ones(N_channels) / np.sqrt(N_channels)

# Symulacja dekoherencji / fluktuacji kwantowych fałszywej próżni.
# Parametr p = prawdopodobieństwo zdepolaryzowania stanu (szum kanału).
p_values = np.linspace(0, 1, 100)

# Dla p=0: Czysty uśredniony stan SHZ-BCC (Wyzerowanie Lambda)
# Dla p=1: Stan kompletnie wymieszany (Maksymalna energia próżni, "Brak BCC")

von_neumann_entropy = []
residual_lambda = []

for p in p_values:
    # Depolarizing channel: rho = (1-p) * |u><u| + p * (I / 8)
    rho_pure = np.outer(state_u, state_u)
    rho_mixed = np.eye(N_channels) / N_channels
    
    rho = (1 - p) * rho_pure + p * rho_mixed
    
    # Obliczanie wartości własnych macierzy gęstości
    eigenvalues = np.linalg.eigvalsh(rho)
    # Zabezpieczenie przed log(0)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    # Entropia Von Neumanna układu na horyzoncie (baza dla Area Law)
    S_VN = -np.sum(eigenvalues * np.log2(eigenvalues))
    von_neumann_entropy.append(S_VN)
    
    # Resztkowa "energia próżni" uciekająca przed zgaszeniem
    # (proporcjonalna do odchylenia od stanu u)
    # Gdy p=0 (ścisłe uśrednienie Z2^3), Lambda = 0
    # Gdy p=1 (brak symetrii BCC), Lambda = Lambda_bare (w kosmos)
    Lambda_eff = (1 - eigenvalues[0]) * 10**120  # Logarytmicznie przeskalowana do wykresu
    if Lambda_eff <= 0: Lambda_eff = 1e-50 # Zabezpieczenie log(0)
    residual_lambda.append(Lambda_eff)

von_neumann_entropy = np.array(von_neumann_entropy)
residual_lambda = np.array(residual_lambda)

# ================= RYSOWANIE WYKRESU =================

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Parametr Zaburzenia Złącza $p$ (0 = Perfekcyjna Symetria SHZ-BCC, 1 = Brak)', fontsize=11)
ax1.set_ylabel('Entropia Von Neumanna (Splątanie Holograficzne)', color=color, fontsize=12)
ax1.plot(p_values, von_neumann_entropy, color=color, lw=3, label='Entropia Horyzontu (Bity kwantowe)')
ax1.tick_params(axis='y', labelcolor=color)

# Punkt idealny
ax1.plot(0, 0, 'b*', markersize=12, label='Idealny Projektor SHZ-BCC ($S=0$)')
# Granica max
ax1.plot(1, 3, 'bo', markersize=8, label=r'Max entropia dla 8 kanałów ($\log_2(8)=3$)')
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend(loc='upper left', fontsize=10)

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Resztkowa Skala Energii Próżni $\Lambda_{eff}$ (rząd wielkości)', color=color, fontsize=12)  
# Używamy logarytmicznej osi, żeby pokazać zjazd ze 120 rzędów w dół
ax2.plot(p_values, np.log10(residual_lambda + 1e-10), color=color, linestyle='--', lw=3, label=r'Stała Kosmologiczna $\Lambda$')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(-10, 130)
ax2.legend(loc='upper right', fontsize=10)

# Oznaczenie punktu widzialnego wszechświata (Dark Energy ~ 10^-120, stąd blisko zera)
ax2.plot(0.0001, np.log10(1), 'r*', markersize=15, label=r'Obserwowana $\Lambda_{obs}$ ($p \approx 10^{-120}$)')

plt.title(r'Holograficzna Sieć Tensorowa: Związek Entropii z Grawitacją', fontsize=14)
fig.tight_layout()  
plt.savefig('/home/user/holography_entanglement.png', dpi=300)
print("Wykres holograficznego splątania: holography_entanglement.png")
