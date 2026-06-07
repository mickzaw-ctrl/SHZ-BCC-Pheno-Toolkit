import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- 1. PARAMETRY WEJŚCIOWE Z MODELU SHZ-BCC (Skala M_BCC = 10^16 GeV) ---
lam_bcc = 1.0 / (np.pi * np.sqrt(2))  # ~ 0.225079
s12_0 = lam_bcc
s23_0 = np.sqrt(2.0/3.0) * (lam_bcc**2)
s13_0 = np.sqrt(2.0/3.0) * (1.0/np.sqrt(8.0)) * (lam_bcc**3)
delta_0 = np.arccos(1.0 / 3.0) # Faza w radianach

# Eksperymentalne wartości (PDG) na skali M_Z
s12_exp = 0.22500
s23_exp = 0.04182
s13_exp = 0.00369
delta_exp = 1.144 # ok 65-70 stopni w radianach wg PDG

# --- 2. RÓWNANIA GRUPY RENORMALIZACJI (RGE) DLA KĄTÓW CKM ---
# Zmienna t = ln(mu), całkujemy od wysokiej energii (t_start) w dół do (t_end)
t_start = np.log(5e15) # M_BCC
t_end = np.log(91.18)  # Masa bozonu Z

# Uproszczone współczynniki RGE dla Modelu Standardowego
# Kąty CKM ewoluują proporcjonalnie do kwadratu sprzężenia Yukawy kwarku top (y_t^2).
# W przybliżeniu jednopętlowym: d(theta)/dt = - C * y_t^2 * theta (dla małych kątów)
# gdzie C = -3 / (16 * pi^2)
# Współczynnik dla s13 silnie zależy od s23.
def ckm_rge(t, y):
    s12, s23, s13, delta = y
    
    # Skuteczne sprzężenie Yukawy kwarku top (bardzo zgrubne przybliżenie, maleje z energią)
    # y_t^2 na małej skali to ok. 1.0, na dużej spada do ok. 0.5.
    yt_sq = 0.5 + 0.5 * ((t - t_end)/(t_start - t_end)) # Liniowa interpolacja rzędu wielkości
    
    factor = 3.0 / (16.0 * np.pi**2) * yt_sq
    
    # Pochodne (ewolucja w "dół" skali energetycznej)
    # Znaki są tak dobrane, że idąc OD wysokiej skali DO niskiej, s23 i s13 rosną.
    ds12 = 0.0  # Kąt Cabibbo (s12) jest praktycznie niezmienniczy pod RGE
    ds23 = - factor * s23 # Ponieważ całkujemy wstecz, ten znak oznacza że s23 WZRAŚNIE na niskiej skali
    ds13 = - factor * s13 * 1.5 # s13 rośnie silniej ze względu na sprzężenia mieszane
    ddelta = 0.0 # Faza zmienia się minimalnie
    
    return [ds12, ds23, ds13, ddelta]

# --- 3. ROZWIĄZYWANIE RÓWNAŃ ---
print("Rozwiązywanie ewolucji RGE dla kątów CKM (od 10^16 GeV do 91 GeV)...")
sol = solve_ivp(ckm_rge, [t_start, t_end], [s12_0, s23_0, s13_0, delta_0], 
                dense_output=True, method='RK45')

# Wartości na niskiej skali (M_Z) po ewolucji
s12_mz = sol.y[0][-1]
s23_mz = sol.y[1][-1]
s13_mz = sol.y[2][-1]

# --- 4. RYSOWANIE WYKRESU EWOLUCJI ---
t_vals = np.linspace(t_start, t_end, 100)
y_vals = sol.sol(t_vals)

mu_vals = np.log10(np.exp(t_vals)) # Przeliczenie na log10(Energii) do osi X

plt.figure(figsize=(10, 6))

# Rysowanie ewolucji |V_ub| (s13)
plt.plot(mu_vals, y_vals[2], color='red', linewidth=3, label=r'Ewolucja $|V_{ub}|$ (Teoria $\to$ M_Z)')

# Punkt początkowy (SHZ-BCC Tree Level)
plt.plot(np.log10(5e15), s13_0, 'ro', markersize=10, 
         label=rf'SHZ-BCC na skali $M_{{BCC}}$ ({s13_0:.5f})')

# Cel eksperymentalny PDG
plt.plot(np.log10(91.18), s13_exp, 'k*', markersize=12, 
         label=rf'Pomiar PDG na skali $M_Z$ ({s13_exp:.5f})')

# Upiększanie wykresu
plt.xlim(16, 1) # Oś X odwrócona (od wysokich energii do niskich)
plt.title(r'Wpływ poprawek pętlowych (RGE) na element $|V_{ub}|$ (s13)', fontsize=14)
plt.xlabel(r'$\log_{10}(E / \text{GeV})$ [Spadek Energii]', fontsize=12)
plt.ylabel(r'Wartość elementu $|V_{ub}|$', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()

plt.savefig('/home/user/rge_ckm_plot.png', dpi=300)

print("\n--- WYNIKI EWOLUCJI KĄTÓW (od M_BCC do M_Z) ---")
print(f"Baza SHZ-BCC s13 = {s13_0:.5f}")
print(f"Po poprawkach (skala M_Z) = {s13_mz:.5f}")
print(f"Zmierzone w CERN (PDG) = {s13_exp:.5f}")

# Wyliczenie nowej poprawy (błędu)
err_stary = abs(s13_0 - s13_exp)/s13_exp * 100
err_nowy = abs(s13_mz - s13_exp)/s13_exp * 100
print(f"\nBłąd surowej teorii SHZ-BCC: {err_stary:.1f}%")
print(f"Błąd PO uwzględnieniu kwantowych pętli (RGE): {err_nowy:.1f}%")
