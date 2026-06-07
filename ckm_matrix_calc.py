import numpy as np
np.set_printoptions(precision=5, suppress=True)

# Fundamentalny parametr z modelu SHZ-BCC
lam_bcc = 1.0 / (np.pi * np.sqrt(2))  # ok. 0.225079

# Wzory na wartości sinusów z pracy (sekcja 7.4)
s12 = lam_bcc
s23 = np.sqrt(2.0/3.0) * (lam_bcc**2)
s13 = np.sqrt(2.0/3.0) * (1.0/np.sqrt(8.0)) * (lam_bcc**3)

# Cosinusy mieszania
c12 = np.sqrt(1 - s12**2)
c23 = np.sqrt(1 - s23**2)
c13 = np.sqrt(1 - s13**2)

# Faza łamania CP (holonomia CP z pracy)
delta_rad = np.arccos(1.0 / 3.0)  # ok. 70.53 stopnia

# Macierz obrotu U12
U12 = np.array([
    [ c12, s12,   0],
    [-s12, c12,   0],
    [   0,   0,   1]
], dtype=complex)

# Macierz obrotu U23
U23 = np.array([
    [1,    0,   0],
    [0,  c23, s23],
    [0, -s23, c23]
], dtype=complex)

# Macierz obrotu U13 z fazą urojeniową (łamanie CP)
e_minus_id = np.cos(delta_rad) - 1j * np.sin(delta_rad)
e_plus_id  = np.cos(delta_rad) + 1j * np.sin(delta_rad)

U13 = np.array([
    [ c13,             0, s13 * e_minus_id],
    [   0,             1,                0],
    [-s13 * e_plus_id, 0,              c13]
], dtype=complex)

# Pełna Macierz CKM: V_CKM = U23 * U13 * U12  (konwencja Standardowa / PDG)
V_CKM = np.matmul(U23, np.matmul(U13, U12))

# Obliczenie modułów elementów (to co mierzą eksperymenty)
V_CKM_abs = np.abs(V_CKM)

# Eksperymentalne wartości (Particle Data Group - PDG 2024 - Moduły)
V_PDG = np.array([
    [0.97435, 0.22500, 0.00369],
    [0.22486, 0.97349, 0.04182],
    [0.00857, 0.04110, 0.999118]
])

print("--- PARAMETRY WEJŚCIOWE Z TEORII SHZ-BCC ---")
print(f"lambda_BCC = {lam_bcc:.6f}")
print(f"s12 = {s12:.6f}")
print(f"s23 = {s23:.6f}")
print(f"s13 = {s13:.6f}")
print(f"Faza delta = {np.degrees(delta_rad):.2f} stopni")
print("\n--- PEŁNA MACIERZ CKM (Moduły |V_ij|) Z SHZ-BCC ---")
print(V_CKM_abs)
print("\n--- MACIERZ ZMIERZONA PRZEZ PDG 2024 ---")
print(V_PDG)

print("\n--- BŁĄD WZGLĘDNY (% ODCHYLENIA TEORII OD EKSPERYMENTU) ---")
error = np.abs((V_CKM_abs - V_PDG) / V_PDG) * 100
print(error)

# Dodatkowy, najważniejszy parametr łamania symetrii materii-antymaterii: Inwariant Jarlskoga
# J = s12 c12 s23 c23 s13 c13^2 sin(delta)
J_bcc = s12 * c12 * s23 * c23 * s13 * (c13**2) * np.sin(delta_rad)
J_pdg = 3.08e-5 # Wartość z PDG
print("\n--- INWARIANT JARLSKOGA (Łamanie CP) ---")
print(f"J_BCC (Teoria) = {J_bcc:.2e}")
print(f"J_PDG (Eksperym)= {J_pdg:.2e}")

