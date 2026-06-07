import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print("1. Generowanie 100 000 syntetycznych wariacji modelu SHZ-BCC...")
N = 100000
np.random.seed(42)

# --- CECHY (FEATURES) - FUNDAMENTALNE PARAMETRY TEORII ---
# M_BCC: Skala łamania symetrii [10^15 do 10^16 GeV]
M_BCC = np.random.uniform(1e15, 1e16, N)
# m_A: Masa "Strażnika Zapachu" (Higgsa 2HDM) [500 do 3000 GeV]
m_A = np.random.uniform(500, 3000, N)
# M_2: Masa sterylnego neutrina [10^9 do 10^11 GeV]
M_2 = np.random.uniform(1e9, 1e11, N)
# N_e: Liczba e-składzeń Inflacji Kosmicznej [50 do 75]
N_e = np.random.uniform(50, 75, N)

# --- FIZYKA (OBSERVABLES) - WYLICZANIE SKUTKÓW ---
# 1. Czas życia protonu (zależy od M_BCC^4)
tau_p = 1.3e35 * (M_BCC / 5e15)**4
# 2. Amplituda Fal Grawitacyjnych (rozrzedzona przez N_e)
G_mu = (M_BCC / 1.22e19)**2
Omega_GW = 1e-4 * np.sqrt(G_mu) * np.exp(-2 * (N_e - 60)) 
# 3. Asymetria Materii (zależy liniowo od masy neutrina M_2 w uproszczeniu termicznym)
Y_B = 8.7e-11 * (M_2 / 1e10)

# --- KRYTERIA PRZETRWANIA (CZY MODEL ODRZUCONY PRZEZ EKSPERYMENTY?) ---
valid_proton = tau_p > 2.4e34             # Super-Kamiokande limit
valid_GW = Omega_GW < 1e-9                # NANOGrav limit
valid_Baryon = (Y_B > 5e-11) & (Y_B < 2e-10) # Pomiary satelity Planck
valid_Collider = m_A > 1200               # Limity z LHC w CERN

# Label (1 - model przetrwał eksperymenty, 0 - obalony)
y = (valid_proton & valid_GW & valid_Baryon & valid_Collider).astype(int)

print(f"Stworzono baze. Liczba wszechświatów zgodnych z danymi: {np.sum(y)} / {N} ({(np.sum(y)/N)*100:.1f}%)")

# Przygotowanie danych dla AI
X = np.column_stack((M_BCC, m_A, M_2, N_e))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n2. Trenowanie modelu Sztucznej Inteligencji (Random Forest)...")
rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Ocena modelu
print("\nRaport skuteczności AI na zbiorze testowym (N=20000):")
print(classification_report(y_test, rf.predict(X_test), target_names=["Odrzucony", "Przetrwał"]))

# --- WAŻNOŚĆ CECH (FEATURE IMPORTANCE) ---
importances = rf.feature_importances_
features = ['Skala Pati-Salama ($M_{BCC}$)', 'Masa Tarczy ($m_A$)', 'Masa Neutrina ($M_2$)', 'Inflacja ($N_e$)']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Wykres 1: Czego AI nauczyło się o hierarchii parametrów?
y_pos = np.arange(len(features))
ax1.barh(y_pos, importances, align='center', color=['#ff5555', '#55ff55', '#5555ff', '#ffaa00'])
ax1.set_yticks(y_pos)
ax1.set_yticklabels(features, fontsize=11)
ax1.invert_yaxis()  
ax1.set_xlabel('Wpływ na przeżywalność modelu (Feature Importance)', fontsize=12)
ax1.set_title('Co decyduje o obaleniu/potwierdzeniu teorii?', fontsize=14)
ax1.grid(True, axis='x', linestyle='--', alpha=0.5)

# Wykres 2: Wizualizacja Lasu Losowego w przestrzeni parametrów (M_BCC vs N_e)
# Zobrazujmy tylko próbkę 2000 punktów dla czytelności
mask_survive = y_test == 1
mask_die = y_test == 0

ax2.scatter(X_test[mask_die, 0][:1000], X_test[mask_die, 3][:1000], 
            color='red', alpha=0.3, label='Odrzucone (Np. za dużo GW / Rozpad protonu)', s=10)
ax2.scatter(X_test[mask_survive, 0][:1000], X_test[mask_survive, 3][:1000], 
            color='green', alpha=0.8, label='Przetrwały (Złota strefa SHZ-BCC)', s=15)

ax2.set_xlabel(r'Skala Unifikacji Pati-Salama $M_{BCC}$ [GeV]', fontsize=12)
ax2.set_ylabel(r'Liczba e-składzeń Inflacji $N_e$', fontsize=12)
ax2.set_title('Przestrzeń przetrwania wyznaczona przez AI', fontsize=14)
ax2.legend(loc='lower right')
ax2.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('/home/user/ai_parameter_scan.png', dpi=300)
print("\nGrafika z analizą AI została zapisana: ai_parameter_scan.png")
