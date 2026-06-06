# 🌌 SHZ-BCC Phenomenology Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Physics: High Energy](https://img.shields.io/badge/Physics-High_Energy-purple.svg)]()
[![Status: Active](https://img.shields.io/badge/Status-Active-success.svg)]()

**SHZ-BCC-Pheno-Toolkit** is an open-source Python suite for simulating the phenomenological and cosmological consequences of the Horizon Boundary Consistency (SHZ-BCC) model.

The SHZ-BCC framework resolves the Cosmological Constant Problem by imposing discrete $Z_2^3$ constraints on causal horizons, projecting out divergent vacuum energy. When embedded in a Pati-Salam $SU(4)_C \times SU(2)_L \times SU(2)_R$ envelope, the model yields exact, parameter-free predictions for the CKM matrix, proton decay, and primordial gravitational waves.

## 🚀 Features

This toolkit provides numerical solvers and Monte Carlo simulations across four distinct frontiers of modern physics:

*   **Flavor Physics:** Renormalization Group Equations (RGEs) for CKM parameters and validation of the analytically derived Cabibbo angle ($\lambda = (\pi\sqrt{2})^{-1}$).
*   **Collider Phenomenology:** Monte Carlo generation of High-Luminosity LHC kinematics ($A^0 \to \tau^+\tau^-$) and Lepton Flavor Violation ($\mu \to e \gamma$).
*   **Cosmology:** Simulations of False Vacuum Inflation, Cosmic Strings (GW backgrounds), and Boltzmann Equations for thermal Leptogenesis (Dark Matter).
*   **Quantum Gravity:** Holographic Tensor Network simulations mapping the $Z_2^3$ constraints to Loop Quantum Gravity (LQG) $SU(2)$ spin networks and Entanglement Entropy.
*   **Machine Learning:** Random Forest AI parameter scanning of 100,000 synthetic universes to determine the theory's Goldilocks survival zone.

## 📁 Repository Structure

*   `src/` - Python source code (Solvers, ML, Monte Carlo).
*   `docs/` - Comprehensive LaTeX articles, phenomenological notes, and Feynman diagrams.
*   `results/` - Output directory for generated high-resolution plots.
*   `web/` - Interactive HTML/JS portal for data visualization.

## 🛠️ Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/YourUsername/SHZ-BCC-Pheno-Toolkit.git
cd SHZ-BCC-Pheno-Toolkit
pip install -r requirements.txt
```

## 💻 Quick Start

Run the Grand Simulation (combining Black Hole Thermodynamics, Leptogenesis, Collider limits, and Hubble Tension):
```bash
python src/grand_simulation.py
```

Run the Machine Learning Parameter Scan:
```bash
python src/ai_parameter_scan.py
```

## 📖 Citation
If you use this toolkit in your research, please consider citing the original SHZ-BCC preprint:
> Ślusarczyk, M. (2026). *Horizon Boundary Consistency and Vacuum-Energy Cancellation: A Quantum-Horizon Effective Framework with Standard-Model Embedding*. arXiv preprint.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
