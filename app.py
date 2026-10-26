import streamlit as st
import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cat Qubit Simulator", layout="wide")

# --- 1. PARTIE PHYSIQUE QUANTIQUE (Le moteur de calcul) ---
class BosonicCatQubitTwin:
    def __init__(self, N_fock=30, alpha=2.0):
        self.dim = N_fock
        self.alpha = alpha
        self.a = np.diag(np.sqrt(np.arange(1, N_fock)), k=1).astype(complex)
        self.a_dag = self.a.conj().T
        self.N_op = self.a_dag @ self.a
        self.parity_op = la.expm(1j * np.pi * self.N_op)
        self.cat_plus = self.build_cat_state(parity=+1)
        self.cat_minus = self.build_cat_state(parity=-1)

    def displacement_op(self, z):
        return la.expm(z * self.a_dag - np.conj(z) * self.a)

    def coherent_state(self, z):
        vac = np.zeros(self.dim, dtype=complex)
        vac[0] = 1.0
        return self.displacement_op(z) @ vac

    def build_cat_state(self, parity=+1):
        state = self.coherent_state(self.alpha) + (1 if parity > 0 else -1) * self.coherent_state(-self.alpha)
        return state / la.norm(state)

    def verify_parity_symmetry(self):
        p_plus = np.vdot(self.cat_plus, self.parity_op @ self.cat_plus).real
        p_minus = np.vdot(self.cat_minus, self.parity_op @ self.cat_minus).real
        return p_plus, p_minus

    def wigner_function(self, psi, x_vec, p_vec):
        W = np.zeros((len(p_vec), len(x_vec)))
        for i, pv in enumerate(p_vec):
            for j, xv in enumerate(x_vec):
                z = (xv + 1j * pv) / np.sqrt(2)
                D_z = self.displacement_op(z)
                W[i, j] = (2.0 / np.pi) * np.vdot(psi, (D_z @ self.parity_op @ D_z.conj().T) @ psi).real
        return W

    def calibrate_z_gate(self):
        a2 = self.a @ self.a
        a_dag2 = self.a_dag @ self.a_dag
        H_cat = -(a_dag2 @ a2) + (self.alpha ** 2) * (a_dag2 + a2)

        def loss(params):
            H_drive = params[0] * (self.a + self.a_dag) + params[1] * self.N_op
            final_state = la.expm(-1j * (H_cat + H_drive)) @ self.cat_plus
            return 1.0 - np.abs(np.vdot(self.cat_minus, final_state)) ** 2

        res = minimize(loss, x0=[0.3, 0.0], method='Nelder-Mead')
        return res.x, 1.0 - res.fun

# --- 2. PARTIE SITE WEB INTERACTIF (Ce que voit le recruteur) ---
st.title("🐈 Simulateur de Cat Qubit & Calibration Quantique")
st.write("Déplacez le curseur à gauche pour modifier le nombre de photons et observer les interférences quantiques en direct.")

alpha = st.sidebar.slider("Amplitude du Qubit (α)", 0.5, 3.0, 2.0, 0.1)
twin = BosonicCatQubitTwin(N_fock=25, alpha=alpha)
p_even, p_odd = twin.verify_parity_symmetry()

col1, col2, col3 = st.columns(3)
col1.metric("Photons moyens |α|²", f"{alpha**2:.2f}")
col2.metric("Parité État Pair", f"{p_even:+.4f}")
col3.metric("Parité État Impair", f"{p_odd:+.4f}")

grid = np.linspace(-4.0, 4.0, 40)
W_plus = twin.wigner_function(twin.cat_plus, grid, grid)
W_minus = twin.wigner_function(twin.cat_minus, grid, grid)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
im0 = axes[0].contourf(grid, grid, W_plus, levels=40, cmap="RdBu_r", vmin=-0.65, vmax=0.65)
axes[0].set_title(f"État Chat Pair |C+⟩ (α = {alpha})")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].contourf(grid, grid, W_minus, levels=40, cmap="RdBu_r", vmin=-0.65, vmax=0.65)
axes[1].set_title(f"État Chat Impair |C-⟩ (α = {alpha})")
plt.colorbar(im1, ax=axes[1])
st.pyplot(fig)

if st.button("🚀 Lancer la calibration automatique de la porte quantique Z"):
    opt_params, fid = twin.calibrate_z_gate()
    st.success(f"Porte calibrée avec succès ! Fidélité : {fid*100:.2f}% (Impulsion optimale = {opt_params[0]:.3f})")
