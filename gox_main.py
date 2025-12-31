# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 23:03:53 2025

@author: martp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from gox_biosensor_engine import run_gox_simulation

# ---------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------

ELECTRODE_AREA_MM2 = 0.2  # mm²

def units_per_electrode_to_mM(E_units, film_thickness_um, kcat_s_inv):
    """Convert enzyme loading in Units per electrode to effective mM inside the film."""
    if E_units <= 0 or film_thickness_um <= 0 or kcat_s_inv <= 0:
        return 0.0

    rate_mol_per_s = E_units * 1e-6 / 60.0
    n_E_mol = rate_mol_per_s / kcat_s_inv

    V_film_L = ELECTRODE_AREA_MM2 * film_thickness_um * 1e-9
    if V_film_L <= 0:
        return 0.0

    return (n_E_mol / V_film_L) * 1e3  # M → mM


# ---------------------------------------------------------
# Streamlit layout
# ---------------------------------------------------------

st.set_page_config(layout="wide")
st.title("GOx Biosensor Simulator (Minimal Working Version)")

sidebar = st.sidebar
sidebar.header("Simulation controls")

# =========================================================
# INPUTS
# =========================================================

sidebar.subheader("Bulk glucose steps")

n_steps = sidebar.slider("Number of glucose steps", 1, 10, 6)
step_duration = sidebar.slider("Step duration (s)", 50, 1000, 150, 10)

default_concs = [0, 4, 6, 8, 10, 12, 14, 16, 18, 20]
glucose_steps_mM = []
for i in range(n_steps):
    glucose_steps_mM.append(
        sidebar.number_input(
            f"Step {i+1} glucose (mM)",
            min_value=0.0,
            max_value=100.0,
            value=float(default_concs[i]),
            step=0.5,
            key=f"glu_{i}",
        )
    )

sidebar.subheader("Kinetics")

Km_glu_mM = sidebar.slider("Km (mM)", 0.1, 50.0, 10.0, 0.1)
kcat_glu = sidebar.slider("kcat (s⁻¹)", 0.01, 500.0, 100.0, 0.5)
k3 = sidebar.slider("k3 (M⁻¹ s⁻¹)", 0.01, 1e4, 100.0, 1.0)

Km_glu_M = Km_glu_mM * 1e-3
km1 = 1.0
k2 = kcat_glu
k1 = (km1 + k2) / Km_glu_M if Km_glu_M > 0 else 0.0

sidebar.subheader("Enzyme loading")

E_units = sidebar.slider("GOx loading (U)", 0.01, 10.0, 1.0, 0.01)
film_thickness_um = sidebar.slider("Film thickness (µm)", 1.0, 200.0, 20.0, 1.0)

E_tot_mM_sim = units_per_electrode_to_mM(E_units, film_thickness_um, kcat_glu)
sidebar.write(f"Effective [GOx] = {E_tot_mM_sim:.3g} mM")

sidebar.subheader("Oxygen")

O2_ppm = sidebar.selectbox("Initial O₂ (ppm)", [0,1,2,3,4,5,6], index=6)
O2_bath_ppm = sidebar.selectbox("Bath O₂ (ppm)", [0,1,2,3,4,5,6], index=6)
O2_mode = sidebar.selectbox("O₂ mode", ["closed", "well-aerated"], index=0)

run_sim = sidebar.button("Run simulation")

# =========================================================
# SIMULATION
# =========================================================

if run_sim:

    result = run_gox_simulation(
        k1=k1,
        km1=km1,
        k2=k2,
        k3=k3,
        E_tot_mM=E_tot_mM_sim,
        O2_mode=O2_mode,
        O2_0_ppm=O2_ppm,
        O2_bath_ppm=O2_bath_ppm,
        glucose_steps_mM=glucose_steps_mM,
        step_duration_s=step_duration,
        n_points=2000,
    )

    # Extract outputs
    t = result["t"]
    P_mM = result["P_M"] * 1e3
    H2O2_mM = result["H2O2_M"] * 1e3
    O2_mM = result["O2_M"] * 1e3
    glucose_mM = result["glucose_mM"]
    current = result["current_AU"]

    # Plot film species
    st.subheader("Film species")
    fig1, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(t, P_mM, label="P (mM)")
    ax1.plot(t, H2O2_mM, label="H₂O₂ (mM)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Concentration (mM)")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(t, O2_mM, color="red", linestyle="--", label="O₂ (mM)")
    ax2.set_ylabel("O₂ (mM)", color="red")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    st.pyplot(fig1)

    # Glucose protocol
    st.subheader("Glucose input")
    fig2, ax_glu = plt.subplots(figsize=(8,3))
    ax_glu.plot(t, glucose_mM, color="purple")
    ax_glu.set_xlabel("Time (s)")
    ax_glu.set_ylabel("Glucose (mM)")
    ax_glu.grid(True)
    st.pyplot(fig2)

    # Current
    st.subheader("Current")
    fig3, ax_cur = plt.subplots(figsize=(8,3))
    ax_cur.plot(t, current, color="black")
    ax_cur.set_xlabel("Time (s)")
    ax_cur.set_ylabel("Current (A.U.)")
    ax_cur.grid(True)
    st.pyplot(fig3)