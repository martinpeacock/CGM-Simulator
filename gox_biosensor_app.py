# -*- coding: utf-8 -*-
# gox_biosensor_app.py
# Streamlit UI for the GOx biosensor physics engine

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from gox_biosensor_engine import (
    run_gox_simulation,
    ppm_to_mM,
)

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.title("Glucose Oxidase Biosensor Simulator")
st.write(
    "Mechanistic GOx model with stepwise glucose input, oxygen in ppm, "
    "amperometric current, CSV export, and parameter sweeps."
)

tabs = st.tabs(["Simulation", "Parameter sweep"])

# ---------------------------------------------------------
# SIMULATION TAB
# ---------------------------------------------------------
with tabs[0]:
    st.header("Single-run simulation")

    # --- Glucose step settings ---
    st.subheader("Glucose step protocol")

    n_steps = st.slider("Number of glucose steps", 1, 10, 6)
    step_duration = st.slider("Step duration (s)", 50, 1000, 150, 10)

    default_concs = [0, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    glucose_steps_mM = []

    for i in range(n_steps):
        val = default_concs[i]
        conc = st.number_input(
            f"Step {i+1} glucose (mM)",
            min_value=0.0, max_value=100.0,
            value=float(val), step=0.5
        )
        glucose_steps_mM.append(conc)

    # --- Kinetic parameters ---
    st.subheader("Kinetic parameters")

    k1   = st.slider("k1 (M⁻¹ s⁻¹)", 0.01, 10.0, 1.0, 0.01)
    km1  = st.slider("k-1 (s⁻¹)", 0.01, 10.0, 0.5, 0.01)
    k2   = st.slider("k2 (s⁻¹)", 0.01, 10.0, 1.0, 0.01)
    k3   = st.slider("k3 (M⁻¹ s⁻¹)", 0.01, 10.0, 1.0, 0.01)

    E_tot_mM = st.slider("Total enzyme [E] (mM)", 0.001, 1.0, 0.1, 0.001)

    # --- Oxygen in ppm ---
    st.subheader("Dissolved oxygen (ppm)")

    O2_ppm = st.selectbox("Initial O₂ (ppm)", [0, 1, 2, 3, 4, 5, 6])
    O2_bath_ppm = st.selectbox("Bath O₂ (ppm)", [0, 1, 2, 3, 4, 5, 6])

    O2_0_mM = ppm_to_mM(O2_ppm)
    O2_bath_mM = ppm_to_mM(O2_bath_ppm)

    O2_mode = st.selectbox("Oxygen mode", ["closed", "well-aerated"])

    # --- Run simulation ---
    if st.button("Run simulation"):
        result = run_gox_simulation(
            k1=k1,
            km1=km1,
            k2=k2,
            k3=k3,
            E_tot_mM=E_tot_mM,
            O2_mode=O2_mode,
            O2_0_ppm=O2_ppm,
            O2_bath_ppm=O2_bath_ppm,
            glucose_steps_mM=glucose_steps_mM,
            step_duration_s=step_duration,
        )

        t = result["t"]
        P_mM = result["P_M"] * 1e3
        H2O2_mM = result["H2O2_M"] * 1e3
        O2_mM = result["O2_M"] * 1e3
        glucose_mM = result["glucose_mM"]
        current = result["current_AU"]

        # --- Plots ---
        st.subheader("Time courses")

        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(t, P_mM, label="Product (mM)")
        ax1.plot(t, H2O2_mM, label="H2O2 (mM)")
        ax1.plot(t, O2_mM, label="O2 (mM)")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Concentration (mM)")
        ax1.grid(True)
        ax1.legend()
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.plot(t, glucose_mM, color="purple")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Glucose (mM)")
        ax2.grid(True)
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots(figsize=(8, 3))
        ax3.plot(t, current)
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Current (A.U.)")
        ax3.grid(True)
        st.pyplot(fig3)

        # --- CSV export ---
        st.subheader("Export time series")

        df = pd.DataFrame({
            "time_s": t,
            "P_mM": P_mM,
            "H2O2_mM": H2O2_mM,
            "O2_mM": O2_mM,
            "glucose_mM": glucose_mM,
            "current_AU": current,
        })

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="gox_biosensor_timeseries.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------
# PARAMETER SWEEP TAB
# ---------------------------------------------------------
with tabs[1]:
    st.header("Parameter sweep: enzyme vs oxygen")

    st.write("Sweeps enzyme loading and oxygen (ppm) and reports peak current.")

    # Sweep settings
    n_steps_sw = st.slider("Number of glucose steps", 1, 6, 4)
    step_duration_sw = st.slider("Step duration (s)", 50, 1000, 150, 50)

    default_concs_sw = [0, 4, 6, 8, 10, 12]
    glucose_steps_mM_sw = []

    for i in range(n_steps_sw):
        conc = st.number_input(
            f"Sweep step {i+1} glucose (mM)",
            min_value=0.0, max_value=100.0,
            value=float(default_concs_sw[i]),
            step=0.5,
            key=f"sw_gluc_{i}"
        )
        glucose_steps_mM_sw.append(conc)

    # Sweep ranges
    E_min = st.number_input("Min enzyme (mM)", 0.001, 1.0, 0.01, 0.001)
    E_max = st.number_input("Max enzyme (mM)", 0.001, 1.0, 0.5, 0.001)
    n_E   = st.slider("Number of enzyme points", 3, 20, 8)

    O2_ppm_min = st.selectbox("Min O₂ (ppm)", [0, 1, 2, 3, 4, 5, 6])
    O2_ppm_max = st.selectbox("Max O₂ (ppm)", [0, 1, 2, 3, 4, 5, 6])
    n_O2       = st.slider("Number of O₂ points", 3, 20, 8)

    O2_mode_sw = st.selectbox("Oxygen mode (sweep)", ["closed", "well-aerated"])
    O2_bath_ppm_sw = st.selectbox("Bath O₂ (ppm, sweep)", [0, 1, 2, 3, 4, 5, 6])

    # Kinetics
    k1_sw   = st.slider("k1 (M⁻¹ s⁻¹)", 0.01, 10.0, 1.0, 0.01)
    km1_sw  = st.slider("k-1 (s⁻¹)", 0.01, 10.0, 0.5, 0.01)
    k2_sw   = st.slider("k2 (s⁻¹)", 0.01, 10.0, 1.0, 0.01)
    k3_sw   = st.slider("k3 (M⁻¹ s⁻¹)", 0.01, 10.0, 1.0, 0.01)

    if st.button("Run parameter sweep"):
        E_vals = np.linspace(E_min, E_max, n_E)
        O2_vals_ppm = np.linspace(O2_ppm_min, O2_ppm_max, n_O2)

        peak_signal = np.zeros((n_O2, n_E))

        for i, O2_ppm_val in enumerate(O2_vals_ppm):
            for j, E_tot_mM in enumerate(E_vals):
                result = run_gox_simulation(
                    k1=k1_sw,
                    km1=km1_sw,
                    k2=k2_sw,
                    k3=k3_sw,
                    E_tot_mM=E_tot_mM,
                    O2_mode=O2_mode_sw,
                    O2_0_ppm=O2_ppm_val,
                    O2_bath_ppm=O2_bath_ppm_sw,
                    glucose_steps_mM=glucose_steps_mM_sw,
                    step_duration_s=step_duration_sw,
                    n_points=800,
                )
                peak_signal[i, j] = np.max(result["current_AU"])

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(
            peak_signal,
            aspect="auto",
            origin="lower",
            extent=[E_min, E_max, O2_ppm_min, O2_ppm_max]
        )
        ax.set_xlabel("Enzyme (mM)")
        ax.set_ylabel("O₂ (ppm)")
        ax.set_title("Peak amperometric signal (A.U.)")
        fig.colorbar(im, ax=ax, label="Peak current (A.U.)")
        st.pyplot(fig)

        df_sweep = pd.DataFrame({
            "E_mM": np.repeat(E_vals, n_O2),
            "O2_ppm": np.tile(O2_vals_ppm, n_E),
            "peak_current_AU": peak_signal.flatten()
        })

        csv_sw = df_sweep.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download sweep results CSV",
            data=csv_sw,
            file_name="gox_biosensor_sweep.csv",
            mime="text/csv",
        )