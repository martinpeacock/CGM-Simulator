# gox_biosensor_app.py
# Streamlit UI for the immobilized GOx biosensor physics engine (Level A, bulk + film)
# Now using enzyme loading in Units per electrode in the UI,
# internally converted to mM for the mechanistic engine.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from gox_biosensor_engine import (
    run_gox_simulation,
    ppm_to_mM,
)

# ---------------------------------------------------------
# CONSTANTS FOR ENZYME UNIT → mM CONVERSION
# ---------------------------------------------------------
# Assumed geometric electrode area (adjust to match your design)
ELECTRODE_AREA_MM2 = 0.2  # e.g. 0.1 mm × 2 mm ~ 0.2 mm²

# Assumed catalytic turnover for GOx (order of magnitude)
KCAT_S_INV = 1000.0       # 1000 s⁻¹ ~ 60000 min⁻¹


def units_per_electrode_to_mM(E_units, film_thickness_um, electrode_area_mm2=ELECTRODE_AREA_MM2, kcat_s_inv=KCAT_S_INV):
    """
    Convert enzyme loading in Units per electrode to an effective concentration in mM
    inside the film, based on film volume and an assumed kcat.

    1 U/electrode = 1 µmol glucose converted per electrode per minute at saturating substrate.

    Steps:
    - n_E (mol enzyme) = (E_units * 1e-6 mol/min) / (kcat_min⁻¹)
    - kcat_min⁻¹ = kcat_s_inv * 60
    - V_film (L) = area(mm²) * thickness(µm) * 1e-9
    - [E] (M) = n_E / V_film
    - [E] (mM) = [E] (M) * 1e3
    """

    if E_units <= 0:
        return 0.0

    # kcat in min⁻¹
    kcat_min_inv = kcat_s_inv * 60.0

    # moles of enzyme per electrode
    n_E_mol = (E_units * 1e-6) / kcat_min_inv  # 1 U = 1 µmol/min

    # film volume in liters
    V_film_L = electrode_area_mm2 * film_thickness_um * 1e-9  # L

    if V_film_L <= 0:
        return 0.0

    E_tot_M = n_E_mol / V_film_L
    E_tot_mM = E_tot_M * 1e3
    return E_tot_mM


# ---------------------------------------------------------
# PAGE LAYOUT
# ---------------------------------------------------------
st.set_page_config(layout="wide")
st.title("Immobilized Glucose Oxidase Biosensor Simulator")

# Sidebar for all controls
sidebar = st.sidebar
sidebar.header("Simulation Controls")

tabs = st.tabs(["Simulation", "Parameter sweep"])

# ---------------------------------------------------------
# SIMULATION TAB
# ---------------------------------------------------------
with tabs[0]:

    # --- Glucose step settings ---
    sidebar.subheader("Bulk glucose steps")

    n_steps = sidebar.slider("Number of glucose steps", 1, 10, 6, key="sim_n_steps")
    step_duration = sidebar.slider("Step duration (s)", 50, 1000, 150, 10, key="sim_step_duration")

    default_concs = [0, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    glucose_steps_mM = []

    for i in range(n_steps):
        conc = sidebar.number_input(
            f"Step {i+1} bulk glucose (mM)",
            min_value=0.0, max_value=100.0,
            value=float(default_concs[i]),
            step=0.5,
            key=f"sim_glucose_step_{i}"
        )
        glucose_steps_mM.append(conc)

    # --- Kinetic parameters ---
    sidebar.subheader("Kinetic parameters (film GOx)")

    k1   = sidebar.slider("k1 (M⁻¹ s⁻¹)", 0.01, 10.0, 1.0, 0.01, key="sim_k1")
    km1  = sidebar.slider("k-1 (s⁻¹)", 0.01, 10.0, 0.5, 0.01, key="sim_km1")
    k2   = sidebar.slider("k2 (s⁻¹)", 0.01, 10.0, 1.0, 0.01, key="sim_k2")
    k3   = sidebar.slider("k3 (M⁻¹ s⁻¹)", 0.01, 10.0, 1.0, 0.01, key="sim_k3")

    # Enzyme loading in Units per electrode (UI)
    sidebar.subheader("Enzyme loading")

    E_units = sidebar.slider(
        "Glucose oxidase loading (U per electrode)",
        min_value=0.01,
        max_value=10.0,
        value=1.0,
        step=0.01,
        key="sim_E_units",
        help="1 U/electrode = 1 µmol glucose converted per electrode per minute at saturating substrate."
    )

    # --- Oxygen in ppm ---
    sidebar.subheader("Dissolved oxygen (bulk / film)")

    O2_ppm = sidebar.selectbox("Initial bulk O₂ (ppm)", [0, 1, 2, 3, 4, 5, 6], key="sim_O2_ppm")
    O2_bath_ppm = sidebar.selectbox("Film bath O₂ (ppm)", [0, 1, 2, 3, 4, 5, 6], key="sim_O2_bath_ppm")

    O2_mode = sidebar.selectbox("Film oxygen mode", ["closed", "well-aerated"], key="sim_O2_mode")

    # --- Geometry & mass transfer ---
    sidebar.subheader("Geometry and mass transfer")

    film_thickness_um = sidebar.slider("Film thickness (µm)", 1.0, 200.0, 20.0, 1.0, key="sim_film_thickness")
    k_mt_glucose = sidebar.number_input(
        "k_mt_glucose (m/s)",
        min_value=1e-7, max_value=1e-3, value=1e-5, step=1e-6, format="%.1e",
        key="sim_k_mt_glucose"
    )
    k_mt_O2 = sidebar.number_input(
        "k_mt_O2 (m/s)",
        min_value=1e-7, max_value=1e-3, value=1e-5, step=1e-6, format="%.1e",
        key="sim_k_mt_O2"
    )
    V_bulk_mL = sidebar.number_input(
        "Bulk volume (mL)",
        min_value=0.1, max_value=100.0, value=1.0, step=0.1,
        key="sim_V_bulk_ML"
    )

    # Convert U/electrode → mM for the engine
    E_tot_mM_sim = units_per_electrode_to_mM(E_units, film_thickness_um)

    run_button = sidebar.button("Run simulation", key="sim_run_button")

    # --- MAIN OUTPUT AREA ---
    if run_button:
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
            film_thickness_um=film_thickness_um,
            k_mt_glucose=k_mt_glucose,
            k_mt_O2=k_mt_O2,
            V_bulk_mL=V_bulk_mL,
        )

        t = result["t"]

        # Film species
        Gluconolactone_mM = result["P_M"] * 1e3
        H2O2_mM = result["H2O2_M"] * 1e3
        O2_film_mM = result["O2_film_M"] * 1e3

        # Glucose profiles
        glucose_bulk_mM = result["glucose_bulk_mM"]
        glucose_film_mM = result["glucose_film_mM"]

        current = result["current_AU"]

        # --- Film species plot with secondary axis ---
        st.subheader("Film species (immobilized region)")

        fig1, ax1 = plt.subplots(figsize=(8, 5))

        # Primary axis
        ax1.plot(t, Gluconolactone_mM, label="Gluconolactone_film (mM)", color="blue")
        ax1.plot(t, H2O2_mM, label="H2O2_film (mM)", color="green")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Gluconolactone / H2O2 (mM)")
        ax1.grid(True)

        # Secondary axis for O2
        ax2 = ax1.twinx()
        ax2.plot(t, O2_film_mM, label="O2_film (mM)", color="red", linestyle="--")
        ax2.set_ylabel("O2_film (mM)", color="red")

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        st.pyplot(fig1)

        # --- Bulk vs film glucose ---
        st.subheader("Bulk vs film glucose")

        fig2, ax2b = plt.subplots(figsize=(8, 3))
        ax2b.plot(t, glucose_bulk_mM, label="Bulk glucose (mM)", color="purple")
        ax2b.plot(t, glucose_film_mM, label="Film glucose (mM)", color="orange", linestyle="--")
        ax2b.set_xlabel("Time (s)")
        ax2b.set_ylabel("Glucose (mM)")
        ax2b.grid(True)
        ax2b.legend()
        st.pyplot(fig2)

        # --- Current ---
        st.subheader("Amperometric current (from film H₂O₂)")

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
            "Gluconolactone_film_mM": Gluconolactone_mM,
            "H2O2_film_mM": H2O2_mM,
            "O2_film_mM": O2_film_mM,
            "glucose_bulk_mM": glucose_bulk_mM,
            "glucose_film_mM": glucose_film_mM,
            "current_AU": current,
            "GOx_loading_U_per_electrode": E_units,
            "GOx_effective_E_tot_mM": E_tot_mM_sim,
        })

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="immobilized_gox_biosensor_timeseries.csv",
            mime="text/csv",
            key="sim_download_csv"
        )


# ---------------------------------------------------------
# PARAMETER SWEEP TAB
# ---------------------------------------------------------
with tabs[1]:

    sidebar.subheader("Sweep settings")

    n_steps_sw = sidebar.slider("Number of bulk glucose steps", 1, 6, 4, key="sw_n_steps")
    step_duration_sw = sidebar.slider("Step duration (s)", 50, 1000, 150, 50, key="sw_step_duration")

    default_concs_sw = [0, 4, 6, 8, 10, 12]
    glucose_steps_mM_sw = []

    for i in range(n_steps_sw):
        conc = sidebar.number_input(
            f"Sweep step {i+1} bulk glucose (mM)",
            min_value=0.0, max_value=100.0,
            value=float(default_concs_sw[i]),
            step=0.5,
            key=f"sw_glucose_step_{i}"
        )
        glucose_steps_mM_sw.append(conc)

    # Sweep ranges in Units per electrode
    E_units_min = sidebar.number_input("Min GOx loading (U per electrode)", 0.01, 50.0, 0.1, 0.01, key="sw_E_units_min")
    E_units_max = sidebar.number_input("Max GOx loading (U per electrode)", 0.01, 50.0, 5.0, 0.01, key="sw_E_units_max")
    n_E   = sidebar.slider("Number of enzyme points", 3, 20, 8, key="sw_n_E")

    O2_ppm_min = sidebar.selectbox("Min bulk O₂ (ppm)", [0, 1, 2, 3, 4, 5, 6], key="sw_O2_ppm_min")
    O2_ppm_max = sidebar.selectbox("Max bulk O₂ (ppm)", [0, 1, 2, 3, 4, 5, 6], key="sw_O2_ppm_max")
    n_O2       = sidebar.slider("Number of O₂ points", 3, 20, 8, key="sw_n_O2")

    O2_mode_sw = sidebar.selectbox("Film oxygen mode (sweep)", ["closed", "well-aerated"], key="sw_O2_mode")
    O2_bath_ppm_sw = sidebar.selectbox("Film bath O₂ (ppm, sweep)", [0, 1, 2, 3, 4, 5, 6], key="sw_O2_bath_ppm")

    # Geometry & mass transfer
    film_thickness_um_sw = sidebar.slider("Film thickness (µm, sweep)", 1.0, 200.0, 20.0, 1.0, key="sw_film_thickness")
    k_mt_glucose_sw = sidebar.number_input(
        "k_mt_glucose (m/s, sweep)",
        min_value=1e-7, max_value=1e-3, value=1e-5, step=1e-6, format="%.1e",
        key="sw_k_mt_glucose"
    )
    k_mt_O2_sw = sidebar.number_input(
        "k_mt_O2 (m/s, sweep)",
        min_value=1e-7, max_value=1e-3, value=1e-5, step=1e-6, format="%.1e",
        key="sw_k_mt_O2"
    )
    V_bulk_mL_sw = sidebar.number_input(
        "Bulk volume (mL, sweep)",
        min_value=0.1, max_value=100.0, value=1.0, step=0.1,
        key="sw_V_bulk_ML"
    )

    # Kinetics
    k1_sw   = sidebar.slider("k1 (M⁻¹ s⁻¹, sweep)", 0.01, 10.0, 1.0, 0.01, key="sw_k1")
    km1_sw  = sidebar.slider("k-1 (s⁻¹, sweep)", 0.01, 10.0, 0.5, 0.01, key="sw_km1")
    k2_sw   = sidebar.slider("k2 (s⁻¹, sweep)", 0.01, 10.0, 1.0, 0.01, key="sw_k2")
    k3_sw   = sidebar.slider("k3 (M⁻¹ s⁻¹, sweep)", 0.01, 10.0, 1.0, 0.01, key="sw_k3")

    run_sweep = sidebar.button("Run parameter sweep", key="sw_run_button")

    if run_sweep:
        E_units_vals = np.linspace(E_units_min, E_units_max, n_E)
        O2_vals_ppm = np.linspace(O2_ppm_min, O2_ppm_max, n_O2)

        peak_signal = np.zeros((n_O2, n_E))

        for i, O2_ppm_val in enumerate(O2_vals_ppm):
            for j, E_units_val in enumerate(E_units_vals):

                # Convert Units → mM for this combination
                E_tot_mM_sw = units_per_electrode_to_mM(E_units_val, film_thickness_um_sw)

                result = run_gox_simulation(
                    k1=k1_sw,
                    km1=km1_sw,
                    k2=k2_sw,
                    k3=k3_sw,
                    E_tot_mM=E_tot_mM_sw,
                    O2_mode=O2_mode_sw,
                    O2_0_ppm=O2_ppm_val,
                    O2_bath_ppm=O2_bath_ppm_sw,
                    glucose_steps_mM=glucose_steps_mM_sw,
                    step_duration_s=step_duration_sw,
                    film_thickness_um=film_thickness_um_sw,
                    k_mt_glucose=k_mt_glucose_sw,
                    k_mt_O2=k_mt_O2_sw,
                    V_bulk_mL=V_bulk_mL_sw,
                    n_points=800,
                )
                peak_signal[i, j] = np.max(result["current_AU"])

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(
            peak_signal,
            aspect="auto",
            origin="lower",
            extent=[E_units_min, E_units_max, O2_ppm_min, O2_ppm_max]
        )
        ax.set_xlabel("GOx loading (U per electrode)")
        ax.set_ylabel("Bulk O₂ (ppm)")
        ax.set_title("Peak film amperometric signal (A.U.)")
        fig.colorbar(im, ax=ax, label="Peak current (A.U.)")
        st.pyplot(fig)

        df_sweep = pd.DataFrame({
            "GOx_loading_U_per_electrode": np.repeat(E_units_vals, n_O2),
            "O2_bulk_ppm": np.tile(O2_vals_ppm, n_E),
            "peak_current_AU": peak_signal.flatten()
        })

        csv_sw = df_sweep.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download sweep results CSV",
            data=csv_sw,
            file_name="immobilized_gox_biosensor_sweep.csv",
            mime="text/csv",
            key="sw_download_csv"
        )