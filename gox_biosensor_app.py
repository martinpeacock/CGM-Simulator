# gox_biosensor_app.py
# Streamlit UI for the GOx biosensor physics engine (Michaelis–Menten mode only)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from gox_biosensor_engine import run_gox_simulation


# ---------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------

# Assumed geometric electrode area (mm²) for film volume
ELECTRODE_AREA_MM2 = 0.2  # adjust if needed


def units_per_electrode_to_mM(
    E_units: float,
    film_thickness_um: float,
    kcat_s_inv: float,
    electrode_area_mm2: float = ELECTRODE_AREA_MM2,
) -> float:
    """
    Convert enzyme loading in Units per electrode to an effective concentration in mM
    inside the film, based on film volume and kcat.

    1 U = 1 µmol substrate converted per minute at saturating substrate.
    """
    if E_units <= 0 or film_thickness_um <= 0 or kcat_s_inv <= 0:
        return 0.0

    # 1 U = 1e-6 mol/min
    rate_mol_per_s = E_units * 1e-6 / 60.0
    n_E_mol = rate_mol_per_s / kcat_s_inv

    # film volume in liters: mm² * µm * 1e-9
    V_film_L = electrode_area_mm2 * film_thickness_um * 1e-9
    if V_film_L <= 0:
        return 0.0

    E_tot_M = n_E_mol / V_film_L
    return E_tot_M * 1e3  # M → mM


# ---------------------------------------------------------
# Streamlit layout
# ---------------------------------------------------------

st.set_page_config(layout="wide")
st.title("Glucose Oxidase Biosensor Simulator (Michaelis–Menten mode)")

sidebar = st.sidebar
sidebar.header("Simulation controls")

tabs = st.tabs(["Simulation", "Parameter sweep"])

# =========================================================
# SIMULATION TAB
# =========================================================
with tabs[0]:

    # ------------------------------
    # Glucose step protocol
    # ------------------------------
    sidebar.subheader("Bulk glucose steps")

    n_steps = sidebar.slider("Number of glucose steps", 1, 10, 6)
    step_duration = sidebar.slider("Step duration (s)", 50, 1000, 150, 10)

    default_concs = [0, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    glucose_steps_mM = [
        sidebar.number_input(
            f"Step {i+1} bulk glucose (mM)",
            min_value=0.0,
            max_value=100.0,
            value=float(default_concs[i]),
            step=0.5,
            key=f"sim_glucose_step_{i}",
        )
        for i in range(n_steps)
    ]

    # ------------------------------
    # Kinetic parameters (MM mode)
    # ------------------------------
    sidebar.subheader("Kinetics (Michaelis–Menten for glucose)")

    Km_glu_mM = sidebar.slider(
        "Glucose Km (mM)",
        0.1,
        50.0,
        10.0,
        0.1,
        help="Typical GOx Km in vitro ~10–30 mM.",
    )

    kcat_glu = sidebar.slider(
        "Glucose kcat (s⁻¹)",
        0.01,
        500.0,
        100.0,
        0.5,
        help="Effective turnover number inside the film.",
    )

    k3 = sidebar.slider(
        "O₂ reaction rate k3 (M⁻¹ s⁻¹)",
        0.01,
        1e4,
        100.0,
        1.0,
        help="Bimolecular rate constant for reduced GOx reacting with O₂.",
    )

    # Map MM → mechanistic
    Km_glu_M = Km_glu_mM * 1e-3
    km1 = 1.0  # fixed ES dissociation
    k2 = kcat_glu
    k1 = (km1 + k2) / Km_glu_M if Km_glu_M > 0 else 0.0

    # ------------------------------
    # Enzyme loading and geometry
    # ------------------------------
    sidebar.subheader("Enzyme loading and film geometry")

    E_units = sidebar.slider(
        "GOx loading (U per electrode)",
        0.01,
        10.0,
        1.0,
        0.01,
        help="1 U = 1 µmol/min at saturating substrate.",
    )

    film_thickness_um = sidebar.slider(
        "Film thickness (µm)", 1.0, 200.0, 20.0, 1.0
    )

    E_tot_mM_sim = units_per_electrode_to_mM(
        E_units=E_units,
        film_thickness_um=film_thickness_um,
        kcat_s_inv=kcat_glu,
        electrode_area_mm2=ELECTRODE_AREA_MM2,
    )

    sidebar.markdown(
        f"**Effective [GOx] in film:** {E_tot_mM_sim:.3g} mM "
        f"(area = {ELECTRODE_AREA_MM2} mm²)"
    )

    # ------------------------------
    # Oxygen settings
    # ------------------------------
    sidebar.subheader("Oxygen")

    O2_ppm = sidebar.selectbox(
        "Initial bulk O₂ (ppm)", [0, 1, 2, 3, 4, 5, 6], index=6
    )
    O2_bath_ppm = sidebar.selectbox(
        "Bath O₂ (ppm)", [0, 1, 2, 3, 4, 5, 6], index=6
    )
    O2_mode = sidebar.selectbox(
        "Oxygen mode", ["closed", "well-aerated"], index=0
    )

    # ------------------------------
    # Run simulation
    # ------------------------------
    if sidebar.button("Run simulation"):

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

        # Physics engine keys:
        # "t", "ES_M", "Ered_M", "P_M", "O2_M", "H2O2_M",
        # "current_AU", "glucose_M", "glucose_mM"
        t = result["t"]
        P_mM = result["P_M"] * 1e3
        H2O2_mM = result["H2O2_M"] * 1e3
        O2_mM = result["O2_M"] * 1e3
        glucose_mM = result["glucose_mM"]
        current = result["current_AU"]

        # --- Film species ---
        st.subheader("Film species (GOx region)")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(t, P_mM, label="Gluconolactone (mM)", color="blue")
        ax1.plot(t, H2O2_mM, label="H₂O₂ (mM)", color="green")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("P, H₂O₂ (mM)")
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.plot(t, O2_mM, label="O₂ (mM)", color="red", linestyle="--")
        ax2.set_ylabel("O₂ (mM)", color="red")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        st.pyplot(fig1)

        # --- Glucose protocol ---
        st.subheader("Glucose input protocol")
        fig2, ax_glu = plt.subplots(figsize=(8, 3))
        ax_glu.plot(t, glucose_mM, color="purple")
        ax_glu.set_xlabel("Time (s)")
        ax_glu.set_ylabel("Glucose (mM)")
        ax_glu.grid(True)
        st.pyplot(fig2)

        # --- Current ---
        st.subheader("Amperometric current")
        fig3, ax_cur = plt.subplots(figsize=(8, 3))
        ax_cur.plot(t, current, color="black")
        ax_cur.set_xlabel("Time (s)")
        ax_cur.set_ylabel("Current (A.U.)")
        ax_cur.grid(True)
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
            "GOx_loading_U_per_electrode": E_units,
            "GOx_effective_E_tot_mM": E_tot_mM_sim,
            "Km_glu_mM": Km_glu_mM,
            "kcat_glu_s^-1": kcat_glu,
            "k1_M^-1_s^-1": k1,
            "km1_s^-1": km1,
            "k2_s^-1": k2,
            "k3_M^-1_s^-1": k3,
            "film_thickness_um": film_thickness_um,
            "O2_mode": O2_mode,
            "O2_0_ppm": O2_ppm,
            "O2_bath_ppm": O2_bath_ppm,
        })

        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            "gox_biosensor_timeseries.csv",
            "text/csv",
        )

# =========================================================
# PARAMETER SWEEP TAB
# =========================================================
with tabs[1]:

    sidebar.subheader("Sweep settings")

    # Glucose step protocol for sweep
    n_steps_sw = sidebar.slider("Number of glucose steps (sweep)", 1, 6, 4)
    step_duration_sw = sidebar.slider(
        "Step duration (s, sweep)", 50, 1000, 150, 50
    )

    default_concs_sw = [0, 4, 6, 8, 10, 12]
    glucose_steps_mM_sw = [
        sidebar.number_input(
            f"Sweep step {i+1} bulk glucose (mM)",
            min_value=0.0,
            max_value=100.0,
            value=float(default_concs_sw[i]),
            step=0.5,
            key=f"sw_glucose_step_{i}",
        )
        for i in range(n_steps_sw)
    ]

    # Sweep ranges
    E_units_min = sidebar.number_input(
        "Min GOx loading (U)", 0.01, 50.0, 0.1, 0.01
    )
    E_units_max = sidebar.number_input(
        "Max GOx loading (U)", 0.01, 50.0, 5.0, 0.01
    )
    n_E = sidebar.slider("Number of enzyme points", 3, 20, 8)

    O2_ppm_min = sidebar.selectbox(
        "Min initial O₂ (ppm)", [0, 1, 2, 3, 4, 5, 6], index=0
    )
    O2_ppm_max = sidebar.selectbox(
        "Max initial O₂ (ppm)", [0, 1, 2, 3, 4, 5, 6], index=6
    )
    n_O2 = sidebar.slider("Number of O₂ points", 3, 20, 8)

    O2_mode_sw = sidebar.selectbox(
        "Oxygen mode (sweep)", ["closed", "well-aerated"], index=0
    )
    O2_bath_ppm_sw = sidebar.selectbox(
        "Bath O₂ (ppm, sweep)", [0, 1, 2, 3, 4, 5, 6], index=6
    )

    film_thickness_um_sw = sidebar.slider(
        "Film thickness (µm, sweep)", 1.0, 200.0, 20.0, 1.0
    )

    Km_glu_mM_sw = sidebar.slider(
        "Glucose Km (mM, sweep)", 0.1, 50.0, 10.0, 0.5
    )
    kcat_glu_sw = sidebar.slider(
        "Glucose kcat (s⁻¹, sweep)", 0.01, 500.0, 100.0, 1.0
    )
    k3_sw = sidebar.slider(
        "O₂ reaction rate k3 (M⁻¹ s⁻¹, sweep)", 0.01, 1e4, 100.0, 1.0
    )

    Km_glu_M_sw = Km_glu_mM_sw * 1e-3
    km1_sw = 1.0
    k2_sw = kcat_glu_sw
    k1_sw = (km1_sw + k2_sw) / Km_glu_M_sw if Km_glu_M_sw > 0 else 0.0

    if sidebar.button("Run parameter sweep"):

        E_units_vals = np.linspace(E_units_min, E_units_max, n_E)
        O2_vals_ppm = np.linspace(O2_ppm_min, O2_ppm_max, n_O2)

        peak_signal = np.zeros((n_O2, n_E))

        for i, O2_ppm_val in enumerate(O2_vals_ppm):
            for j, E_units_val in enumerate(E_units_vals):

                E_tot_mM_sw = units_per_electrode_to_mM(
                    E_units=E_units_val,
                    film_thickness_um=film_thickness_um_sw,
                    kcat_s_inv=kcat_glu_sw,
                    electrode_area_mm2=ELECTRODE_AREA_MM2,
                )

                result_sw = run_gox_simulation(
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
                    n_points=800,
                )

                peak_signal[i, j] = np.max(result_sw["current_AU"])

        st.subheader("Peak current vs GOx loading and O₂")

        fig_sw, ax_sw = plt.subplots(figsize=(7, 5))
        im = ax_sw.imshow(
            peak_signal,
            aspect="auto",
            origin="lower",
            extent=[E_units_min, E_units_max, O2_ppm_min, O2_ppm_max],
        )
        ax_sw.set_xlabel("GOx loading (U per electrode)")
        ax_sw.set_ylabel("Initial O₂ (ppm)")
        ax_sw.set_title("Peak amperometric current (A.U.)")
        fig_sw.colorbar(im, ax=ax_sw, label="Peak current (A.U.)")
        st.pyplot(fig_sw)

        df_sweep = pd.DataFrame({
            "GOx_loading_U_per_electrode": np.repeat(E_units_vals, n_O2),
            "O2_0_ppm": np.tile(O2_vals_ppm, n_E),
            "peak_current_AU": peak_signal.flatten(),
            "Km_glu_mM": Km_glu_mM_sw,
            "kcat_glu_s^-1": kcat_glu_sw,
            "k1_M^-1_s^-1": k1_sw,
            "km1_s^-1": km1_sw,
            "k2_s^-1": k2_sw,
            "k3_M^-1_s^-1": k3_sw,
            "film_thickness_um": film_thickness_um_sw,
            "O2_mode": O2_mode_sw,
            "O2_bath_ppm": O2_bath_ppm_sw,
        })

        st.download_button(
            "Download sweep CSV",
            df_sweep.to_csv(index=False).encode("utf-8"),
            "gox_biosensor_sweep.csv",
            "text/csv",
        )