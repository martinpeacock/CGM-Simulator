# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 18:01:36 2025

@author: martp
"""

# gox_biosensor_engine.py
"""
Core physics engine for a glucose oxidase (GOx) biosensor.

Features:
- Mechanistic Eox/ES/Ered + O2 + H2O2 model
- Stepwise glucose input (time-dependent forcing)
- Oxygen specified in ppm (converted to mM)
- Amperometric current ~ d[H2O2]/dt

This module is UI-agnostic and can be used from Streamlit, notebooks, or scripts.
"""

import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------
# Oxygen: ppm → mM conversion
# ---------------------------------------------------------
def ppm_to_mM(ppm):
    """
    Convert dissolved O2 from ppm to mM.
    Approx: 1 ppm O2 ~ 0.03125 mM.

    Parameters
    ----------
    ppm : float or array-like
        Dissolved O2 in ppm.

    Returns
    -------
    float or ndarray
        Dissolved O2 in mM.
    """
    return ppm * 0.03125


# ---------------------------------------------------------
# Default glucose protocol helper
# ---------------------------------------------------------
def make_step_glucose_protocol(glucose_steps_mM, step_duration_s):
    """
    Create a stepwise glucose concentration function S(t) [M].

    Parameters
    ----------
    glucose_steps_mM : list of floats
        Glucose concentrations for each step in mM.
        Example: [0, 4, 6, 8, 10, 12, 14]
    step_duration_s : float
        Duration of each step in seconds.

    Returns
    -------
    glucose_input : function
        Function(t) -> S(t) in M.
    t_end : float
        Total simulation time in seconds.
    """
    glucose_steps_mM = list(glucose_steps_mM)
    n_steps = len(glucose_steps_mM)
    t_end = n_steps * step_duration_s

    def glucose_input(t):
        if t < 0:
            return 0.0
        idx = int(t // step_duration_s)
        if idx < 0:
            idx = 0
        if idx >= n_steps:
            idx = n_steps - 1
        return glucose_steps_mM[idx] * 1e-3  # mM → M

    return glucose_input, t_end


# ---------------------------------------------------------
# Mechanistic GOx ODEs
# ---------------------------------------------------------
def gox_odes(t, y, k1, km1, k2, k3, E_tot, O2_mode, O2_bath_M, glucose_input):
    """
    Mechanistic GOx model:

    Eox + S <-> ES -> Ered + P
    Ered + O2 -> Eox + H2O2

    State vector
    ------------
    y = [ES, Ered, P, O2, H2O2]

    Parameters
    ----------
    k1, km1, k2, k3 : float
        Kinetic rate constants.
    E_tot : float
        Total enzyme concentration [M].
    O2_mode : {"closed", "well-aerated"}
        Oxygen boundary condition mode.
    O2_bath_M : float
        Bath O2 concentration [M] for "well-aerated" mode.
    glucose_input : function
        Function(t) -> S(t) [M].

    Returns
    -------
    dydt : list of float
        Time derivatives of state variables.
    """
    ES, Ered, P, O2, H2O2 = y

    S = glucose_input(t)

    # Mass conservation of enzyme
    Eox = E_tot - ES - Ered

    # Base ODEs
    dES   =  k1 * Eox * S - (km1 + k2) * ES
    dEred =  k2 * ES - k3 * Ered * O2
    dP    =  k2 * ES
    dO2   = -k3 * Ered * O2
    dH2O2 =  k3 * Ered * O2

    # Oxygen mode
    if O2_mode == "well-aerated":
        k_oxy = 0.01  # s^-1, relaxation rate to bath
        dO2 += k_oxy * (O2_bath_M - O2)

    return [dES, dEred, dP, dO2, dH2O2]


# ---------------------------------------------------------
# High-level simulation wrapper
# ---------------------------------------------------------
def run_gox_simulation(
    k1=1.0,
    km1=0.5,
    k2=1.0,
    k3=1.0,
    E_tot_mM=0.1,
    O2_mode="closed",
    O2_0_ppm=6.0,
    O2_bath_ppm=6.0,
    glucose_steps_mM=None,
    step_duration_s=150.0,
    n_points=2000,
):
    """
    Run a GOx biosensor simulation with stepwise glucose and O2 in ppm.

    Parameters
    ----------
    k1, km1, k2, k3 : float
        Kinetic parameters.
    E_tot_mM : float
        Total enzyme concentration [mM].
    O2_mode : {"closed", "well-aerated"}
        Oxygen boundary condition mode.
    O2_0_ppm : float
        Initial dissolved O2 in ppm.
    O2_bath_ppm : float
        Bath O2 in ppm for "well-aerated" mode.
    glucose_steps_mM : list of float or None
        Glucose concentration steps [mM]. If None, a default protocol is used.
    step_duration_s : float
        Duration of each glucose step [s].
    n_points : int
        Number of time points for output.

    Returns
    -------
    result : dict
        {
          "t": time array [s],
          "ES_M": [M],
          "Ered_M": [M],
          "P_M": [M],
          "O2_M": [M],
          "H2O2_M": [M],
          "current_AU": d[H2O2]/dt [M/s, a.u.],
          "glucose_M": [M],
          "glucose_mM": [mM],
        }
    """
    # Default glucose steps if none provided
    if glucose_steps_mM is None:
        glucose_steps_mM = [0, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    # Build glucose input and time span
    glucose_input, t_end = make_step_glucose_protocol(glucose_steps_mM, step_duration_s)

    # Convert parameters
    E_tot_M = E_tot_mM * 1e-3
    O2_0_mM = ppm_to_mM(O2_0_ppm)
    O2_bath_mM = ppm_to_mM(O2_bath_ppm)
    O2_0_M = O2_0_mM * 1e-3
    O2_bath_M = O2_bath_mM * 1e-3

    # Initial conditions
    ES0 = 0.0
    Ered0 = 0.0
    P0 = 0.0
    H2O2_0 = 0.0
    y0 = [ES0, Ered0, P0, O2_0_M, H2O2_0]

    # Time grid
    t_span = (0.0, t_end)
    t_eval = np.linspace(*t_span, n_points)

    # Solve ODEs
    sol = solve_ivp(
        gox_odes,
        t_span,
        y0,
        t_eval=t_eval,
        args=(k1, km1, k2, k3, E_tot_M, O2_mode, O2_bath_M, glucose_input),
    )

    t = sol.t
    ES, Ered, P, O2, H2O2 = sol.y

    # Current ~ d[H2O2]/dt
    current = np.gradient(H2O2, t)

    # Glucose profile
    glucose_M = np.array([glucose_input(tt) for tt in t])
    glucose_mM = glucose_M * 1e3

    return {
        "t": t,
        "ES_M": ES,
        "Ered_M": Ered,
        "P_M": P,
        "O2_M": O2,
        "H2O2_M": H2O2,
        "current_AU": current,
        "glucose_M": glucose_M,
        "glucose_mM": glucose_mM,
    }


# ---------------------------------------------------------
# Optional: simple demo if run as a script
# ---------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    result = run_gox_simulation()

    t = result["t"]
    P_mM = result["P_M"] * 1e3
    H2O2_mM = result["H2O2_M"] * 1e3
    O2_mM = result["O2_M"] * 1e3
    glucose_mM = result["glucose_mM"]
    current = result["current_AU"]

    # Concentrations
    plt.figure(figsize=(9, 5))
    plt.plot(t, P_mM, label="P (mM)")
    plt.plot(t, H2O2_mM, label="H2O2 (mM)")
    plt.plot(t, O2_mM, label="O2 (mM)")
    plt.xlabel("Time (s)")
    plt.ylabel("Concentration (mM)")
    plt.title("GOx biosensor physics engine: concentrations")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Glucose protocol
    plt.figure(figsize=(9, 3))
    plt.plot(t, glucose_mM, color="purple")
    plt.xlabel("Time (s)")
    plt.ylabel("Glucose (mM)")
    plt.title("Glucose input protocol")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Current
    plt.figure(figsize=(9, 3))
    plt.plot(t, current)
    plt.xlabel("Time (s)")
    plt.ylabel("Current (A.U.)")
    plt.title("Amperometric signal ~ d[H2O2]/dt")
    plt.grid(True)
    plt.tight_layout()
    plt.show()