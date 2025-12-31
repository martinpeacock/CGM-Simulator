# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 18:53:17 2025

@author: martp
"""

# gox_biosensor_engine.py
"""
Core physics engine for a glucose oxidase (GOx) biosensor with immobilized enzyme.

Level A model:
- Bulk compartment (well-mixed): glucose_bulk, O2_bulk
- Film compartment on electrode: glucose_film, O2_film, ES, Ered, P, H2O2
- Enzyme is immobilized only in the film
- Mass transfer between bulk and film (reaction-diffusion lumped as k_mt)
- Oxygen specified in ppm (converted to mM)
- Amperometric current ~ d[H2O2_film]/dt

Geometry:
- Cylindrical electrode, diameter = 0.1 mm, length = 2 mm
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
    """
    return ppm * 0.03125


# ---------------------------------------------------------
# Default glucose protocol helper (bulk)
# ---------------------------------------------------------
def make_step_glucose_protocol(glucose_steps_mM, step_duration_s):
    """
    Create a stepwise bulk glucose concentration function S_bulk(t) [M].

    glucose_steps_mM : list of glucose concentrations in mM
    step_duration_s  : duration of each step in seconds

    Returns
    -------
    glucose_bulk_input : function(t) → S_bulk(t) in M
    t_end              : total simulation time in seconds
    """
    glucose_steps_mM = list(glucose_steps_mM)
    n_steps = len(glucose_steps_mM)
    t_end = n_steps * step_duration_s

    def glucose_bulk_input(t):
        if t < 0:
            return 0.0
        idx = int(t // step_duration_s)
        if idx < 0:
            idx = 0
        if idx >= n_steps:
            idx = n_steps - 1
        return glucose_steps_mM[idx] * 1e-3  # mM → M

    return glucose_bulk_input, t_end


# ---------------------------------------------------------
# Mechanistic GOx ODEs (film compartment only)
# ---------------------------------------------------------
def gox_bifilm_odes(
    t, y,
    k1, km1, k2, k3,
    E_tot_M,
    O2_mode,
    O2_bath_M,
    k_mt_glucose, k_mt_O2,
    V_bulk, V_film,
    A_electrode,
    glucose_bulk_input,
    O2_bulk_in_M
):
    """
    Two-compartment model:

    States (y):
      0: S_bulk   [M]   (glucose in bulk)
      1: O2_bulk  [M]   (O2 in bulk)
      2: S_film   [M]   (glucose in film near electrode)
      3: O2_film  [M]   (O2 in film near electrode)
      4: ES       [M]   (enzyme-substrate complex in film)
      5: Ered     [M]   (reduced enzyme in film)
      6: P        [M]   (product in film)
      7: H2O2     [M]   (H2O2 in film)

    Bulk:
      - No reaction, only mass transfer to film.
      - S_bulk is forced externally by glucose_bulk_input(t) (for simplicity, we can
        either treat it as a state or overwrite with forcing; here we treat it as
        a state relaxing toward input).

    Film:
      - Full GOx mechanism on S_film, O2_film.
      - Mass transfer with bulk based on k_mt and area A_electrode.
    """
    S_bulk, O2_bulk, S_film, O2_film, ES, Ered, P, H2O2 = y

    # External forcing for bulk glucose (target concentration)
    S_bulk_target = glucose_bulk_input(t)

    # Option: For O2 we can either let it evolve or relax toward an imposed bulk O2
    # Here: O2_bulk relaxes toward a fixed inlet value O2_bulk_in_M (e.g. set by ppm)
    O2_bulk_target = O2_bulk_in_M

    # Mass transfer terms (flux [mol/s] / volume → dC/dt)
    # Flux = k_mt * A * (C_bulk - C_film)
    # Contribution to dC/dt_bulk = -Flux / V_bulk
    # Contribution to dC/dt_film = +Flux / V_film

    # Glucose mass transfer
    J_S = k_mt_glucose * A_electrode * (S_bulk - S_film)  # mol/s
    dS_bulk_mt = -J_S / V_bulk
    dS_film_mt = +J_S / V_film

    # O2 mass transfer
    J_O2 = k_mt_O2 * A_electrode * (O2_bulk - O2_film)    # mol/s
    dO2_bulk_mt = -J_O2 / V_bulk
    dO2_film_mt = +J_O2 / V_film

    # Simple relaxation of bulk toward target (optional, mimics reservoir)
    k_relax_bulk = 0.01  # s^-1
    dS_bulk_relax = k_relax_bulk * (S_bulk_target - S_bulk)
    dO2_bulk_relax = k_relax_bulk * (O2_bulk_target - O2_bulk)

    # Total bulk dynamics
    dS_bulk = dS_bulk_mt + dS_bulk_relax
    dO2_bulk = dO2_bulk_mt + dO2_bulk_relax

    # Enzyme mass conservation in film
    Eox = E_tot_M - ES - Ered

    # GOx reaction in film
    dES   =  k1 * Eox * S_film - (km1 + k2) * ES
    dEred =  k2 * ES - k3 * Ered * O2_film
    dP    =  k2 * ES
    dO2_reac   = -k3 * Ered * O2_film
    dH2O2 =  k3 * Ered * O2_film

    # Oxygen mode for film O2
    dO2_film = dO2_film_mt + dO2_reac
    if O2_mode == "well-aerated":
        k_oxy = 0.01  # s^-1
        dO2_film += k_oxy * (O2_bath_M - O2_film)

    # Glucose in film: mass transfer + consumption
    # For now, glucose is only consumed via ES formation (irreversible consumption
    # in this simple picture: 1 S per ES -> P turnover).
    # Net S_film consumption ≈ k1 * Eox * S_film - km1*ES (binding/unbinding).
    # To avoid double-counting, we can treat the net consumption as k2*ES (product formation).
    # A simple choice: dS_film_reac = -k2 * ES
    dS_film_reac = -k2 * ES
    dS_film = dS_film_mt + dS_film_reac

    return [
        dS_bulk,
        dO2_bulk,
        dS_film,
        dO2_film,
        dES,
        dEred,
        dP,
        dH2O2,
    ]


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
    # geometry & mass transfer
    film_thickness_um=20.0,
    k_mt_glucose=1e-5,  # m/s
    k_mt_O2=1e-5,       # m/s
    V_bulk_mL=1.0,
):
    """
    Run a GOx biosensor simulation with bulk + immobilized film.

    Parameters
    ----------
    k1, km1, k2, k3 : float
        Kinetic parameters.
    E_tot_mM : float
        Total enzyme concentration in the film [mM].
    O2_mode : {"closed", "well-aerated"}
        Oxygen boundary condition mode for the film.
    O2_0_ppm : float
        Initial dissolved O2 in bulk in ppm.
    O2_bath_ppm : float
        Bath O2 in ppm for "well-aerated" mode (film).
    glucose_steps_mM : list of float or None
        Bulk glucose concentration steps [mM]. If None, a default protocol is used.
    step_duration_s : float
        Duration of each glucose step [s].
    n_points : int
        Number of time points for output.
    film_thickness_um : float
        Immobilized film thickness [µm].
    k_mt_glucose, k_mt_O2 : float
        Mass transfer coefficients [m/s] for glucose and O2.
    V_bulk_mL : float
        Effective bulk volume [mL].

    Returns
    -------
    result : dict
        Contains time series for bulk and film species and amperometric current.
    """
    # Default glucose steps in bulk
    if glucose_steps_mM is None:
        glucose_steps_mM = [0, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    glucose_bulk_input, t_end = make_step_glucose_protocol(glucose_steps_mM, step_duration_s)

    # Geometry: cylindrical electrode
    diameter_mm = 0.1
    length_mm = 2.0
    radius_m = (diameter_mm * 1e-3) / 2.0
    length_m = length_mm * 1e-3

    # Lateral surface area A ≈ π d L (ignore tips for now)
    A_electrode = np.pi * (diameter_mm * 1e-3) * length_m  # [m^2]

    # Film volume
    film_thickness_m = film_thickness_um * 1e-6
    V_film = A_electrode * film_thickness_m  # [m^3]

    # Bulk volume
    V_bulk = V_bulk_mL * 1e-6  # [m^3]  (1 mL = 1e-6 m^3)

    # Convert parameters
    E_tot_M = E_tot_mM * 1e-3
    O2_0_mM = ppm_to_mM(O2_0_ppm)
    O2_bath_mM = ppm_to_mM(O2_bath_ppm)
    O2_0_M = O2_0_mM * 1e-3
    O2_bath_M = O2_bath_mM * 1e-3

    # Initial conditions
    # Start with bulk glucose at first step value:
    S_bulk0 = glucose_bulk_input(0.0)
    O2_bulk0 = O2_0_M
    # Assume film initially equals bulk
    S_film0 = S_bulk0
    O2_film0 = O2_bulk0

    ES0 = 0.0
    Ered0 = 0.0
    P0 = 0.0
    H2O2_0 = 0.0

    y0 = [S_bulk0, O2_bulk0, S_film0, O2_film0, ES0, Ered0, P0, H2O2_0]

    # Time grid
    t_span = (0.0, t_end)
    t_eval = np.linspace(*t_span, n_points)

    # Fixed bulk O2 "reservoir" value (for relaxation target)
    O2_bulk_in_M = O2_0_M

    sol = solve_ivp(
        gox_bifilm_odes,
        t_span,
        y0,
        t_eval=t_eval,
        args=(
            k1, km1, k2, k3,
            E_tot_M,
            O2_mode,
            O2_bath_M,
            k_mt_glucose, k_mt_O2,
            V_bulk, V_film,
            A_electrode,
            glucose_bulk_input,
            O2_bulk_in_M,
        ),
    )

    t = sol.t
    S_bulk, O2_bulk, S_film, O2_film, ES, Ered, P, H2O2 = sol.y

    # Current ~ d[H2O2_film]/dt
    current = np.gradient(H2O2, t)

    # Glucose profiles
    glucose_bulk_M = np.array([glucose_bulk_input(tt) for tt in t])
    glucose_bulk_mM = glucose_bulk_M * 1e3
    S_film_mM = S_film * 1e3

    return {
        "t": t,
        "S_bulk_M": S_bulk,
        "O2_bulk_M": O2_bulk,
        "S_film_M": S_film,
        "O2_film_M": O2_film,
        "ES_M": ES,
        "Ered_M": Ered,
        "P_M": P,
        "H2O2_M": H2O2,
        "current_AU": current,
        "glucose_bulk_M": glucose_bulk_M,
        "glucose_bulk_mM": glucose_bulk_mM,
        "glucose_film_mM": S_film_mM,
        "A_electrode_m2": A_electrode,
        "V_film_m3": V_film,
        "V_bulk_m3": V_bulk,
    }


# ---------------------------------------------------------
# Optional: demo if run as a script
# ---------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    result = run_gox_simulation()

    t = result["t"]
    P_mM = result["P_M"] * 1e3
    H2O2_mM = result["H2O2_M"] * 1e3
    O2_film_mM = result["O2_film_M"] * 1e3
    glucose_bulk_mM = result["glucose_bulk_mM"]
    glucose_film_mM = result["glucose_film_mM"]
    current = result["current_AU"]

    # Film concentrations
    plt.figure(figsize=(9, 5))
    plt.plot(t, P_mM, label="P_film (mM)")
    plt.plot(t, H2O2_mM, label="H2O2_film (mM)")
    plt.plot(t, O2_film_mM, label="O2_film (mM)")
    plt.xlabel("Time (s)")
    plt.ylabel("Concentration (mM)")
    plt.title("Film species (immobilized GOx)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Bulk vs film glucose
    plt.figure(figsize=(9, 3))
    plt.plot(t, glucose_bulk_mM, label="Glucose_bulk (mM)", color="purple")
    plt.plot(t, glucose_film_mM, label="Glucose_film (mM)", color="orange", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Glucose (mM)")
    plt.title("Bulk vs film glucose")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Current
    plt.figure(figsize=(9, 3))
    plt.plot(t, current)
    plt.xlabel("Time (s)")
    plt.ylabel("Current (A.U.)")
    plt.title("Amperometric signal ~ d[H2O2_film]/dt")
    plt.grid(True)
    plt.tight_layout()
    plt.show()