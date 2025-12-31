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