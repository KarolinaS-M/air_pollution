import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

st.set_page_config(layout="wide")
st.title("Stochastic Air Pollution Model – Random Walk Simulation")

# =========================
# Model scaling explanation
# =========================

st.markdown(
"""
**Model scaling assumptions**

• One simulation time step corresponds to 10 minutes  
• One spatial unit corresponds to approximately 11 meters  

The wind strength parameter represents the deterministic drift per time step of the random walk.  
Under this scaling, a value of 0.8 corresponds to approximately 9.1 meters of systematic displacement per simulation step.
"""
)

# =====================
# Sidebar – parameters
# =====================

scenario = st.sidebar.selectbox(
    "Scenario",
    ["Basic", "Wind", "Wind + Deposition", "Compare"]
)

wind_strength = st.sidebar.slider("Wind strength (drift per step)", 0.0, 2.0, 0.8)
wind_angle = st.sidebar.slider("Wind direction (degrees)", 0, 360, 15)
deposition_prob = st.sidebar.slider("Deposition probability", 0.0, 0.05, 0.01)

n_steps = st.sidebar.slider("Time steps", 50, 300, 200)
emission_rate = st.sidebar.slider("Particles per step", 10, 80, 40)

theta = np.deg2rad(wind_angle)
wind = (wind_strength * np.cos(theta), wind_strength * np.sin(theta))

# =====================
# Educational comment
# =====================

st.info(
    "In the Basic and Wind scenarios, the number of airborne particles evolves identically because wind "
    "affects spatial displacement but not population size. Only deposition introduces an absorption "
    "mechanism that reduces the number of particles over time."
)

# =====================
# Simulation
# =====================

def simulate(wind, deposition):

    rng = np.random.default_rng(1)
    x, y = np.array([]), np.array([])
    alive = np.array([], dtype=bool)

    Xhist, Yhist, stats = [], [], []

    for _ in range(n_steps):

        x = np.concatenate([x, np.zeros(emission_rate)])
        y = np.concatenate([y, np.zeros(emission_rate)])
        alive = np.concatenate([alive, np.ones(emission_rate, bool)])

        if deposition > 0:
            dead = (rng.random(alive.size) < deposition) & alive
            alive[dead] = False

        idx = np.where(alive)[0]
        if idx.size:
            x[idx] += rng.choice([-1, 1], idx.size) + wind[0]
            y[idx] += rng.choice([-1, 1], idx.size) + wind[1]

        xa, ya = x[alive], y[alive]

        stats.append([
            xa.mean() if len(xa) else 0,
            ya.mean() if len(ya) else 0,
            xa.var() if len(xa) else 0,
            ya.var() if len(ya) else 0,
            xa.std() if len(xa) else 0,
            ya.std() if len(ya) else 0,
            len(xa)
        ])

        Xhist.append(xa.copy())
        Yhist.append(ya.copy())

    cols = ["mean_x", "mean_y", "var_x", "var_y", "std_x", "std_y", "count"]
    return Xhist, Yhist, pd.DataFrame(stats, columns=cols)

# =====================
# Run simulation
# =====================

if st.button("Run simulation"):

    if scenario != "Compare":

        wind_used = (0,0) if scenario == "Basic" else wind
        dep_used = deposition_prob if scenario == "Wind + Deposition" else 0

        X, Y, df = simulate(wind_used, dep_used)

        col_cloud, col_stats = st.columns([2,1])

        with col_cloud:
            st.subheader("Particle cloud evolution")
            cloud_box = st.empty()

        with col_stats:
            st.subheader("Mean position")
            mean_box = st.empty()
            st.subheader("Variance")
            var_box = st.empty()
            st.subheader("Standard deviation")
            std_box = st.empty()

        for t in range(len(X)):

            fig, ax = plt.subplots(figsize=(2.3,2.3))
            ax.scatter(X[t], Y[t], s=6, alpha=0.4)
            ax.scatter(df.mean_x[t], df.mean_y[t], color="red", s=20)

            ax.set_xlim(-200, 200)
            ax.set_ylim(-200, 200)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)
            ax.set_title(f"Step {t}", fontsize=8)

            cloud_box.pyplot(fig)
            plt.close(fig)

            mean_box.line_chart(df[["mean_x", "mean_y"]].iloc[:t+1])
            var_box.line_chart(df[["var_x", "var_y"]].iloc[:t+1])
            std_box.line_chart(df[["std_x", "std_y"]].iloc[:t+1])

            time.sleep(0.04)

        # =====================
        # Final moments
        # =====================

        st.subheader("Final moments")
        st.dataframe(df.tail(1))

        # =====================
        # Histograms
        # =====================

        st.subheader("Final distributions")

        fig, ax = plt.subplots(1, 2, figsize=(9, 2))

        ax[0].hist(X[-1], bins=30)
        ax[0].set_title("X-direction", fontsize=10)

        ax[1].hist(Y[-1], bins=30)
        ax[1].set_title("Y-direction", fontsize=10)

        for a in ax:
            a.tick_params(axis='both', labelsize=7)

        st.pyplot(fig)

    else:
        st.subheader("Airborne particle count comparison")

        _, _, d1 = simulate((0,0), 0)
        _, _, d2 = simulate(wind, 0)
        _, _, d3 = simulate(wind, deposition_prob)

        st.line_chart(pd.DataFrame({
            "Basic": d1["count"],
            "Wind": d2["count"],
            "Wind + Deposition": d3["count"]
        }))