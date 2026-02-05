import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

st.set_page_config(layout="wide")
st.title("Stochastic Air Pollution Model – Random Walk with Drift and Deposition")

# =====================
# Sidebar
# =====================

scenario = st.sidebar.selectbox(
    "Scenario",
    ["Basic", "Wind", "Wind + Deposition", "Compare"]
)

wind_strength = st.sidebar.slider("Wind strength", 0.0, 2.0, 0.8)
wind_angle = st.sidebar.slider("Wind direction (deg)", 0, 360, 15)
deposition_prob = st.sidebar.slider("Deposition probability", 0.0, 0.05, 0.01)

n_steps = st.sidebar.slider("Time steps", 50, 300, 200)
emission_rate = st.sidebar.slider("Particles per step", 10, 80, 40)

theta = np.deg2rad(wind_angle)
wind = (wind_strength*np.cos(theta), wind_strength*np.sin(theta))

# =====================
# Simulation function
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

    cols = ["mean_x","mean_y","var_x","var_y","std_x","std_y","count"]
    return Xhist, Yhist, pd.DataFrame(stats, columns=cols)

# =====================
# Educational comment
# =====================

st.info(
    "Note: In the Basic and Wind scenarios the number of airborne particles grows identically, "
    "because wind affects spatial displacement but not population size. Only deposition introduces "
    "an absorption mechanism that reduces the number of particles over time."
)

# =====================
# Run simulation
# =====================

if st.button("Run simulation"):

    if scenario != "Compare":
        X, Y, df = simulate(wind if scenario!="Basic" else (0,0),
                            deposition_prob if scenario=="Wind + Deposition" else 0)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Particle cloud evolution")
            area = st.empty()

        with col2:
            st.subheader("Statistics in time")
            st.line_chart(df[["var_x","var_y"]])

        for t in range(len(X)):

            fig, ax = plt.subplots(figsize=(3,3))
            ax.scatter(X[t], Y[t], s=6, alpha=0.4)
            ax.scatter(df.mean_x[t], df.mean_y[t], color="red", s=25)

            ax.set_xlim(-200,200)
            ax.set_ylim(-200,200)
            ax.set_aspect("equal")
            area.pyplot(fig)
            plt.close(fig)

            time.sleep(0.04)

        st.subheader("Final distribution comparison")

        fig, ax = plt.subplots(1,2, figsize=(6,3))
        ax[0].hist(X[-1], bins=30)
        ax[0].set_title("X-direction")

        ax[1].hist(Y[-1], bins=30)
        ax[1].set_title("Y-direction")

        st.pyplot(fig)

        st.subheader("Final moments")
        st.dataframe(df.tail(1))

    else:
        st.subheader("Airborne particle count comparison")

        _, _, d1 = simulate((0,0),0)
        _, _, d2 = simulate(wind,0)
        _, _, d3 = simulate(wind,deposition_prob)

        st.line_chart(pd.DataFrame({
            "Basic": d1["count"],
            "Wind": d2["count"],
            "Wind + Deposition": d3["count"]
        }))