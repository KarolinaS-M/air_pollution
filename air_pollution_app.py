import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

st.set_page_config(page_title="Stochastic Air Pollution Simulator", layout="wide")

st.title("Stochastic Air Pollution Modeling – Random Walk Simulation")

# =========================
# Sidebar – parameters
# =========================

st.sidebar.header("Model Parameters")

scenario = st.sidebar.selectbox(
    "Scenario",
    ["Basic", "Wind", "Wind + Deposition", "Compare all"]
)

wind_strength = st.sidebar.slider("Wind strength", 0.0, 2.0, 0.8, 0.1)
wind_angle = st.sidebar.slider("Wind direction (degrees)", 0, 360, 15)

deposition_prob = st.sidebar.slider("Deposition probability", 0.0, 0.05, 0.01, 0.001)

n_steps = st.sidebar.slider("Time steps", 50, 400, 200)
emission_rate = st.sidebar.slider("Particles per step", 10, 100, 40)

step_size = 1.0

# =========================
# Wind vector
# =========================

theta = np.deg2rad(wind_angle)
wind = (wind_strength * np.cos(theta), wind_strength * np.sin(theta))

# =========================
# Simulation function
# =========================

def run_simulation(wind, deposition):

    rng = np.random.default_rng(1)
    x, y = np.array([]), np.array([])
    alive = np.array([], dtype=bool)

    Xhist, Yhist, counts = [], [], []

    for _ in range(n_steps):

        x = np.concatenate([x, np.zeros(emission_rate)])
        y = np.concatenate([y, np.zeros(emission_rate)])
        alive = np.concatenate([alive, np.ones(emission_rate, bool)])

        if deposition > 0:
            dead = (rng.random(alive.size) < deposition) & alive
            alive[dead] = False

        idx = np.where(alive)[0]
        if idx.size:
            x[idx] += rng.choice([-step_size, step_size], idx.size) + wind[0]
            y[idx] += rng.choice([-step_size, step_size], idx.size) + wind[1]

        Xhist.append(x.copy())
        Yhist.append(y.copy())
        counts.append(alive.sum())

    return Xhist, Yhist, counts

# =========================
# Visualization loop
# =========================

def animate(X, Y, counts, title):

    cloud_area = st.empty()
    count_chart = st.empty()

    history = []

    for t in range(len(X)):

        xa = X[t][:]
        ya = Y[t][:]

        fig, ax = plt.subplots(figsize=(5,5))
        ax.scatter(xa, ya, s=6, alpha=0.35)

        if len(xa):
            ax.scatter(np.mean(xa), np.mean(ya), color="red", s=40)

        ax.set_xlim(-250, 250)
        ax.set_ylim(-250, 250)
        ax.set_aspect("equal")
        ax.set_title(f"{title} – step {t}")

        cloud_area.pyplot(fig)
        plt.close(fig)

        history.append(counts[t])
        df = pd.DataFrame({"time": range(len(history)), "airborne": history})
        count_chart.line_chart(df.set_index("time"))

        time.sleep(0.04)

# =========================
# Run scenarios
# =========================

if st.button("Run simulation"):

    if scenario == "Basic":
        X, Y, counts = run_simulation((0,0), 0.0)
        animate(X, Y, counts, "Basic diffusion")

    elif scenario == "Wind":
        X, Y, counts = run_simulation(wind, 0.0)
        animate(X, Y, counts, "Wind-driven diffusion")

    elif scenario == "Wind + Deposition":
        X, Y, counts = run_simulation(wind, deposition_prob)
        animate(X, Y, counts, "Wind + deposition")

    else:
        st.subheader("Scenario comparison")

        X1, Y1, c1 = run_simulation((0,0), 0.0)
        X2, Y2, c2 = run_simulation(wind, 0.0)
        X3, Y3, c3 = run_simulation(wind, deposition_prob)

        df = pd.DataFrame({
            "Basic": c1,
            "Wind": c2,
            "Wind + Deposition": c3
        })

        st.line_chart(df)

        st.success("Comparison completed")

# =========================
# Save results
# =========================

if st.button("Save last simulation"):

    df = pd.DataFrame({
        "time": range(len(counts)),
        "airborne_particles": counts
    })

    df.to_csv("pollution_simulation.csv", index=False)

    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False),
        file_name="pollution_simulation.csv",
        mime="text/csv"
    )