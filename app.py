import math
from typing import Callable, Dict, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.interpolate import CubicSpline


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Feynman Path Integral Visualizer",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
    .main-title {font-size: 2.3rem; font-weight: 700; margin-bottom: 0.2rem;}
    .subtitle {font-size: 1.05rem; color: #5b6470; margin-bottom: 1.2rem;}
    .soft-card {
        background: linear-gradient(180deg, rgba(240,244,255,0.92), rgba(248,250,255,0.92));
        padding: 1rem 1rem 0.8rem 1rem;
        border-radius: 1rem;
        border: 1px solid rgba(120,140,180,0.18);
        margin-bottom: 1rem;
    }
    .small-muted {color: #68707c; font-size: 0.95rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">Feynman Path Integral Visualizer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">A teaching app for seeing how quantum amplitudes emerge from many possible paths — and why the classical path still matters.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="soft-card">
    <b>What this app shows</b><br>
    Instead of assuming that a particle follows a single trajectory, the path-integral picture adds contributions from many trajectories connecting the same endpoints. 
    Each path contributes a complex phase \(e^{iS/\hbar}\), where \(S\) is the action. 
    This app is built as a <i>didactic visualization</i>: it samples many discretized paths, computes their actions in a vectorized way with <code>numpy</code> and <code>scipy</code>, and shows how constructive and destructive interference arise.
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Theory block
# -----------------------------
with st.expander("Theory, intuition, and references", expanded=False):
    st.markdown(
        r"""
### Core idea
For fixed endpoints \((x_a,0)\) and \((x_b,T)\), the propagator can be written formally as

$$
K(x_b,T;x_a,0)=\int \mathcal{D}[x(t)]\, e^{iS[x]/\hbar}
$$

with the action

$$
S[x]=\int_0^T L(x,\dot x,t)\,dt,
\qquad
L = \tfrac12 m\dot x^2 - V(x).
$$

The key point for students is this: **quantum motion is not described by one path, but by interference among many paths.**

### Why the classical path still appears
The classical trajectory is special because it makes the action stationary:

$$
\delta S = 0.
$$

Near that path, the phase \(S/\hbar\) changes slowly, so nearby contributions tend to add coherently. Far from it, phases fluctuate rapidly and cancel. This is the idea of **stationary phase**.

### What this app computes numerically
This app does not evaluate the formal continuum path integral exactly. Instead, it:
- discretizes time into many small steps,
- generates large batches of trial paths in a vectorized way,
- computes their discrete actions,
- converts those actions into complex amplitudes,
- shows how the cumulative sum behaves.

This makes the physics accessible without hiding the core mathematics.

### The panels
- **Free particle:** the classical path is a straight line.
- **Harmonic oscillator:** the potential bends the preferred paths.
- **Two path families:** a double-slit-style intuition panel showing how two bundles of paths interfere.
- **Real time vs imaginary time:** why oscillatory factors are hard numerically, and why Euclidean-time methods are so useful.

### References
1. R. P. Feynman and A. R. Hibbs, *Quantum Mechanics and Path Integrals*.
2. L. S. Schulman, *Techniques and Applications of Path Integration*.
3. H. Kleinert, *Path Integrals in Quantum Mechanics, Statistics, Polymer Physics, and Financial Markets*.
4. D. J. Griffiths and D. F. Schroeter, *Introduction to Quantum Mechanics*.
5. M. Chaichian and A. Demichev, *Path Integrals in Physics*.
        """
    )


# -----------------------------
# Sidebar controls
# -----------------------------
section = st.sidebar.radio(
    "Choose a panel",
    [
        "Free particle",
        "Harmonic oscillator",
        "Two path families",
        "Real time vs imaginary time",
    ],
)

seed = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=7, step=1)
show_count = st.sidebar.slider("Number of sample paths shown", min_value=10, max_value=120, value=35, step=5)


# -----------------------------
# Numerical helpers
# -----------------------------
def free_classical_path(t: np.ndarray, xa: float, xb: float, T: float) -> np.ndarray:
    return xa + (xb - xa) * (t / T)


def ho_classical_path(t: np.ndarray, xa: float, xb: float, T: float, omega: float) -> np.ndarray:
    s = np.sin(omega * T)
    if abs(s) < 1e-4:
        return free_classical_path(t, xa, xb, T)
    return xa * np.cos(omega * t) + ((xb - xa * np.cos(omega * T)) / s) * np.sin(omega * t)


def potential_free(x: np.ndarray, m: float, omega: float = 0.0) -> np.ndarray:
    return np.zeros_like(x)


def potential_ho(x: np.ndarray, m: float, omega: float) -> np.ndarray:
    return 0.5 * m * omega**2 * x**2


def make_reference_family(
    t: np.ndarray, xa: float, xb: float, T: float, bump: float, sign: float
) -> np.ndarray:
    base = free_classical_path(t, xa, xb, T)
    return base + sign * bump * np.sin(np.pi * t / T)


def generate_paths_from_reference(
    t: np.ndarray,
    x_ref: np.ndarray,
    n_paths: int,
    sigma: float,
    n_ctrl: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    T = t[-1]
    tau = t / T
    ctrl_tau = np.linspace(0.0, 1.0, n_ctrl)
    ref_ctrl = np.interp(ctrl_tau, tau, x_ref)
    noise = rng.normal(scale=sigma, size=(n_paths, n_ctrl))
    noise[:, 0] = 0.0
    noise[:, -1] = 0.0
    ctrl = ref_ctrl[None, :] + noise
    spline = CubicSpline(ctrl_tau, ctrl, axis=1, bc_type="clamped")
    return spline(tau)


def compute_actions(
    t: np.ndarray,
    paths: np.ndarray,
    m: float,
    V: Callable[[np.ndarray, float, float], np.ndarray],
    omega: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt = t[1] - t[0]
    dx = np.diff(paths, axis=1)
    v_mid = dx / dt
    x_mid = 0.5 * (paths[:, 1:] + paths[:, :-1])
    kinetic = 0.5 * m * v_mid**2
    potential = V(x_mid, m, omega)
    lagrangian = kinetic - potential
    action = dt * np.sum(lagrangian, axis=1)
    euclidean_action = dt * np.sum(kinetic + potential, axis=1)
    mean_x = dt * np.sum(x_mid, axis=1) / (t[-1] - t[0])
    return action, euclidean_action, mean_x


def phase_data(actions: np.ndarray, hbar_eff: float, ref_action: float) -> Dict[str, np.ndarray]:
    delta = actions - ref_action
    amps = np.exp(1j * delta / hbar_eff)
    sort_idx = np.argsort(np.abs(delta))
    walk = np.cumsum(amps[sort_idx]) / np.arange(1, len(amps) + 1)
    return {
        "delta": delta,
        "amps": amps,
        "sort_idx": sort_idx,
        "walk": walk,
    }


def trajectory_density(t: np.ndarray, paths: np.ndarray, bins_t: int = 90, bins_x: int = 90):
    tt = np.broadcast_to(t[None, :], paths.shape).ravel()
    xx = paths.ravel()
    hist, t_edges, x_edges = np.histogram2d(tt, xx, bins=[bins_t, bins_x])
    return hist.T, t_edges, x_edges


def make_trajectory_figure(
    t: np.ndarray,
    paths: np.ndarray,
    x_ref: np.ndarray,
    sample_count: int,
    title: str,
) -> go.Figure:
    fig = go.Figure()
    n = min(sample_count, paths.shape[0])
    step = max(paths.shape[0] // n, 1)
    sample = paths[::step][:n]
    for row in sample:
        fig.add_trace(
            go.Scatter(
                x=t,
                y=row,
                mode="lines",
                line=dict(width=1),
                opacity=0.18,
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=x_ref,
            mode="lines",
            line=dict(width=4),
            name="Reference / classical path",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Time t",
        yaxis_title="Position x(t)",
        margin=dict(l=20, r=20, t=45, b=20),
        height=380,
    )
    return fig


def make_density_figure(t: np.ndarray, paths: np.ndarray, x_ref: np.ndarray, title: str) -> go.Figure:
    hist, t_edges, x_edges = trajectory_density(t, paths)
    tc = 0.5 * (t_edges[:-1] + t_edges[1:])
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=tc,
            y=xc,
            z=hist,
            colorbar_title="Counts",
        )
    )
    fig.add_trace(
        go.Scatter(x=t, y=x_ref, mode="lines", line=dict(width=3, color="white"), name="Reference path")
    )
    fig.update_layout(
        title=title,
        xaxis_title="Time t",
        yaxis_title="Position x(t)",
        margin=dict(l=20, r=20, t=45, b=20),
        height=380,
    )
    return fig


def make_action_histogram(delta_actions: np.ndarray, hbar_eff: float, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=delta_actions / hbar_eff, nbinsx=55, name="ΔS/ħ"))
    fig.update_layout(
        title=title,
        xaxis_title="(S - S_ref) / ħ",
        yaxis_title="Number of paths",
        bargap=0.06,
        margin=dict(l=20, r=20, t=45, b=20),
        height=360,
    )
    return fig


def make_phasor_figure(amps: np.ndarray, walk: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.real(amps),
            y=np.imag(amps),
            mode="markers",
            marker=dict(size=6),
            opacity=0.35,
            name="Individual phases",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.real(walk),
            y=np.imag(walk),
            mode="lines+markers",
            line=dict(width=3),
            marker=dict(size=4),
            name="Cumulative average",
        )
    )
    circle_theta = np.linspace(0, 2 * np.pi, 300)
    fig.add_trace(
        go.Scatter(
            x=np.cos(circle_theta),
            y=np.sin(circle_theta),
            mode="lines",
            line=dict(width=1, dash="dash"),
            name="Unit circle",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Re",
        yaxis_title="Im",
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
        margin=dict(l=20, r=20, t=45, b=20),
        height=360,
    )
    return fig


def metrics_box(ref_action: float, phase_info: Dict[str, np.ndarray], label: str):
    walk = phase_info["walk"]
    coherence = np.abs(np.mean(phase_info["amps"]))
    final_mag = np.abs(walk[-1])
    col1, col2, col3 = st.columns(3)
    col1.metric(f"{label}: reference action", f"{ref_action:.3f}")
    col2.metric("|mean phase factor|", f"{coherence:.3f}")
    col3.metric("Final cumulative magnitude", f"{final_mag:.3f}")


@st.cache_data(show_spinner=False)
def cached_family_data(
    model: str,
    xa: float,
    xb: float,
    T: float,
    m: float,
    omega: float,
    n_steps: int,
    n_paths: int,
    sigma: float,
    n_ctrl: int,
    seed: int,
):
    t = np.linspace(0.0, T, n_steps)
    if model == "free":
        x_ref = free_classical_path(t, xa, xb, T)
        V = potential_free
    else:
        x_ref = ho_classical_path(t, xa, xb, T, omega)
        V = potential_ho
    paths = generate_paths_from_reference(t, x_ref, n_paths, sigma, n_ctrl, seed)
    action, euc_action, mean_x = compute_actions(t, paths, m, V, omega)
    ref_action, ref_euc, _ = compute_actions(t, x_ref[None, :], m, V, omega)
    phase_info = phase_data(action, hbar_eff=1.0, ref_action=ref_action[0])
    return {
        "t": t,
        "x_ref": x_ref,
        "paths": paths,
        "action": action,
        "euc_action": euc_action,
        "ref_action": ref_action[0],
        "ref_euc_action": ref_euc[0],
        "mean_x": mean_x,
        "phase_info": phase_info,
    }


@st.cache_data(show_spinner=False)
def cached_two_family_data(
    xa: float,
    xb: float,
    T: float,
    m: float,
    n_steps: int,
    n_paths_each: int,
    sigma: float,
    n_ctrl: int,
    bump: float,
    hbar_eff: float,
    seed: int,
):
    t = np.linspace(0.0, T, n_steps)
    top_ref = make_reference_family(t, xa, xb, T, bump, +1.0)
    bot_ref = make_reference_family(t, xa, xb, T, bump, -1.0)

    top_paths = generate_paths_from_reference(t, top_ref, n_paths_each, sigma, n_ctrl, seed)
    bot_paths = generate_paths_from_reference(t, bot_ref, n_paths_each, sigma, n_ctrl, seed + 1000)

    top_action, _, _ = compute_actions(t, top_paths, m, potential_free, 0.0)
    bot_action, _, _ = compute_actions(t, bot_paths, m, potential_free, 0.0)
    top_ref_action, _, _ = compute_actions(t, top_ref[None, :], m, potential_free, 0.0)
    bot_ref_action, _, _ = compute_actions(t, bot_ref[None, :], m, potential_free, 0.0)

    top_amp = np.exp(1j * (top_action - top_ref_action[0]) / hbar_eff)
    bot_amp = np.exp(1j * (bot_action - bot_ref_action[0]) / hbar_eff)
    total_amp = np.mean(top_amp) + np.mean(bot_amp)

    return {
        "t": t,
        "top_ref": top_ref,
        "bot_ref": bot_ref,
        "top_paths": top_paths,
        "bot_paths": bot_paths,
        "top_action": top_action,
        "bot_action": bot_action,
        "top_amp": top_amp,
        "bot_amp": bot_amp,
        "total_amp": total_amp,
        "top_ref_action": top_ref_action[0],
        "bot_ref_action": bot_ref_action[0],
    }


# -----------------------------
# Panel: Free particle
# -----------------------------
if section == "Free particle":
    with st.expander("How to use this panel", expanded=True):
        st.markdown(
            """
- Set the endpoints and the total travel time.
- Increase the path spread to sample wilder trajectories around the classical straight line.
- Decrease the effective \(\hbar\) to make stationary phase sharper.
- Compare the path cloud with the cumulative phasor plot: when phases cancel strongly, the cumulative average stays small.
            """
        )

    c1, c2, c3 = st.columns(3)
    xa = c1.slider("Initial position x_a", -2.0, 2.0, -0.9, 0.1)
    xb = c2.slider("Final position x_b", -2.0, 2.0, 0.9, 0.1)
    T = c3.slider("Total time T", 0.5, 5.0, 2.0, 0.1)

    c4, c5, c6, c7 = st.columns(4)
    m = c4.slider("Mass m", 0.2, 3.0, 1.0, 0.1)
    hbar_eff = c5.slider("Effective ħ", 0.05, 1.0, 0.25, 0.01)
    sigma = c6.slider("Path spread", 0.02, 1.2, 0.25, 0.01)
    n_paths = c7.slider("Number of sampled paths", 200, 3000, 900, 100)

    n_steps = 220
    n_ctrl = 11
    data = cached_family_data("free", xa, xb, T, m, 0.0, n_steps, n_paths, sigma, n_ctrl, seed)
    phase_info = phase_data(data["action"], hbar_eff, data["ref_action"])
    metrics_box(data["ref_action"], phase_info, "Free particle")

    col1, col2 = st.columns(2)
    col1.plotly_chart(
        make_trajectory_figure(data["t"], data["paths"], data["x_ref"], show_count, "Sampled paths around the classical straight line"),
        use_container_width=True,
    )
    col2.plotly_chart(
        make_density_figure(data["t"], data["paths"], data["x_ref"], "Path density in the (t, x) plane"),
        use_container_width=True,
    )

    col3, col4 = st.columns(2)
    col3.plotly_chart(
        make_action_histogram(phase_info["delta"], hbar_eff, "How far the sampled actions are from the classical one"),
        use_container_width=True,
    )
    col4.plotly_chart(
        make_phasor_figure(phase_info["amps"], phase_info["walk"], "Interference of phase factors"),
        use_container_width=True,
    )

    st.caption(
        "Interpretation: the classical straight line is not the only path. It is the path around which nearby paths tend to keep similar phases, so their contributions add more coherently."
    )


# -----------------------------
# Panel: Harmonic oscillator
# -----------------------------
elif section == "Harmonic oscillator":
    with st.expander("How to use this panel", expanded=True):
        st.markdown(
            """
- Change the oscillator frequency to see how the preferred path bends.
- Compare the free-particle intuition with the confining potential.
- Watch how the action distribution and phase cloud change when the trial paths become broader.
- If \(\sin(\omega T)\) is very small, the exact endpoint-conditioned classical path becomes numerically delicate; the app then smoothly falls back to a linear reference path.
            """
        )

    c1, c2, c3 = st.columns(3)
    xa = c1.slider("Initial position x_a", -2.0, 2.0, -0.7, 0.1)
    xb = c2.slider("Final position x_b", -2.0, 2.0, 0.7, 0.1)
    T = c3.slider("Total time T", 0.5, 6.0, 2.2, 0.1)

    c4, c5, c6, c7, c8 = st.columns(5)
    m = c4.slider("Mass m", 0.2, 3.0, 1.0, 0.1)
    omega = c5.slider("Angular frequency ω", 0.2, 4.0, 1.2, 0.05)
    hbar_eff = c6.slider("Effective ħ", 0.05, 1.0, 0.22, 0.01)
    sigma = c7.slider("Path spread", 0.02, 1.2, 0.18, 0.01)
    n_paths = c8.slider("Number of sampled paths", 200, 3000, 1000, 100)

    n_steps = 220
    n_ctrl = 11
    data = cached_family_data("ho", xa, xb, T, m, omega, n_steps, n_paths, sigma, n_ctrl, seed)
    phase_info = phase_data(data["action"], hbar_eff, data["ref_action"])
    metrics_box(data["ref_action"], phase_info, "Harmonic oscillator")

    col1, col2 = st.columns(2)
    col1.plotly_chart(
        make_trajectory_figure(data["t"], data["paths"], data["x_ref"], show_count, "Sampled paths and the classical oscillator path"),
        use_container_width=True,
    )
    col2.plotly_chart(
        make_density_figure(data["t"], data["paths"], data["x_ref"], "Trajectory density in the oscillator potential"),
        use_container_width=True,
    )

    col3, col4 = st.columns(2)
    col3.plotly_chart(
        make_action_histogram(phase_info["delta"], hbar_eff, "Distribution of action differences"),
        use_container_width=True,
    )
    col4.plotly_chart(
        make_phasor_figure(phase_info["amps"], phase_info["walk"], "Cumulative phase sum for the oscillator"),
        use_container_width=True,
    )

    st.caption(
        "Interpretation: the potential changes the stationary path. The path integral still includes many paths, but the most coherent neighborhood now surrounds the oscillator's classical solution."
    )


# -----------------------------
# Panel: Two path families
# -----------------------------
elif section == "Two path families":
    with st.expander("How to use this panel", expanded=True):
        st.markdown(
            """
- This is a simplified double-slit-style panel.
- The app generates two bundles of paths: an upper family and a lower family.
- Each family contributes its own average complex amplitude.
- The final signal depends on how these two family averages add in the complex plane.
            """
        )

    c1, c2, c3 = st.columns(3)
    xa = c1.slider("Initial position x_a", -2.0, 2.0, -1.0, 0.1)
    xb = c2.slider("Final position x_b", -2.0, 2.0, 1.0, 0.1)
    T = c3.slider("Total time T", 0.5, 5.0, 2.0, 0.1)

    c4, c5, c6, c7, c8 = st.columns(5)
    m = c4.slider("Mass m", 0.2, 3.0, 1.0, 0.1)
    bump = c5.slider("Family separation", 0.1, 2.0, 0.7, 0.05)
    hbar_eff = c6.slider("Effective ħ", 0.05, 1.0, 0.22, 0.01)
    sigma = c7.slider("Path spread inside each family", 0.02, 0.8, 0.12, 0.01)
    n_paths_each = c8.slider("Paths per family", 100, 1800, 700, 100)

    data = cached_two_family_data(xa, xb, T, m, 240, n_paths_each, sigma, 11, bump, hbar_eff, seed)

    top_mean = np.mean(data["top_amp"])
    bot_mean = np.mean(data["bot_amp"])
    total_amp = top_mean + bot_mean

    colm1, colm2, colm3 = st.columns(3)
    colm1.metric("|Upper-family mean amplitude|", f"{abs(top_mean):.3f}")
    colm2.metric("|Lower-family mean amplitude|", f"{abs(bot_mean):.3f}")
    colm3.metric("|Combined amplitude|", f"{abs(total_amp):.3f}")

    fig_paths = go.Figure()
    n = min(show_count, data["top_paths"].shape[0])
    step = max(data["top_paths"].shape[0] // n, 1)
    for row in data["top_paths"][::step][:n]:
        fig_paths.add_trace(go.Scatter(x=data["t"], y=row, mode="lines", opacity=0.15, showlegend=False))
    for row in data["bot_paths"][::step][:n]:
        fig_paths.add_trace(go.Scatter(x=data["t"], y=row, mode="lines", opacity=0.15, showlegend=False))
    fig_paths.add_trace(go.Scatter(x=data["t"], y=data["top_ref"], mode="lines", line=dict(width=4), name="Upper family reference"))
    fig_paths.add_trace(go.Scatter(x=data["t"], y=data["bot_ref"], mode="lines", line=dict(width=4), name="Lower family reference"))
    fig_paths.update_layout(title="Two bundles of candidate paths", xaxis_title="Time t", yaxis_title="Position x(t)", height=390)

    fig_ph = go.Figure()
    fig_ph.add_trace(go.Scatter(x=np.real(data["top_amp"]), y=np.imag(data["top_amp"]), mode="markers", opacity=0.3, name="Upper family"))
    fig_ph.add_trace(go.Scatter(x=np.real(data["bot_amp"]), y=np.imag(data["bot_amp"]), mode="markers", opacity=0.3, name="Lower family"))
    fig_ph.add_trace(go.Scatter(x=[0, np.real(top_mean)], y=[0, np.imag(top_mean)], mode="lines+markers", line=dict(width=4), name="Upper mean"))
    fig_ph.add_trace(go.Scatter(x=[0, np.real(bot_mean)], y=[0, np.imag(bot_mean)], mode="lines+markers", line=dict(width=4), name="Lower mean"))
    fig_ph.add_trace(go.Scatter(x=[0, np.real(total_amp)], y=[0, np.imag(total_amp)], mode="lines+markers", line=dict(width=5), name="Combined"))
    th = np.linspace(0, 2 * np.pi, 300)
    fig_ph.add_trace(go.Scatter(x=np.cos(th), y=np.sin(th), mode="lines", line=dict(width=1, dash="dash"), name="Unit circle"))
    fig_ph.update_layout(title="How the two families add in the complex plane", xaxis_title="Re", yaxis_title="Im", yaxis_scaleanchor="x", yaxis_scaleratio=1, height=390)

    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_paths, use_container_width=True)
    col2.plotly_chart(fig_ph, use_container_width=True)

    st.caption(
        "Interpretation: this panel is not a full slit experiment solver. It is a clean visualization of how two broad classes of paths can contribute separate complex amplitudes that then interfere."
    )


# -----------------------------
# Panel: Real time vs imaginary time
# -----------------------------
else:
    with st.expander("How to use this panel", expanded=True):
        st.markdown(
            """
- Real-time path integrals use the oscillatory factor \(e^{iS/\hbar}\), which causes strong cancellations.
- Imaginary-time methods replace time by \(t \to -i\tau\), leading to positive weights like \(e^{-S_E/\hbar}\).
- This panel shows why Euclidean-time sampling is numerically much more stable.
            """
        )

    model = st.radio("Reference model", ["Free particle", "Harmonic oscillator"], horizontal=True)
    c1, c2, c3 = st.columns(3)
    xa = c1.slider("Initial position x_a", -2.0, 2.0, -0.8, 0.1)
    xb = c2.slider("Final position x_b", -2.0, 2.0, 0.8, 0.1)
    T = c3.slider("Total time T", 0.5, 5.0, 2.0, 0.1)

    c4, c5, c6, c7, c8 = st.columns(5)
    m = c4.slider("Mass m", 0.2, 3.0, 1.0, 0.1)
    omega = c5.slider("Angular frequency ω", 0.2, 4.0, 1.0, 0.05, disabled=(model == "Free particle"))
    hbar_eff = c6.slider("Effective ħ", 0.05, 1.0, 0.25, 0.01)
    sigma = c7.slider("Path spread", 0.02, 1.0, 0.20, 0.01)
    n_paths = c8.slider("Number of sampled paths", 200, 3000, 1000, 100)

    if model == "Free particle":
        data = cached_family_data("free", xa, xb, T, m, 0.0, 220, n_paths, sigma, 11, seed)
        label = "free"
    else:
        data = cached_family_data("ho", xa, xb, T, m, omega, 220, n_paths, sigma, 11, seed)
        label = "oscillator"

    phase_info = phase_data(data["action"], hbar_eff, data["ref_action"])
    delta_e = data["euc_action"] - data["ref_euc_action"]
    weights = np.exp(-delta_e / hbar_eff)
    weights = weights / np.max(weights)

    col1, col2 = st.columns(2)
    col1.plotly_chart(
        make_phasor_figure(phase_info["amps"], phase_info["walk"], f"Real time: oscillatory phases for the {label} model"),
        use_container_width=True,
    )

    fig_w = go.Figure()
    fig_w.add_trace(go.Histogram(x=delta_e / hbar_eff, nbinsx=55, name="ΔS_E / ħ"))
    fig_w.update_layout(
        title="Imaginary time: Euclidean action differences",
        xaxis_title="(S_E - S_E,ref) / ħ",
        yaxis_title="Number of paths",
        height=360,
    )
    col2.plotly_chart(fig_w, use_container_width=True)

    idx = np.argsort(weights)[::-1]
    fig_t = go.Figure()
    sample_idx = idx[: min(show_count, len(idx))]
    for i in sample_idx:
        fig_t.add_trace(go.Scatter(x=data["t"], y=data["paths"][i], mode="lines", opacity=0.22, showlegend=False))
    fig_t.add_trace(go.Scatter(x=data["t"], y=data["x_ref"], mode="lines", line=dict(width=4), name="Reference path"))
    fig_t.update_layout(title="Paths with the largest Euclidean weights", xaxis_title="Time t", yaxis_title="Position x(t)", height=390)

    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=phase_info["delta"] / hbar_eff, y=np.abs(phase_info["amps"]), mode="markers", opacity=0.35, name="|e^{iΔS/ħ}| = 1"))
    fig_p.add_trace(go.Scatter(x=delta_e / hbar_eff, y=weights, mode="markers", opacity=0.35, name="e^{-ΔS_E/ħ}"))
    fig_p.update_layout(title="Why Euclidean weights are easier numerically", xaxis_title="Action difference / ħ", yaxis_title="Weight magnitude", height=390)

    col3, col4 = st.columns(2)
    col3.plotly_chart(fig_t, use_container_width=True)
    col4.plotly_chart(fig_p, use_container_width=True)

    st.caption(
        "Interpretation: in real time, every path has the same modulus and differs only by phase, which causes severe cancellations. In imaginary time, large-action paths are exponentially suppressed, which is why Euclidean methods are much more tractable numerically."
    )


# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
---
**Implementation note.** The expensive parts of the app are vectorized: batches of paths are represented as 2D <code>numpy</code> arrays, spline interpolation is performed for whole path ensembles at once, and discrete actions are evaluated without Python loops over paths. This keeps the app suitable for GitHub + Streamlit deployment while remaining transparent enough for teaching.
""",
    unsafe_allow_html=True,
)
