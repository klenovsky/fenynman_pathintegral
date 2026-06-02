import math
from typing import Callable, Dict, Iterable, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.interpolate import CubicSpline


# -------------------------------------------------
# Page setup
# -------------------------------------------------
st.set_page_config(
    page_title="Feynman Path Integral Explorer",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------
# Styling
# -------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.0rem;
        max-width: 1380px;
    }
    .hero {
        background: linear-gradient(135deg, rgba(42,88,165,0.10), rgba(144,190,240,0.10));
        border: 1px solid rgba(80,120,180,0.16);
        border-radius: 20px;
        padding: 1.15rem 1.25rem 1.0rem 1.25rem;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2.15rem;
        font-weight: 760;
        margin-bottom: 0.25rem;
        line-height: 1.15;
    }
    .hero-subtitle {
        font-size: 1.02rem;
        color: #667085;
        line-height: 1.5;
    }
    .soft-note {
        background: rgba(120, 160, 220, 0.08);
        border: 1px solid rgba(120, 160, 220, 0.12);
        border-radius: 14px;
        padding: 0.8rem 0.9rem;
        margin-bottom: 0.8rem;
    }
    .metric-note {
        font-size: 0.94rem;
        color: #5f6772;
        margin-top: -0.25rem;
        margin-bottom: 0.6rem;
    }
    .caption-box {
        font-size: 0.93rem;
        color: #626a76;
        background: rgba(130, 145, 165, 0.08);
        border-radius: 12px;
        padding: 0.65rem 0.8rem;
        margin-top: 0.35rem;
    }
    code {
        color: #0f4479;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------
# Localization helpers
# -------------------------------------------------
LANG_OPTIONS = ["English", "Čeština"]


def tr(en: str, cz: str) -> str:
    return en if st.session_state.get("app_lang", "English") == "English" else cz


def t_math(en: str, cz: str) -> str:
    return tr(en, cz)


def init_lang() -> None:
    if "app_lang" not in st.session_state:
        st.session_state.app_lang = "English"


init_lang()


# -------------------------------------------------
# Defaults and reset helpers
# -------------------------------------------------
DEFAULTS = {
    "global": {
        "seed": 7,
        "show_paths": 36,
        "anim_paths": 48,
    },
    "free": {
        "free_xa": -0.9,
        "free_xb": 0.9,
        "free_T": 2.0,
        "free_m": 1.0,
        "free_hbar": 0.25,
        "free_sigma": 0.25,
        "free_npaths": 900,
    },
    "ho": {
        "ho_xa": -0.7,
        "ho_xb": 0.7,
        "ho_T": 2.2,
        "ho_m": 1.0,
        "ho_omega": 1.2,
        "ho_hbar": 0.22,
        "ho_sigma": 0.18,
        "ho_npaths": 1000,
    },
    "slit": {
        "slit_detector": 0.15,
        "slit_sep": 0.95,
        "slit_hbar": 0.22,
        "slit_sigma": 0.11,
        "slit_npaths": 700,
        "slit_T": 2.0,
        "slit_m": 1.0,
        "slit_scan_span": 1.5,
    },
    "imag": {
        "imag_model": "Free particle",
        "imag_xa": -0.8,
        "imag_xb": 0.8,
        "imag_T": 2.0,
        "imag_m": 1.0,
        "imag_omega": 1.0,
        "imag_hbar": 0.25,
        "imag_sigma": 0.20,
        "imag_npaths": 1000,
    },
}


def ensure_defaults(group: str) -> None:
    for key, value in DEFAULTS[group].items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_group(group: str) -> None:
    for key, value in DEFAULTS[group].items():
        st.session_state[key] = value


ensure_defaults("global")
ensure_defaults("free")
ensure_defaults("ho")
ensure_defaults("slit")
ensure_defaults("imag")


# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.selectbox(
        tr("Language", "Jazyk"),
        LANG_OPTIONS,
        key="app_lang",
    )

    section = st.radio(
        tr("Choose a panel", "Vyber panel"),
        [
            tr("Free particle", "Volná částice"),
            tr("Harmonic oscillator", "Harmonický oscilátor"),
            tr("Double slit: two path families", "Dvojštěrbina: dvě rodiny drah"),
            tr("Real time vs imaginary time", "Reálný čas vs imaginární čas"),
        ],
    )

    st.markdown("---")
    st.slider(
        tr("Random seed", "Seed generátoru"),
        min_value=0,
        max_value=9999,
        step=1,
        key="seed",
    )
    st.slider(
        tr("Paths shown in static path plots", "Počet drah ve statických grafech"),
        min_value=12,
        max_value=100,
        step=4,
        key="show_paths",
    )
    st.slider(
        tr("Paths used in Play animation", "Počet drah v animaci Play"),
        min_value=12,
        max_value=80,
        step=4,
        key="anim_paths",
    )


# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown(
    f'''
    <div class="hero">
        <div class="hero-title">{tr("Feynman Path Integral Explorer", "Průzkumník Feynmanových drahových integrálů")}</div>
        <div class="hero-subtitle">{tr(
            "A teaching app for students: not one path, but many possible paths; not one answer, but an interference sum of amplitudes.",
            "Výuková aplikace pro studenty: ne jedna dráha, ale mnoho možných drah; ne jedna odpověď, ale interferenční součet amplitud."
        )}</div>
    </div>
    ''',
    unsafe_allow_html=True,
)

st.markdown(
    f'''
    <div class="soft-note">
    <b>{tr("What makes this app useful", "Proč je tato aplikace užitečná")}</b><br>
    {tr(
        "It turns the formal expression K = ∫𝒟[x] e^{iS/ħ} into pictures: path clouds, action distributions, phasors in the complex plane, and an animated Play mode showing how individual path contributions add up.",
        "Převádí formální výraz K = ∫𝒟[x] e^{iS/ħ} do obrázků: mračna drah, rozdělení akcí, fázory v komplexní rovině a animovaný režim Play ukazující, jak se jednotlivé příspěvky drah skládají."
    )}
    </div>
    ''',
    unsafe_allow_html=True,
)


# -------------------------------------------------
# Theory expander
# -------------------------------------------------
with st.expander(tr("Theory, intuition, and references", "Teorie, intuice a reference"), expanded=False):
    st.markdown(
        t_math(
            r"""
### 1. Core idea
For fixed endpoints $x_a,0$ and $x_b,T$, the propagator is written formally as

$$
K(x_b,T;x_a,0)=\int \mathcal{D}[x(t)]\,e^{iS[x]/\hbar},
$$

where the action is

$$
S[x]=\int_0^T L(x,\dot x,t)\,dt,
\qquad
L=\frac{1}{2}m\dot x^2 - V(x).
$$

The message for students is simple: quantum motion is not described by a single path. It is described by **interference of amplitudes coming from many paths**.

### 2. Why the classical path still appears
The classical path is not singled out because other paths are forbidden. It is special because it makes the action stationary,

$$
\delta S = 0.
$$

Near that path, the phase $S/\hbar$ changes slowly, so nearby contributions tend to add coherently. Far from it, phases oscillate rapidly and tend to cancel. This is the **stationary-phase idea**.

### 3. What the app computes numerically
This app is intentionally didactic. It does **not** evaluate the continuum path integral exactly. Instead, it:
- discretizes the time interval,
- generates many smooth trial paths with vectorized `numpy` arrays,
- evaluates discrete actions for whole ensembles at once,
- converts them into phase factors $e^{iS/\hbar}$,
- shows how cumulative interference builds up.

### 4. Why several panels are useful
- **Free particle:** the classical reference path is a straight line.
- **Harmonic oscillator:** the potential bends the preferred path.
- **Double slit:** two families of paths contribute separate amplitudes that interfere.
- **Real vs imaginary time:** real time gives oscillatory weights; imaginary time gives exponentially damped weights.

### 5. What to look for in the plots
- **Path cloud:** many possible paths between the same endpoints.
- **Density plot:** where sampled paths accumulate in the $(t,x)$ plane.
- **Action histogram:** how far trial paths are from the reference action.
- **Phasor plot:** each path contributes one complex number on the unit circle.
- **Play animation:** the cumulative sum updates as paths are added one by one.

### 6. Suggested references
1. R. P. Feynman and A. R. Hibbs, *Quantum Mechanics and Path Integrals*.
2. L. S. Schulman, *Techniques and Applications of Path Integration*.
3. H. Kleinert, *Path Integrals in Quantum Mechanics, Statistics, Polymer Physics, and Financial Markets*.
4. D. J. Griffiths and D. F. Schroeter, *Introduction to Quantum Mechanics*.
5. R. Shankar, *Principles of Quantum Mechanics*.
6. M. Chaichian and A. Demichev, *Path Integrals in Physics*.
            """,
            r"""
### 1. Základní myšlenka
Pro pevné koncové body $x_a,0$ a $x_b,T$ lze propagátor formálně zapsat jako

$$
K(x_b,T;x_a,0)=\int \mathcal{D}[x(t)]\,e^{iS[x]/\hbar},
$$

kde akce je

$$
S[x]=\int_0^T L(x,\dot x,t)\,dt,
\qquad
L=\frac{1}{2}m\dot x^2 - V(x).
$$

Hlavní sdělení pro studenty je jednoduché: kvantový pohyb není popsán jedinou drahou. Je popsán **interferencí amplitud pocházejících z mnoha drah**.

### 2. Proč se přesto objevuje klasická dráha
Klasická dráha není výjimečná tím, že by ostatní dráhy byly zakázané. Je výjimečná tím, že činí akci stacionární,

$$
\delta S = 0.
$$

V jejím okolí se fáze $S/\hbar$ mění pomalu, takže blízké příspěvky se obvykle sčítají koherentně. Daleko od ní fáze rychle oscilují a příspěvky se mají tendenci rušit. To je idea **stacionární fáze**.

### 3. Co aplikace numericky počítá
Tato aplikace je záměrně didaktická. **Neřeší** přesně kontinuální drahový integrál. Místo toho:
- diskretizuje časový interval,
- generuje mnoho hladkých zkušebních drah pomocí vektorových polí `numpy`,
- vyhodnocuje diskrétní akce pro celé ansámbly najednou,
- převádí je na fázové faktory $e^{iS/\hbar}$,
- ukazuje, jak se buduje kumulativní interference.

### 4. Proč jsou užitečné různé panely
- **Volná částice:** klasická referenční dráha je přímka.
- **Harmonický oscilátor:** potenciál zakřivuje preferovanou dráhu.
- **Dvojštěrbina:** dvě rodiny drah dávají oddělené amplitudy, které interferují.
- **Reálný vs imaginární čas:** v reálném čase jsou váhy oscilující, v imaginárním čase exponenciálně tlumené.

### 5. Na co se dívat v grafech
- **Mračno drah:** mnoho možných drah mezi stejnými konci.
- **Hustotní mapa:** kde se vzorkované dráhy koncentrují v rovině $(t,x)$.
- **Histogram akcí:** jak daleko jsou zkušební dráhy od referenční akce.
- **Fázorový graf:** každá dráha přispívá jedním komplexním číslem na jednotkové kružnici.
- **Animace Play:** kumulativní součet se aktualizuje při přidávání drah po jedné.

### 6. Doporučené reference
1. R. P. Feynman a A. R. Hibbs, *Quantum Mechanics and Path Integrals*.
2. L. S. Schulman, *Techniques and Applications of Path Integration*.
3. H. Kleinert, *Path Integrals in Quantum Mechanics, Statistics, Polymer Physics, and Financial Markets*.
4. D. J. Griffiths a D. F. Schroeter, *Introduction to Quantum Mechanics*.
5. R. Shankar, *Principles of Quantum Mechanics*.
6. M. Chaichian a A. Demichev, *Path Integrals in Physics*.
            """
        )
    )


# -------------------------------------------------
# Numerical helpers
# -------------------------------------------------
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


def make_reference_family(t: np.ndarray, xa: float, xb: float, T: float, bump: float, sign: float) -> np.ndarray:
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
    T = float(t[-1])
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


def phase_data(actions: np.ndarray, hbar_eff: float, ref_action: float, order: str = "stationary") -> Dict[str, np.ndarray]:
    delta = actions - ref_action
    amps = np.exp(1j * delta / hbar_eff)
    if order == "stationary":
        sort_idx = np.argsort(np.abs(delta))
    else:
        sort_idx = np.arange(len(actions))
    ordered = amps[sort_idx]
    walk = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return {
        "delta": delta,
        "amps": amps,
        "sort_idx": sort_idx,
        "ordered_amps": ordered,
        "walk": walk,
    }


def trajectory_density(t: np.ndarray, paths: np.ndarray, bins_t: int = 90, bins_x: int = 90):
    tt = np.broadcast_to(t[None, :], paths.shape).ravel()
    xx = paths.ravel()
    hist, t_edges, x_edges = np.histogram2d(tt, xx, bins=[bins_t, bins_x])
    return hist.T, t_edges, x_edges


def choose_path_indices(total_paths: int, wanted: int) -> np.ndarray:
    wanted = min(wanted, total_paths)
    return np.linspace(0, total_paths - 1, wanted, dtype=int)


def make_trajectory_figure(
    t: np.ndarray,
    paths: np.ndarray,
    x_ref: np.ndarray,
    sample_count: int,
    title: str,
    yaxis_title: str,
) -> go.Figure:
    fig = go.Figure()
    idx = choose_path_indices(paths.shape[0], sample_count)
    for row in paths[idx]:
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
            name=tr("Reference / classical path", "Referenční / klasická dráha"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=tr("Time t", "Čas t"),
        yaxis_title=yaxis_title,
        margin=dict(l=20, r=20, t=48, b=20),
        height=390,
        legend=dict(orientation="h"),
    )
    return fig


def make_density_figure(t: np.ndarray, paths: np.ndarray, x_ref: np.ndarray, title: str, yaxis_title: str) -> go.Figure:
    hist, t_edges, x_edges = trajectory_density(t, paths)
    tc = 0.5 * (t_edges[:-1] + t_edges[1:])
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=tc,
            y=xc,
            z=hist,
            colorbar_title=tr("Counts", "Počty"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=x_ref,
            mode="lines",
            line=dict(width=3, color="white"),
            name=tr("Reference path", "Referenční dráha"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=tr("Time t", "Čas t"),
        yaxis_title=yaxis_title,
        margin=dict(l=20, r=20, t=48, b=20),
        height=390,
    )
    return fig


def make_action_histogram(delta_actions: np.ndarray, hbar_eff: float, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=delta_actions / hbar_eff, nbinsx=56, name="ΔS/ħ"))
    fig.update_layout(
        title=title,
        xaxis_title=tr("(S - S_ref) / ħ", "(S - S_ref) / ħ"),
        yaxis_title=tr("Number of paths", "Počet drah"),
        bargap=0.06,
        margin=dict(l=20, r=20, t=48, b=20),
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
            name=tr("Individual phases", "Jednotlivé fáze"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.real(walk),
            y=np.imag(walk),
            mode="lines+markers",
            line=dict(width=3),
            marker=dict(size=4),
            name=tr("Cumulative average", "Kumulativní průměr"),
        )
    )
    th = np.linspace(0, 2 * np.pi, 300)
    fig.add_trace(
        go.Scatter(
            x=np.cos(th),
            y=np.sin(th),
            mode="lines",
            line=dict(width=1, dash="dash"),
            name=tr("Unit circle", "Jednotková kružnice"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Re",
        yaxis_title="Im",
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
        margin=dict(l=20, r=20, t=48, b=20),
        height=360,
    )
    return fig


def make_addition_animation(
    t: np.ndarray,
    paths: np.ndarray,
    x_ref: np.ndarray,
    ordered_indices: np.ndarray,
    ordered_amps: np.ndarray,
    title: str,
    yaxis_title: str,
) -> go.Figure:
    n = len(ordered_indices)
    ordered_paths = paths[ordered_indices]
    walk = np.cumsum(ordered_amps) / np.arange(1, n + 1)
    running_path = np.cumsum(ordered_paths, axis=0) / np.arange(1, n + 1)[:, None]
    final_path = running_path[-1]
    th = np.linspace(0, 2 * np.pi, 300)

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.56, 0.44],
        subplot_titles=(
            tr("Running path sum (normalized)", "Průběžný součet drah (normalizovaný)"),
            tr("Cumulative phasor sum", "Kumulativní součet fázorů"),
        ),
    )

    # Left panel: the final averaged path is shown as a faint shadow from the start.
    fig.add_trace(
        go.Scatter(
            x=t,
            y=final_path,
            mode="lines",
            line=dict(width=6, dash="dot"),
            opacity=0.18,
            name=tr("Final normalized sum (shadow)", "Konečný normalizovaný součet (stín)"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=x_ref,
            mode="lines",
            line=dict(width=4),
            name=tr("Reference path", "Referenční dráha"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=running_path[0],
            mode="lines",
            line=dict(width=4),
            name=tr("Running normalized sum", "Běžící normalizovaný součet"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=ordered_paths[0],
            mode="lines",
            line=dict(width=2),
            opacity=0.75,
            name=tr("Current path", "Aktuální dráha"),
        ),
        row=1,
        col=1,
    )

    # Right panel: phasor addition with faint final walk shown from the start.
    fig.add_trace(
        go.Scatter(
            x=np.real(ordered_amps),
            y=np.imag(ordered_amps),
            mode="markers",
            marker=dict(size=6),
            opacity=0.18,
            name=tr("All sampled path phases", "Fáze všech vybraných drah"),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=np.cos(th),
            y=np.sin(th),
            mode="lines",
            line=dict(width=1, dash="dash"),
            name=tr("Unit circle", "Jednotková kružnice"),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=np.real(walk),
            y=np.imag(walk),
            mode="lines",
            line=dict(width=5, dash="dot"),
            opacity=0.16,
            name=tr("Final cumulative walk (shadow)", "Konečná kumulativní trajektorie (stín)"),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=np.real(walk[:1]),
            y=np.imag(walk[:1]),
            mode="lines+markers",
            line=dict(width=3),
            marker=dict(size=7),
            name=tr("Running cumulative walk", "Běžící kumulativní trajektorie"),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[np.real(walk[0])],
            y=[np.imag(walk[0])],
            mode="markers",
            marker=dict(size=10),
            name=tr("Current cumulative point", "Aktuální kumulativní bod"),
        ),
        row=1,
        col=2,
    )

    frames = []
    for k in range(n):
        frame = go.Frame(
            data=[
                go.Scatter(x=t, y=final_path),
                go.Scatter(x=t, y=x_ref),
                go.Scatter(x=t, y=running_path[k]),
                go.Scatter(x=t, y=ordered_paths[k]),
                go.Scatter(x=np.real(ordered_amps), y=np.imag(ordered_amps)),
                go.Scatter(x=np.cos(th), y=np.sin(th)),
                go.Scatter(x=np.real(walk), y=np.imag(walk)),
                go.Scatter(x=np.real(walk[: k + 1]), y=np.imag(walk[: k + 1])),
                go.Scatter(x=[np.real(walk[k])], y=[np.imag(walk[k])]),
            ],
            name=str(k),
            layout=go.Layout(
                title=(
                    f"{title}<br><sup>{tr('Added paths', 'Přidané dráhy')}: {k + 1}/{n}, "
                    f"{tr('cumulative magnitude', 'velikost součtu')}: {abs(walk[k]):.3f}</sup>"
                )
            ),
        )
        frames.append(frame)

    fig.frames = frames
    fig.update_xaxes(title_text=tr("Time t", "Čas t"), row=1, col=1)
    fig.update_yaxes(title_text=yaxis_title, row=1, col=1)
    fig.update_xaxes(title_text="Re", row=1, col=2)
    fig.update_yaxes(title_text="Im", scaleanchor="x2", scaleratio=1, row=1, col=2)
    fig.update_layout(
        title=(
            f"{title}<br><sup>{tr('Press Play to watch the running path sum and the phasor sum build up', 'Stiskni Play a sleduj vznik běžícího součtu drah i součtu fázorů')}</sup>"
        ),
        height=500,
        margin=dict(l=20, r=20, t=88, b=20),
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.02,
                "y": 1.16,
                "buttons": [
                    {
                        "label": tr("Play", "Play"),
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 150, "redraw": False}, "fromcurrent": True, "transition": {"duration": 0}}],
                    },
                    {
                        "label": tr("Pause", "Pauza"),
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.1,
                "len": 0.85,
                "y": -0.08,
                "currentvalue": {"prefix": tr("Step ", "Krok ")},
                "steps": [
                    {
                        "label": str(k + 1),
                        "method": "animate",
                        "args": [[str(k)], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
                    }
                    for k in range(n)
                ],
            }
        ],
        showlegend=False,
    )
    return fig


def make_real_imag_addition_animation(
    t: np.ndarray,
    paths: np.ndarray,
    x_ref: np.ndarray,
    ordered_indices: np.ndarray,
    ordered_amps: np.ndarray,
    ordered_weights: np.ndarray,
    title: str,
    yaxis_title: str,
) -> go.Figure:
    n = len(ordered_indices)
    ordered_paths = paths[ordered_indices]
    running_path = np.cumsum(ordered_paths, axis=0) / np.arange(1, n + 1)[:, None]
    final_path = running_path[-1]
    real_curve = np.abs(np.cumsum(ordered_amps) / np.arange(1, n + 1))
    imag_curve = np.cumsum(ordered_weights) / np.arange(1, n + 1)
    steps = np.arange(1, n + 1)

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.56, 0.44],
        subplot_titles=(
            tr("Running path sum (normalized)", "Průběžný součet drah (normalizovaný)"),
            tr("Real-time cancellation vs imaginary-time damping", "Rušení v reálném čase vs tlumení v imaginárním čase"),
        ),
    )

    fig.add_trace(
        go.Scatter(x=t, y=final_path, mode="lines", line=dict(width=6, dash="dot"), opacity=0.18, name=tr("Final normalized sum (shadow)", "Konečný normalizovaný součet (stín)")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=x_ref, mode="lines", line=dict(width=4), name=tr("Reference path", "Referenční dráha")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=running_path[0], mode="lines", line=dict(width=4), name=tr("Running normalized sum", "Běžící normalizovaný součet")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=ordered_paths[0], mode="lines", line=dict(width=2), opacity=0.75, name=tr("Current path", "Aktuální dráha")),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(x=steps, y=real_curve, mode="lines", line=dict(width=5, dash="dot"), opacity=0.18, name=tr("Final real-time curve (shadow)", "Konečná křivka reálného času (stín)")),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=steps, y=imag_curve, mode="lines", line=dict(width=5, dash="dot"), opacity=0.18, name=tr("Final imaginary-time curve (shadow)", "Konečná křivka imaginárního času (stín)")),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=steps[:1], y=real_curve[:1], mode="lines+markers", line=dict(width=3), marker=dict(size=7), name=tr("Running real-time magnitude", "Běžící velikost v reálném čase")),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=steps[:1], y=imag_curve[:1], mode="lines+markers", line=dict(width=3), marker=dict(size=7), name=tr("Running imaginary-time mean weight", "Běžící střední váha v imaginárním čase")),
        row=1, col=2,
    )

    frames = []
    for k in range(n):
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(x=t, y=final_path),
                    go.Scatter(x=t, y=x_ref),
                    go.Scatter(x=t, y=running_path[k]),
                    go.Scatter(x=t, y=ordered_paths[k]),
                    go.Scatter(x=steps, y=real_curve),
                    go.Scatter(x=steps, y=imag_curve),
                    go.Scatter(x=steps[: k + 1], y=real_curve[: k + 1]),
                    go.Scatter(x=steps[: k + 1], y=imag_curve[: k + 1]),
                ],
                name=str(k),
                layout=go.Layout(
                    title=(
                        f"{title}<br><sup>{tr('Added paths', 'Přidané dráhy')}: {k + 1}/{n}, "
                        f"{tr('real-time magnitude', 'velikost v reálném čase')}: {real_curve[k]:.3f}, "
                        f"{tr('imaginary-time mean weight', 'střední váha v imaginárním čase')}: {imag_curve[k]:.3f}</sup>"
                    )
                ),
            )
        )

    fig.frames = frames
    fig.update_xaxes(title_text=tr("Time t", "Čas t"), row=1, col=1)
    fig.update_yaxes(title_text=yaxis_title, row=1, col=1)
    fig.update_xaxes(title_text=tr("Added paths", "Přidané dráhy"), row=1, col=2)
    fig.update_yaxes(title_text=tr("Magnitude / mean weight", "Velikost / střední váha"), row=1, col=2)
    fig.update_layout(
        title=f"{title}<br><sup>{tr('Press Play to compare running real-time cancellation and imaginary-time damping', 'Stiskni Play a porovnej průběžné rušení v reálném čase s tlumením v imaginárním čase')}</sup>",
        height=500,
        margin=dict(l=20, r=20, t=88, b=20),
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.02,
                "y": 1.16,
                "buttons": [
                    {
                        "label": tr("Play", "Play"),
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 150, "redraw": False}, "fromcurrent": True, "transition": {"duration": 0}}],
                    },
                    {
                        "label": tr("Pause", "Pauza"),
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.1,
                "len": 0.85,
                "y": -0.08,
                "currentvalue": {"prefix": tr("Step ", "Krok ")},
                "steps": [
                    {
                        "label": str(k + 1),
                        "method": "animate",
                        "args": [[str(k)], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
                    }
                    for k in range(n)
                ],
            }
        ],
        showlegend=False,
    )
    return fig


# -------------------------------------------------
# Cached data builders
# -------------------------------------------------
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
    return {
        "t": t,
        "x_ref": x_ref,
        "paths": paths,
        "action": action,
        "euc_action": euc_action,
        "ref_action": ref_action[0],
        "ref_euc_action": ref_euc[0],
        "mean_x": mean_x,
    }


@st.cache_data(show_spinner=False)
def cached_double_slit_data(
    detector_y: float,
    slit_sep: float,
    T: float,
    m: float,
    n_steps: int,
    n_paths_each: int,
    sigma: float,
    n_ctrl: int,
    hbar_eff: float,
    seed: int,
    scan_span: float,
):
    # Use tau in [0, T] as evolution parameter, y(tau) as transverse coordinate.
    t = np.linspace(0.0, T, n_steps)
    source_y = 0.0
    upper_slit_y = 0.5 * slit_sep
    lower_slit_y = -0.5 * slit_sep

    # Reference families through upper and lower slit.
    xa = source_y
    xb = detector_y
    upper_ref = make_reference_family(t, xa, xb, T, upper_slit_y, +1.0)
    lower_ref = make_reference_family(t, xa, xb, T, abs(lower_slit_y), -1.0)

    upper_paths = generate_paths_from_reference(t, upper_ref, n_paths_each, sigma, n_ctrl, seed)
    lower_paths = generate_paths_from_reference(t, lower_ref, n_paths_each, sigma, n_ctrl, seed + 791)

    upper_action, _, _ = compute_actions(t, upper_paths, m, potential_free, 0.0)
    lower_action, _, _ = compute_actions(t, lower_paths, m, potential_free, 0.0)
    upper_ref_action, _, _ = compute_actions(t, upper_ref[None, :], m, potential_free, 0.0)
    lower_ref_action, _, _ = compute_actions(t, lower_ref[None, :], m, potential_free, 0.0)

    upper_amp = np.exp(1j * (upper_action - upper_ref_action[0]) / hbar_eff)
    lower_amp = np.exp(1j * (lower_action - lower_ref_action[0]) / hbar_eff)

    # Detector scan from a simple two-segment classical-action model.
    ys = np.linspace(-scan_span, scan_span, 220)
    source = np.array([0.0, source_y])
    upper_slit = np.array([0.5, upper_slit_y])
    lower_slit = np.array([0.5, lower_slit_y])
    T_half = T / 2.0

    det = np.stack([np.ones_like(ys), ys], axis=1)
    upper_action_scan = (m / (2 * T_half)) * (
        np.sum((upper_slit - source) ** 2) + np.sum((det - upper_slit[None, :]) ** 2, axis=1)
    )
    lower_action_scan = (m / (2 * T_half)) * (
        np.sum((lower_slit - source) ** 2) + np.sum((det - lower_slit[None, :]) ** 2, axis=1)
    )
    scan_amp = np.exp(1j * upper_action_scan / hbar_eff) + np.exp(1j * lower_action_scan / hbar_eff)
    intensity = np.abs(scan_amp) ** 2
    intensity /= np.max(intensity)

    return {
        "t": t,
        "upper_ref": upper_ref,
        "lower_ref": lower_ref,
        "upper_paths": upper_paths,
        "lower_paths": lower_paths,
        "upper_amp": upper_amp,
        "lower_amp": lower_amp,
        "upper_ref_action": upper_ref_action[0],
        "lower_ref_action": lower_ref_action[0],
        "screen_y": ys,
        "screen_intensity": intensity,
        "source_y": source_y,
        "upper_slit_y": upper_slit_y,
        "lower_slit_y": lower_slit_y,
        "detector_y": detector_y,
    }


# -------------------------------------------------
# Panel helpers
# -------------------------------------------------
def metrics_box(ref_action: float, phase_info: Dict[str, np.ndarray], label: str) -> None:
    walk = phase_info["walk"]
    coherence = np.abs(np.mean(phase_info["amps"]))
    final_mag = np.abs(walk[-1])
    col1, col2, col3 = st.columns(3)
    col1.metric(label, f"{ref_action:.3f}")
    col2.metric(tr("|mean phase factor|", "|střední fázový faktor|"), f"{coherence:.3f}")
    col3.metric(tr("Final cumulative magnitude", "Konečná velikost součtu"), f"{final_mag:.3f}")


def section_header(title_en: str, title_cz: str, subtitle_en: str, subtitle_cz: str) -> None:
    st.markdown(f"### {tr(title_en, title_cz)}")
    st.markdown(f'<div class="metric-note">{tr(subtitle_en, subtitle_cz)}</div>', unsafe_allow_html=True)


def caption(text_en: str, text_cz: str) -> None:
    st.markdown(f'<div class="caption-box">{tr(text_en, text_cz)}</div>', unsafe_allow_html=True)


# -------------------------------------------------
# Panel: Free particle
# -------------------------------------------------
if section == tr("Free particle", "Volná částice"):
    section_header(
        "Free particle",
        "Volná částice",
        "The classical path is a straight line. The path integral still samples many curved alternatives around it.",
        "Klasická dráha je přímka. Drahový integrál přesto vzorkuje mnoho zakřivených alternativ v jejím okolí.",
    )

    with st.expander(tr("How to use this panel and what the plots mean", "Jak panel používat a co ukazují grafy"), expanded=True):
        st.markdown(
            tr(
                """
- Set the initial point, final point, and total travel time.
- Increase **Path spread** to generate wilder trajectories around the classical straight line.
- Decrease **Effective ħ** to make stationary phase sharper.
- **Sampled paths** shows individual candidate paths and the straight classical reference.
- **Path density** shows where the sampled ensemble spends most of its time in the $t,x$ plane.
- **Action histogram** shows how far the trial actions are from the reference action.
- **Phasor interference** puts each path contribution on the complex plane.
- **Play: path-by-path accumulation** adds paths one by one so you can watch the cumulative complex sum build up.
                """,
                """
- Nastav počáteční bod, koncový bod a celkový čas pohybu.
- Zvýšením **Rozptylu drah** vytvoříš divočejší trajektorie kolem klasické přímky.
- Snížením **Efektivního ħ** se zvýrazní stacionární fáze.
- **Vzorkované dráhy** ukazují jednotlivé kandidátní dráhy a klasickou přímku.
- **Hustota drah** ukazuje, kde se ansámbl drah nejvíce koncentruje v rovině $t,x$.
- **Histogram akcí** ukazuje, jak daleko jsou zkušební akce od referenční akce.
- **Interference fázorů** převádí každý příspěvek dráhy do komplexní roviny.
- **Play: skládání drah po jedné** přidává dráhy postupně, takže lze sledovat vznik kumulativního komplexního součtu.
                """
            )
        )

    ctop = st.columns([1, 1, 1, 0.8])
    with ctop[3]:
        if st.button(tr("Reset free-particle defaults", "Reset výchozích hodnot volné částice"), use_container_width=True):
            reset_group("free")
            st.rerun()

    c1, c2, c3 = st.columns(3)
    c1.slider(tr("Initial position x_a", "Počáteční poloha x_a"), -2.0, 2.0, step=0.1, key="free_xa")
    c2.slider(tr("Final position x_b", "Koncová poloha x_b"), -2.0, 2.0, step=0.1, key="free_xb")
    c3.slider(tr("Total time T", "Celkový čas T"), 0.5, 5.0, step=0.1, key="free_T")

    c4, c5, c6, c7 = st.columns(4)
    c4.slider(tr("Mass m", "Hmotnost m"), 0.2, 3.0, step=0.1, key="free_m")
    c5.slider(tr("Effective ħ", "Efektivní ħ"), 0.05, 1.0, step=0.01, key="free_hbar")
    c6.slider(tr("Path spread", "Rozptyl drah"), 0.02, 1.2, step=0.01, key="free_sigma")
    c7.slider(tr("Number of sampled paths", "Počet vzorkovaných drah"), 200, 3000, step=100, key="free_npaths")

    data = cached_family_data(
        "free",
        st.session_state.free_xa,
        st.session_state.free_xb,
        st.session_state.free_T,
        st.session_state.free_m,
        0.0,
        220,
        st.session_state.free_npaths,
        st.session_state.free_sigma,
        11,
        st.session_state.seed,
    )
    phase_info = phase_data(data["action"], st.session_state.free_hbar, data["ref_action"], order="stationary")
    metrics_box(data["ref_action"], phase_info, tr("Reference action", "Referenční akce"))

    col1, col2 = st.columns(2)
    col1.plotly_chart(
        make_trajectory_figure(
            data["t"], data["paths"], data["x_ref"], st.session_state.show_paths,
            tr("Sampled paths around the classical straight line", "Vzorkované dráhy kolem klasické přímky"),
            tr("Position x(t)", "Poloha x(t)"),
        ),
        use_container_width=True,
    )
    col2.plotly_chart(
        make_density_figure(
            data["t"], data["paths"], data["x_ref"],
            tr("Path density in the (t, x) plane", "Hustota drah v rovině (t, x)"),
            tr("Position x(t)", "Poloha x(t)"),
        ),
        use_container_width=True,
    )

    col3, col4 = st.columns(2)
    col3.plotly_chart(
        make_action_histogram(phase_info["delta"], st.session_state.free_hbar, tr("Action histogram", "Histogram akcí")),
        use_container_width=True,
    )
    col4.plotly_chart(
        make_phasor_figure(phase_info["amps"], phase_info["walk"], tr("Phasor interference", "Interference fázorů")),
        use_container_width=True,
    )

    anim_indices = phase_info["sort_idx"][: min(st.session_state.anim_paths, len(phase_info["sort_idx"]))]
    anim_amps = phase_info["ordered_amps"][: len(anim_indices)]
    st.plotly_chart(
        make_addition_animation(
            data["t"],
            data["paths"],
            data["x_ref"],
            anim_indices,
            anim_amps,
            tr("Play: path-by-path accumulation for the free particle", "Play: skládání drah po jedné pro volnou částici"),
            tr("Position x(t)", "Poloha x(t)"),
        ),
        use_container_width=True,
    )

    caption(
        "The classical path is not the only allowed path. It is the path around which nearby phases vary slowly enough to add more coherently.",
        "Klasická dráha není jediná povolená dráha. Je to dráha, kolem níž se blízké fáze mění dost pomalu, aby se sčítaly koherentněji.",
    )


# -------------------------------------------------
# Panel: Harmonic oscillator
# -------------------------------------------------
elif section == tr("Harmonic oscillator", "Harmonický oscilátor"):
    section_header(
        "Harmonic oscillator",
        "Harmonický oscilátor",
        "A confining potential bends the stationary path and reshapes the phase landscape.",
        "Konfinující potenciál zakřivuje stacionární dráhu a mění krajinu fází.",
    )

    with st.expander(tr("How to use this panel and what the plots mean", "Jak panel používat a co ukazují grafy"), expanded=True):
        st.markdown(
            tr(
                """
- Change the endpoints and the total time.
- Increase **Angular frequency ω** to strengthen the restoring force.
- Compare the bent classical path with the cloud of nearby alternatives.
- The first two plots show the geometry of the sampled paths.
- The histogram shows the distribution of action differences.
- The phasor plot shows how these action differences become phase differences.
- The Play animation adds trajectories in stationary-phase order.
                """,
                """
- Měň koncové body a celkový čas.
- Zvýšením **Úhlové frekvence ω** zesílíš navracející sílu.
- Porovnej zakřivenou klasickou dráhu s mračnem blízkých alternativ.
- První dva grafy ukazují geometrii vzorkovaných drah.
- Histogram ukazuje rozdělení rozdílů akcí.
- Fázorový graf ukazuje, jak se rozdíly akcí převádějí na rozdíly fází.
- Animace Play přidává trajektorie v pořadí stacionární fáze.
                """
            )
        )

    ctop = st.columns([1, 1, 1, 0.8])
    with ctop[3]:
        if st.button(tr("Reset oscillator defaults", "Reset výchozích hodnot oscilátoru"), use_container_width=True):
            reset_group("ho")
            st.rerun()

    c1, c2, c3 = st.columns(3)
    c1.slider(tr("Initial position x_a", "Počáteční poloha x_a"), -2.0, 2.0, step=0.1, key="ho_xa")
    c2.slider(tr("Final position x_b", "Koncová poloha x_b"), -2.0, 2.0, step=0.1, key="ho_xb")
    c3.slider(tr("Total time T", "Celkový čas T"), 0.5, 6.0, step=0.1, key="ho_T")

    c4, c5, c6, c7, c8 = st.columns(5)
    c4.slider(tr("Mass m", "Hmotnost m"), 0.2, 3.0, step=0.1, key="ho_m")
    c5.slider(tr("Angular frequency ω", "Úhlová frekvence ω"), 0.2, 4.0, step=0.05, key="ho_omega")
    c6.slider(tr("Effective ħ", "Efektivní ħ"), 0.05, 1.0, step=0.01, key="ho_hbar")
    c7.slider(tr("Path spread", "Rozptyl drah"), 0.02, 1.2, step=0.01, key="ho_sigma")
    c8.slider(tr("Number of sampled paths", "Počet vzorkovaných drah"), 200, 3000, step=100, key="ho_npaths")

    data = cached_family_data(
        "ho",
        st.session_state.ho_xa,
        st.session_state.ho_xb,
        st.session_state.ho_T,
        st.session_state.ho_m,
        st.session_state.ho_omega,
        220,
        st.session_state.ho_npaths,
        st.session_state.ho_sigma,
        11,
        st.session_state.seed,
    )
    phase_info = phase_data(data["action"], st.session_state.ho_hbar, data["ref_action"], order="stationary")
    metrics_box(data["ref_action"], phase_info, tr("Reference action", "Referenční akce"))

    col1, col2 = st.columns(2)
    col1.plotly_chart(
        make_trajectory_figure(
            data["t"], data["paths"], data["x_ref"], st.session_state.show_paths,
            tr("Sampled paths and the classical oscillator path", "Vzorkované dráhy a klasická dráha oscilátoru"),
            tr("Position x(t)", "Poloha x(t)"),
        ),
        use_container_width=True,
    )
    col2.plotly_chart(
        make_density_figure(
            data["t"], data["paths"], data["x_ref"],
            tr("Trajectory density in the oscillator potential", "Hustota trajektorií v potenciálu oscilátoru"),
            tr("Position x(t)", "Poloha x(t)"),
        ),
        use_container_width=True,
    )

    col3, col4 = st.columns(2)
    col3.plotly_chart(
        make_action_histogram(phase_info["delta"], st.session_state.ho_hbar, tr("Action histogram", "Histogram akcí")),
        use_container_width=True,
    )
    col4.plotly_chart(
        make_phasor_figure(phase_info["amps"], phase_info["walk"], tr("Phasor interference", "Interference fázorů")),
        use_container_width=True,
    )

    anim_indices = phase_info["sort_idx"][: min(st.session_state.anim_paths, len(phase_info["sort_idx"]))]
    anim_amps = phase_info["ordered_amps"][: len(anim_indices)]
    st.plotly_chart(
        make_addition_animation(
            data["t"],
            data["paths"],
            data["x_ref"],
            anim_indices,
            anim_amps,
            tr("Play: path-by-path accumulation for the oscillator", "Play: skládání drah po jedné pro oscilátor"),
            tr("Position x(t)", "Poloha x(t)"),
        ),
        use_container_width=True,
    )

    caption(
        "The potential does not remove alternative paths. It changes which neighborhood of paths keeps phase coherence best.",
        "Potenciál neodstraňuje alternativní dráhy. Mění pouze to, které okolí drah si nejlépe zachovává fázovou koherenci.",
    )


# -------------------------------------------------
# Panel: Double slit
# -------------------------------------------------
elif section == tr("Double slit: two path families", "Dvojštěrbina: dvě rodiny drah"):
    section_header(
        "Double slit: two path families",
        "Dvojštěrbina: dvě rodiny drah",
        "A simple two-slit-style panel: upper and lower path bundles contribute separate amplitudes that interfere on the screen.",
        "Zjednodušený panel typu dvojštěrbiny: horní a dolní svazek drah dávají oddělené amplitudy, které interferují na stínítku.",
    )

    with st.expander(tr("How to use this panel and what the plots mean", "Jak panel používat a co ukazují grafy"), expanded=True):
        st.markdown(
            tr(
                """
- Set the detector position on the screen and the slit separation.
- The first plot is a geometric sketch with the source, barrier, two slits, and the chosen detector point.
- The second plot scans detector positions and shows a normalized interference pattern from two reference families.
- The path plot shows sampled upper and lower bundles that pass through the two slit regions.
- The complex-plane plot shows upper-family and lower-family mean amplitudes and their sum.
- The Play animation adds sampled paths one by one and updates the cumulative sum.
                """,
                """
- Nastav polohu detektoru na stínítku a vzdálenost štěrbin.
- První graf je geometrické schéma se zdrojem, bariérou, dvěma štěrbinami a zvoleným bodem detektoru.
- Druhý graf skenuje polohu detektoru a ukazuje normalizovaný interferenční obraz ze dvou referenčních rodin drah.
- Graf drah ukazuje vzorkované horní a dolní svazky procházející oblastmi obou štěrbin.
- Graf v komplexní rovině ukazuje střední amplitudy horní a dolní rodiny a jejich součet.
- Animace Play přidává vzorkované dráhy po jedné a průběžně aktualizuje kumulativní součet.
                """
            )
        )

    ctop = st.columns([1, 1, 1, 0.8])
    with ctop[3]:
        if st.button(tr("Reset double-slit defaults", "Reset výchozích hodnot dvojštěrbiny"), use_container_width=True):
            reset_group("slit")
            st.rerun()

    c1, c2, c3 = st.columns(3)
    c1.slider(tr("Detector position on screen", "Poloha detektoru na stínítku"), -1.2, 1.2, step=0.05, key="slit_detector")
    c2.slider(tr("Slit separation", "Vzdálenost štěrbin"), 0.25, 1.5, step=0.05, key="slit_sep")
    c3.slider(tr("Total time T", "Celkový čas T"), 0.5, 5.0, step=0.1, key="slit_T")

    c4, c5, c6, c7, c8, c9 = st.columns(6)
    c4.slider(tr("Mass m", "Hmotnost m"), 0.2, 3.0, step=0.1, key="slit_m")
    c5.slider(tr("Effective ħ", "Efektivní ħ"), 0.05, 1.0, step=0.01, key="slit_hbar")
    c6.slider(tr("Path spread in each family", "Rozptyl drah v každé rodině"), 0.02, 0.8, step=0.01, key="slit_sigma")
    c7.slider(tr("Paths per family", "Počet drah na rodinu"), 120, 1800, step=60, key="slit_npaths")
    c8.slider(tr("Detector scan span", "Rozsah skenu detektoru"), 0.6, 2.0, step=0.05, key="slit_scan_span")
    c9.markdown("")

    data = cached_double_slit_data(
        st.session_state.slit_detector,
        st.session_state.slit_sep,
        st.session_state.slit_T,
        st.session_state.slit_m,
        220,
        st.session_state.slit_npaths,
        st.session_state.slit_sigma,
        11,
        st.session_state.slit_hbar,
        st.session_state.seed,
        st.session_state.slit_scan_span,
    )

    top_mean = np.mean(data["upper_amp"])
    bottom_mean = np.mean(data["lower_amp"])
    total_mean = top_mean + bottom_mean
    phasor_all = np.concatenate([data["upper_amp"], data["lower_amp"]])
    order_idx = np.concatenate([
        np.arange(len(data["upper_amp"])),
        len(data["upper_amp"]) + np.arange(len(data["lower_amp"]))
    ])
    # Build ordered set for animation from both families in stationary-phase order.
    ref_actions = np.concatenate([
        np.angle(data["upper_amp"]),
        np.angle(data["lower_amp"])
    ])
    # Use angle magnitude order only as an intuitive visualization order.
    anim_order = np.argsort(np.abs(ref_actions))[: min(st.session_state.anim_paths, len(ref_actions))]
    anim_paths = np.vstack([data["upper_paths"], data["lower_paths"]])
    ordered_amps = phasor_all[anim_order]
    ordered_indices = anim_order

    met1, met2, met3 = st.columns(3)
    met1.metric(tr("|upper-family mean amplitude|", "|střední amplituda horní rodiny|"), f"{abs(top_mean):.3f}")
    met2.metric(tr("|lower-family mean amplitude|", "|střední amplituda dolní rodiny|"), f"{abs(bottom_mean):.3f}")
    met3.metric(tr("|combined amplitude|", "|kombinovaná amplituda|"), f"{abs(total_mean):.3f}")

    # Schematic figure with explicit slit openings.
    fig_scheme = go.Figure()
    x_source, x_barrier, x_screen = 0.0, 0.50, 1.0
    barrier_w = 0.035
    gap_half = 0.10
    y_lim = max(1.1, 0.9 * st.session_state.slit_scan_span)

    # Barrier body as three filled rectangles so the two slit openings are unmistakable.
    barrier_style = dict(line=dict(width=1), fillcolor="rgba(90, 100, 120, 0.45)")
    fig_scheme.add_shape(type="rect", x0=x_barrier - barrier_w / 2, x1=x_barrier + barrier_w / 2, y0=-y_lim, y1=data["lower_slit_y"] - gap_half, **barrier_style)
    fig_scheme.add_shape(type="rect", x0=x_barrier - barrier_w / 2, x1=x_barrier + barrier_w / 2, y0=data["lower_slit_y"] + gap_half, y1=data["upper_slit_y"] - gap_half, **barrier_style)
    fig_scheme.add_shape(type="rect", x0=x_barrier - barrier_w / 2, x1=x_barrier + barrier_w / 2, y0=data["upper_slit_y"] + gap_half, y1=y_lim, **barrier_style)

    # Slit openings highlighted explicitly.
    slit_style = dict(line=dict(width=2), fillcolor="rgba(255,255,255,0.95)")
    fig_scheme.add_shape(type="rect", x0=x_barrier - barrier_w / 2, x1=x_barrier + barrier_w / 2, y0=data["upper_slit_y"] - gap_half, y1=data["upper_slit_y"] + gap_half, **slit_style)
    fig_scheme.add_shape(type="rect", x0=x_barrier - barrier_w / 2, x1=x_barrier + barrier_w / 2, y0=data["lower_slit_y"] - gap_half, y1=data["lower_slit_y"] + gap_half, **slit_style)

    fig_scheme.add_shape(type="line", x0=x_screen, x1=x_screen, y0=-y_lim, y1=y_lim, line=dict(width=4, dash="dot"))
    fig_scheme.add_trace(go.Scatter(x=[x_source], y=[0.0], mode="markers+text", text=[tr("source", "zdroj")], textposition="bottom right", marker=dict(size=11), name=tr("Source", "Zdroj")))
    fig_scheme.add_trace(go.Scatter(x=[x_barrier, x_barrier], y=[data["upper_slit_y"], data["lower_slit_y"]], mode="markers+text", text=[tr("upper slit", "horní štěrbina"), tr("lower slit", "dolní štěrbina")], textposition="middle right", marker=dict(size=10), name=tr("Slits", "Štěrbiny")))
    fig_scheme.add_trace(go.Scatter(x=[x_screen], y=[data["detector_y"]], mode="markers+text", text=[tr("detector", "detektor")], textposition="middle left", marker=dict(size=11), name=tr("Detector", "Detektor")))
    fig_scheme.add_trace(go.Scatter(x=[x_source, x_barrier, x_screen], y=[0.0, data["upper_slit_y"], data["detector_y"]], mode="lines", line=dict(width=3, dash="dash"), name=tr("Upper reference family", "Horní referenční rodina")))
    fig_scheme.add_trace(go.Scatter(x=[x_source, x_barrier, x_screen], y=[0.0, data["lower_slit_y"], data["detector_y"]], mode="lines", line=dict(width=3, dash="dash"), name=tr("Lower reference family", "Dolní referenční rodina")))
    fig_scheme.update_layout(
        title=tr("Geometry with a barrier and two explicit slit openings", "Geometrie s bariérou a dvěma explicitními štěrbinami"),
        xaxis_title=tr("Propagation direction", "Směr šíření"),
        yaxis_title=tr("Transverse coordinate", "Příčná souřadnice"),
        height=410,
        margin=dict(l=20, r=20, t=48, b=20),
        showlegend=False,
    )
    fig_scheme.update_xaxes(range=[-0.05, 1.08])
    fig_scheme.update_yaxes(range=[-y_lim, y_lim])

    fig_screen = go.Figure()
    fig_screen.add_trace(go.Scatter(x=data["screen_y"], y=data["screen_intensity"], mode="lines", line=dict(width=3), name=tr("Normalized intensity", "Normalizovaná intenzita")))
    fig_screen.add_trace(go.Scatter(x=[data["detector_y"], data["detector_y"]], y=[0, 1.02], mode="lines", line=dict(width=2, dash="dash"), name=tr("Chosen detector", "Zvolený detektor")))
    fig_screen.update_layout(
        title=tr("Interference pattern on the screen", "Interferenční obraz na stínítku"),
        xaxis_title=tr("Detector position", "Poloha detektoru"),
        yaxis_title=tr("Normalized intensity", "Normalizovaná intenzita"),
        height=410,
        margin=dict(l=20, r=20, t=48, b=20),
    )

    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_scheme, use_container_width=True)
    col2.plotly_chart(fig_screen, use_container_width=True)

    fig_paths = go.Figure()
    idx_u = choose_path_indices(data["upper_paths"].shape[0], st.session_state.show_paths)
    idx_l = choose_path_indices(data["lower_paths"].shape[0], st.session_state.show_paths)
    x_prog = np.linspace(0, 1, data["t"].size)
    for row in data["upper_paths"][idx_u]:
        fig_paths.add_trace(go.Scatter(x=x_prog, y=row, mode="lines", opacity=0.14, showlegend=False))
    for row in data["lower_paths"][idx_l]:
        fig_paths.add_trace(go.Scatter(x=x_prog, y=row, mode="lines", opacity=0.14, showlegend=False))
    fig_paths.add_trace(go.Scatter(x=x_prog, y=data["upper_ref"], mode="lines", line=dict(width=4), name=tr("Upper reference", "Horní reference")))
    fig_paths.add_trace(go.Scatter(x=x_prog, y=data["lower_ref"], mode="lines", line=dict(width=4), name=tr("Lower reference", "Dolní reference")))
    fig_paths.add_shape(type="line", x0=0.5, x1=0.5, y0=-y_lim, y1=data["lower_slit_y"] - gap_half, line=dict(width=3))
    fig_paths.add_shape(type="line", x0=0.5, x1=0.5, y0=data["lower_slit_y"] + gap_half, y1=data["upper_slit_y"] - gap_half, line=dict(width=3))
    fig_paths.add_shape(type="line", x0=0.5, x1=0.5, y0=data["upper_slit_y"] + gap_half, y1=y_lim, line=dict(width=3))
    fig_paths.add_trace(go.Scatter(x=[0.5, 0.5], y=[data["upper_slit_y"], data["lower_slit_y"]], mode="markers", marker=dict(size=9), showlegend=False))
    fig_paths.update_layout(
        title=tr("Sampled path bundles passing through the slit regions", "Vzorkované svazky drah procházející oblastmi štěrbin"),
        xaxis_title=tr("Propagation parameter", "Parametr šíření"),
        yaxis_title=tr("Transverse coordinate", "Příčná souřadnice"),
        height=410,
        margin=dict(l=20, r=20, t=48, b=20),
    )

    fig_ph = go.Figure()
    fig_ph.add_trace(go.Scatter(x=np.real(data["upper_amp"]), y=np.imag(data["upper_amp"]), mode="markers", opacity=0.28, name=tr("Upper family", "Horní rodina")))
    fig_ph.add_trace(go.Scatter(x=np.real(data["lower_amp"]), y=np.imag(data["lower_amp"]), mode="markers", opacity=0.28, name=tr("Lower family", "Dolní rodina")))
    fig_ph.add_trace(go.Scatter(x=[0, np.real(top_mean)], y=[0, np.imag(top_mean)], mode="lines+markers", line=dict(width=4), name=tr("Upper mean", "Horní střed")))
    fig_ph.add_trace(go.Scatter(x=[0, np.real(bottom_mean)], y=[0, np.imag(bottom_mean)], mode="lines+markers", line=dict(width=4), name=tr("Lower mean", "Dolní střed")))
    fig_ph.add_trace(go.Scatter(x=[0, np.real(total_mean)], y=[0, np.imag(total_mean)], mode="lines+markers", line=dict(width=5), name=tr("Combined", "Kombinace")))
    th = np.linspace(0, 2 * np.pi, 300)
    fig_ph.add_trace(go.Scatter(x=np.cos(th), y=np.sin(th), mode="lines", line=dict(width=1, dash="dash"), name=tr("Unit circle", "Jednotková kružnice")))
    fig_ph.update_layout(
        title=tr("How the two path families add in the complex plane", "Jak se dvě rodiny drah sčítají v komplexní rovině"),
        xaxis_title="Re",
        yaxis_title="Im",
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
        height=410,
        margin=dict(l=20, r=20, t=48, b=20),
    )

    col3, col4 = st.columns(2)
    col3.plotly_chart(fig_paths, use_container_width=True)
    col4.plotly_chart(fig_ph, use_container_width=True)

    st.plotly_chart(
        make_addition_animation(
            x_prog,
            anim_paths,
            0.5 * (data["upper_ref"] + data["lower_ref"]),
            ordered_indices,
            ordered_amps,
            tr("Play: path-by-path accumulation for the double slit", "Play: skládání drah po jedné pro dvojštěrbinu"),
            tr("Transverse coordinate", "Příčná souřadnice"),
        ),
        use_container_width=True,
    )

    caption(
        "This is a teaching model, not a full wave-optics slit solver. Its purpose is to make the amplitude logic visible: two broad families of paths contribute separate complex numbers that interfere on the screen.",
        "Toto je výukový model, nikoli plný řešič vlnové optiky dvojštěrbiny. Jeho cílem je zviditelnit logiku amplitud: dvě široké rodiny drah dávají oddělená komplexní čísla, která na stínítku interferují.",
    )


# -------------------------------------------------
# Panel: Real time vs imaginary time
# -------------------------------------------------
else:
    section_header(
        "Real time vs imaginary time",
        "Reálný čas vs imaginární čas",
        "Real-time weights oscillate. Imaginary-time weights suppress large-action paths and are numerically calmer.",
        "Váhy v reálném čase oscilují. Váhy v imaginárním čase potlačují dráhy s velkou akcí a numericky jsou klidnější.",
    )

    with st.expander(tr("How to use this panel and what the plots mean", "Jak panel používat a co ukazují grafy"), expanded=True):
        st.markdown(
            tr(
                """
- Choose either the free particle or the harmonic oscillator as the reference model.
- The first plot shows complex-phase cancellation in real time.
- The second plot shows Euclidean action differences in imaginary time.
- The third plot highlights the paths with the largest Euclidean weights.
- The last plot directly compares unit-modulus real-time factors with exponentially damped imaginary-time weights.
                """,
                """
- Vyber si buď volnou částici, nebo harmonický oscilátor jako referenční model.
- První graf ukazuje rušení komplexních fází v reálném čase.
- Druhý graf ukazuje rozdíly eukleidovských akcí v imaginárním čase.
- Třetí graf zvýrazňuje dráhy s největšími eukleidovskými vahami.
- Poslední graf přímo porovnává faktory jednotkové velikosti v reálném čase s exponenciálně tlumenými vahami v imaginárním čase.
                """
            )
        )

    ctop = st.columns([1, 1, 1, 0.8])
    with ctop[3]:
        if st.button(tr("Reset real/imaginary-time defaults", "Reset výchozích hodnot reálného/imaginárního času"), use_container_width=True):
            reset_group("imag")
            st.rerun()

    if "imag_model_code" not in st.session_state:
        st.session_state.imag_model_code = "free"
    model_labels = {
        "free": tr("Free particle", "Volná částice"),
        "ho": tr("Harmonic oscillator", "Harmonický oscilátor"),
    }
    current_code = st.radio(
        tr("Reference model", "Referenční model"),
        ["free", "ho"],
        index=0 if st.session_state.imag_model_code == "free" else 1,
        format_func=lambda x: model_labels[x],
        horizontal=True,
    )
    st.session_state.imag_model_code = current_code

    c1, c2, c3 = st.columns(3)
    c1.slider(tr("Initial position x_a", "Počáteční poloha x_a"), -2.0, 2.0, step=0.1, key="imag_xa")
    c2.slider(tr("Final position x_b", "Koncová poloha x_b"), -2.0, 2.0, step=0.1, key="imag_xb")
    c3.slider(tr("Total time T", "Celkový čas T"), 0.5, 5.0, step=0.1, key="imag_T")

    c4, c5, c6, c7, c8 = st.columns(5)
    c4.slider(tr("Mass m", "Hmotnost m"), 0.2, 3.0, step=0.1, key="imag_m")
    c5.slider(tr("Angular frequency ω", "Úhlová frekvence ω"), 0.2, 4.0, step=0.05, key="imag_omega", disabled=(current_code == "free"))
    c6.slider(tr("Effective ħ", "Efektivní ħ"), 0.05, 1.0, step=0.01, key="imag_hbar")
    c7.slider(tr("Path spread", "Rozptyl drah"), 0.02, 1.0, step=0.01, key="imag_sigma")
    c8.slider(tr("Number of sampled paths", "Počet vzorkovaných drah"), 200, 3000, step=100, key="imag_npaths")

    if current_code == "free":
        data = cached_family_data(
            "free",
            st.session_state.imag_xa,
            st.session_state.imag_xb,
            st.session_state.imag_T,
            st.session_state.imag_m,
            0.0,
            220,
            st.session_state.imag_npaths,
            st.session_state.imag_sigma,
            11,
            st.session_state.seed,
        )
        label = tr("free particle", "volná částice")
    else:
        data = cached_family_data(
            "ho",
            st.session_state.imag_xa,
            st.session_state.imag_xb,
            st.session_state.imag_T,
            st.session_state.imag_m,
            st.session_state.imag_omega,
            220,
            st.session_state.imag_npaths,
            st.session_state.imag_sigma,
            11,
            st.session_state.seed,
        )
        label = tr("harmonic oscillator", "harmonický oscilátor")

    phase_info = phase_data(data["action"], st.session_state.imag_hbar, data["ref_action"], order="stationary")
    delta_e = data["euc_action"] - data["ref_euc_action"]
    weights = np.exp(-delta_e / st.session_state.imag_hbar)
    weights = weights / np.max(weights)

    col1, col2 = st.columns(2)
    col1.plotly_chart(
        make_phasor_figure(phase_info["amps"], phase_info["walk"], tr(f"Real time: oscillatory phases for the {label}", f"Reálný čas: oscilující fáze pro model {label}")),
        use_container_width=True,
    )

    fig_w = go.Figure()
    fig_w.add_trace(go.Histogram(x=delta_e / st.session_state.imag_hbar, nbinsx=55, name="ΔS_E / ħ"))
    fig_w.update_layout(
        title=tr("Imaginary time: Euclidean action differences", "Imaginární čas: rozdíly eukleidovských akcí"),
        xaxis_title=tr("(S_E - S_E,ref) / ħ", "(S_E - S_E,ref) / ħ"),
        yaxis_title=tr("Number of paths", "Počet drah"),
        height=360,
        margin=dict(l=20, r=20, t=48, b=20),
    )
    col2.plotly_chart(fig_w, use_container_width=True)

    idx = np.argsort(weights)[::-1]
    fig_t = go.Figure()
    sample_idx = idx[: min(st.session_state.show_paths, len(idx))]
    for i in sample_idx:
        fig_t.add_trace(go.Scatter(x=data["t"], y=data["paths"][i], mode="lines", opacity=0.22, showlegend=False))
    fig_t.add_trace(go.Scatter(x=data["t"], y=data["x_ref"], mode="lines", line=dict(width=4), name=tr("Reference path", "Referenční dráha")))
    fig_t.update_layout(
        title=tr("Paths with the largest Euclidean weights", "Dráhy s největšími eukleidovskými vahami"),
        xaxis_title=tr("Time t", "Čas t"),
        yaxis_title=tr("Position x(t)", "Poloha x(t)"),
        height=390,
        margin=dict(l=20, r=20, t=48, b=20),
    )

    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=phase_info["delta"] / st.session_state.imag_hbar, y=np.abs(phase_info["amps"]), mode="markers", opacity=0.35, name=tr("|e^{iΔS/ħ}| = 1", "|e^{iΔS/ħ}| = 1")))
    fig_p.add_trace(go.Scatter(x=delta_e / st.session_state.imag_hbar, y=weights, mode="markers", opacity=0.35, name=tr("e^{-ΔS_E/ħ}", "e^{-ΔS_E/ħ}")))
    fig_p.update_layout(
        title=tr("Why Euclidean weights are numerically calmer", "Proč jsou eukleidovské váhy numericky klidnější"),
        xaxis_title=tr("Action difference / ħ", "Rozdíl akcí / ħ"),
        yaxis_title=tr("Weight magnitude", "Velikost váhy"),
        height=390,
        margin=dict(l=20, r=20, t=48, b=20),
    )

    col3, col4 = st.columns(2)
    col3.plotly_chart(fig_t, use_container_width=True)
    col4.plotly_chart(fig_p, use_container_width=True)

    imag_ordered_weights = weights[phase_info["sort_idx"]][: min(st.session_state.anim_paths, len(phase_info["sort_idx"]))]
    anim_indices = phase_info["sort_idx"][: min(st.session_state.anim_paths, len(phase_info["sort_idx"]))]
    anim_amps = phase_info["ordered_amps"][: len(anim_indices)]
    st.plotly_chart(
        make_real_imag_addition_animation(
            data["t"],
            data["paths"],
            data["x_ref"],
            anim_indices,
            anim_amps,
            imag_ordered_weights,
            tr("Play: building the sum in real and imaginary time", "Play: budování součtu v reálném a imaginárním čase"),
            tr("Position x(t)", "Poloha x(t)"),
        ),
        use_container_width=True,
    )

    caption(
        "In real time, every sampled path has unit modulus and differs only by phase, which makes cancellation severe. In imaginary time, large-action paths are exponentially suppressed.",
        "V reálném čase má každá vzorkovaná dráha jednotkovou velikost a liší se jen fází, takže rušení je silné. V imaginárním čase jsou dráhy s velkou akcí exponenciálně potlačeny.",
    )


# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown(
    f'''
    <div class="caption-box">
    <b>{tr("Implementation note", "Poznámka k implementaci")}</b>: {tr(
        "The expensive parts are vectorized: path ensembles are stored as 2D numpy arrays, cubic splines are evaluated for whole batches at once, and actions are computed without Python loops over paths. This keeps the app suitable for GitHub and Streamlit deployment while staying transparent enough for teaching.",
        "Výpočetně náročné části jsou vektorizované: ansámbly drah jsou uložené jako 2D pole numpy, kubické splajny se vyhodnocují pro celé dávky najednou a akce se počítají bez pythonovských smyček přes jednotlivé dráhy. Aplikace tak zůstává vhodná pro GitHub a Streamlit a zároveň dost průhledná pro výuku."
    )}
    </div>
    ''',
    unsafe_allow_html=True,
)
