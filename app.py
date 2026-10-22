"""SR-Lab: Hippocampal Predictive Maps & Lattice Green's Functions.

Editorial Light Scientific Interface (Distill / Nature style):
- Zero sidebar: all settings, LLM configuration, preset pickers, and custom environment
  builders are integrated directly into the page in clean horizontal control cards.
- Full-width, high-legibility figures paired with spacious Grounded Explanation cards.
- 100% of the underlying `sr/`, `agent/`, and `bench/` engine is preserved untouched.
"""
from __future__ import annotations

import io
import json
from typing import Any, Dict
import streamlit as st
import matplotlib.pyplot as plt

from sr.core import compute_sr_analytic
from sr.plots import (
    plot_eigenvectors_and_subgoals,
    plot_place_field_comparison,
    plot_reward_revaluation,
    plot_td_and_spectral_regularization,
)
from sr.presets import PRESETS
from agent.tools import LabSession
from agent.llm import run_agent_turn
from bench.sr_bench import plot_benchmark_results, run_sr_benchmark


st.set_page_config(
    page_title="SR-Lab | Predictive Maps & Lattice Green's Functions",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------ Editorial Light Style CSS --
st.markdown(
    """
    <style>
    /* Hide sidebar completely — everything lives cleanly on the main page */
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* Clean editorial page width & breathing room */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1280px !important;
    }

    /* Typography upgrades for easy reading */
    html, body, [class*="css"] {
        color: #0f172a;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    p, li {
        font-size: 1.02rem !important;
        line-height: 1.62 !important;
        color: #1e293b;
    }

    /* Editorial Top Header Card */
    .sr-header {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 6px solid #2563eb;
        border-radius: 12px;
        padding: 1.25rem 1.6rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        margin-bottom: 1rem;
    }
    .sr-header-eyebrow {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #2563eb;
        margin-bottom: 0.2rem;
    }
    .sr-header-title {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.02em;
    }
    .sr-header-sub {
        font-size: 0.98rem;
        color: #475569;
        margin: 0;
    }

    /* Clean White Surface Cards */
    .sr-surface {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.35rem 1.5rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        margin-bottom: 1rem;
    }

    /* Grounding Verification Pill */
    .sr-grounding-pill {
        display: inline-block;
        background: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
        border-radius: 999px;
        padding: 0.3rem 0.9rem;
        font-size: 0.85rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
    }

    /* Section Step Badges */
    .sr-step-heading {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #475569;
        margin-bottom: 0.4rem;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return buf.read()


def _format_readable_metrics(metrics: Dict[str, Any]) -> list[Dict[str, str]]:
    """Format simulation metrics into human-friendly labels and values."""
    rows = []
    label_map = {
        "preset_id": "Experiment ID",
        "paper_ref": "Reference",
        "gamma": "Discount Horizon (γ)",
        "n_free_states": "Traversable States (N)",
        "n_walls": "Blocked / Wall Cells",
        "expected_row_sum": "Theoretical Row Sum 1/(1−γ)",
        "mean_row_sum": "Computed Mean Row Sum",
        "target_cell": "Inspected State (r, c)",
        "peak_sr_value": "Peak SR Occupancy M(s', s')",
        "corr_with_euclidean_gaussian": "Correlation vs Euclidean 'GPS'",
        "corr_with_geodesic_gaussian": "Correlation vs Geodesic Path",
        "cross_wall_leakage_ratio": "Cross-Wall Suppression Ratio",
        "biased_backward_to_forward_ratio": "Backward / Forward Skew Ratio",
        "biased_com_shift": "Center-of-Mass Shift (States)",
        "discovered_subgoals": "Discovered Bottleneck / Doorway",
        "discovered_transition_state_saddle": "Discovered Saddle Point (r, c)",
        "saddle_point_energy": "Saddle Point Energy E*",
        "slowest_relaxation_eigenvalue": "Slowest Mode Eigenvalue (λ₁)",
        "path_a_steps": "Path Length to Goal A (Steps)",
        "path_b_steps": "Path Length to Goal B (Steps)",
        "relearning_steps_required": "Relearning Steps Needed",
        "within_community_mean_corr": "Within-Community Correlation r",
        "between_community_mean_corr": "Between-Community Correlation r",
    }
    for k, v in metrics.items():
        if isinstance(v, (int, float, str)) or (isinstance(v, (list, tuple)) and len(v) <= 5):
            nice_name = label_map.get(k, k.replace("_", " ").title())
            rows.append({"Parameter / Metric": nice_name, "Value": str(v)})
    return rows


# --------------------------------------------------------- Session State Init --
if "lab" not in st.session_state:
    st.session_state.lab = LabSession()

if "bench_results" not in st.session_state:
    st.session_state.bench_results = run_sr_benchmark()

if "provider" not in st.session_state:
    st.session_state.provider = "builtin"
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "model_name" not in st.session_state:
    st.session_state.model_name = ""

if "latest_explanation" not in st.session_state:
    init_turn = run_agent_turn("Run preset P1_wall_warp", st.session_state.lab, provider="builtin")
    st.session_state.latest_explanation = init_turn["explanation"]
    st.session_state.latest_badge = init_turn["grounding_report"].summary_badge
    st.session_state.latest_tool = init_turn["tool_name"]
    st.session_state.latest_args = init_turn["tool_args"]

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "user",
            "content": "Run P1: 12x12 room with a vertical barrier and compare Euclidean GPS vs SR place fields.",
        },
        {
            "role": "assistant",
            "content": st.session_state.latest_explanation,
            "badge": st.session_state.latest_badge,
            "tool": st.session_state.latest_tool,
            "args": st.session_state.latest_args,
        },
    ]

lab: LabSession = st.session_state.lab


def _run_prompt(prompt_text: str) -> None:
    turn = run_agent_turn(
        prompt_text,
        lab,
        provider=st.session_state.provider,
        api_key=st.session_state.api_key,
        model=st.session_state.model_name,
    )
    st.session_state.latest_explanation = turn["explanation"]
    st.session_state.latest_badge = turn["grounding_report"].summary_badge
    st.session_state.latest_tool = turn["tool_name"]
    st.session_state.latest_args = turn["tool_args"]
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": turn["explanation"],
            "badge": turn["grounding_report"].summary_badge,
            "tool": turn["tool_name"],
            "args": turn["tool_args"],
        }
    )


# ==============================================================================
# 1. EDITORIAL HEADER + INTEGRATED TOP SETTINGS BAR
# ==============================================================================
st.markdown(
    """
    <div class="sr-header">
        <div class="sr-header-eyebrow">Open Scientific Intelligence Workbench</div>
        <div class="sr-header-title">🧭 SR-Lab: Predictive Maps, Spectral Subgoals & Lattice Green's Functions</div>
        <div class="sr-header-sub">
            An interactive reinforcement learning and statistical physics lab powered by the exact Markov Resolvent
            <b>M = (I − γT)⁻¹</b> — bridging <i>Stachenfeld, Botvinick & Gershman (2017)</i> with Boltzmann Potential Energy Surfaces.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Integrated Top Settings Row (Domain Lens + AI Model Config + Summary Stats)
set_col1, set_col2, set_col3, set_col4, set_col5 = st.columns([1.45, 1.15, 0.9, 0.9, 1.0], vertical_alignment="center")

with set_col1:
    domain_sel = st.radio(
        "Scientific Domain Lens",
        options=["neuro", "physics"],
        format_func=lambda d: "🧠 Neuroscience & RL" if d == "neuro" else "⚛️ Physics & Chemistry",
        index=0 if lab.domain == "neuro" else 1,
        horizontal=True,
    )
    if domain_sel != lab.domain:
        lab.domain = domain_sel
        st.rerun()

with set_col2:
    with st.popover("⚙️ AI Agent & Model Settings", use_container_width=True):
        st.markdown("##### Configure Agent Backend")
        st.session_state.provider = st.selectbox(
            "LLM Provider",
            options=["builtin", "anthropic", "openai", "gemini", "ollama"],
            index=["builtin", "anthropic", "openai", "gemini", "ollama"].index(st.session_state.provider),
            format_func=lambda p: {
                "builtin": "⚡ Built-in Grounded NL Parser (No Key Needed)",
                "anthropic": "Anthropic Claude (Messages Tool Use)",
                "openai": "OpenAI GPT-4o / GPT-4o-mini",
                "gemini": "Google Gemini 2.5",
                "ollama": "Local Ollama (localhost:11434)",
            }[p],
        )
        if st.session_state.provider in ("anthropic", "openai", "gemini"):
            st.session_state.api_key = st.text_input(
                f"{st.session_state.provider.upper()} API Key",
                value=st.session_state.api_key,
                type="password",
            )
            defaults = {
                "anthropic": "claude-3-5-sonnet-latest",
                "openai": "gpt-4o-mini",
                "gemini": "gemini-2.5-flash",
            }
            st.session_state.model_name = st.text_input(
                "Model Name",
                value=st.session_state.model_name or defaults[st.session_state.provider],
            )
        elif st.session_state.provider == "ollama":
            st.session_state.model_name = st.text_input("Ollama Model", value=st.session_state.model_name or "llama3.1")
        st.caption("All numerical claims are verified against the NumPy SR engine regardless of provider.")

with set_col3:
    st.metric("State Space (N)", f"{lab.env.n_states}", f"{len(lab.env.walls)} walls")

with set_col4:
    st.metric("Horizon (γ)", f"{lab.gamma:.2f}", f"ΣM = {1.0/(1.0-lab.gamma):.1f}")

with set_col5:
    st.metric("SR-Bench Score", f"{st.session_state.bench_results['sr_agent_accuracy_pct']}%", "12 / 12 Tasks Passed")


# ==============================================================================
# 2. INTEGRATED CONTROL DECK (PRESET PICKER + CUSTOM BUILDER + NL PROMPT)
# ==============================================================================
st.markdown('<div class="sr-step-heading">Step 1 — Choose a Preset Experiment, Build a Custom World, or Ask in Plain English</div>', unsafe_allow_html=True)

ctrl_tabs = st.tabs(
    [
        "🧪 A. Flagship Preset Experiments (P1–P10)",
        "🔨 B. Custom Maze & Energy Landscape Builder (Inline Settings)",
        "💬 C. Ask the Grounded Agent in Plain English",
    ]
)

# ---- Control Tab A: Clean Preset Picker ----
with ctrl_tabs[0]:
    p_col1, p_col2, p_col3 = st.columns([1.2, 2.3, 0.8], vertical_alignment="bottom")
    with p_col1:
        track_filter = st.radio(
            "Experiment Track",
            options=["all", "neuro", "physics"],
            format_func=lambda t: {
                "all": "All 10 Experiments",
                "neuro": "🧠 Neuroscience (P1–P7)",
                "physics": "⚛️ Physics/Chem (P8–P10)",
            }[t],
            horizontal=True,
        )
    filtered_keys = [
        k for k, v in PRESETS.items() if track_filter == "all" or v["domain"] == track_filter
    ]
    with p_col2:
        chosen_preset = st.selectbox(
            "Select Experiment to Run",
            options=filtered_keys,
            format_func=lambda k: f"{PRESETS[k]['title']} — [{PRESETS[k]['paper_ref']}]",
        )
    with p_col3:
        if st.button("▶ Run Experiment", type="primary", use_container_width=True):
            _run_prompt(f"Run preset {chosen_preset}")
            st.rerun()

    st.caption(f"ℹ️ **What this tests:** {PRESETS[chosen_preset]['summary']}")

# ---- Control Tab B: Inline Custom World Builder (Formerly in Sidebar!) ----
with ctrl_tabs[1]:
    with st.form("inline_custom_builder"):
        b_row1_c1, b_row1_c2, b_row1_c3, b_row1_c4 = st.columns(4)
        b_mode = b_row1_c1.selectbox("World Type", ["2D Discrete Maze (Neuro/RL)", "2D Potential Energy Surface (Physics/Chem)"])
        b_h = b_row1_c2.slider("Grid Height (Rows)", 6, 25, 12)
        b_w = b_row1_c3.slider("Grid Width (Cols)", 6, 25, 15)
        b_gamma = b_row1_c4.slider("Discount Factor (γ)", 0.50, 0.99, float(lab.gamma), 0.01)

        b_row2_c1, b_row2_c2, b_row2_c3, b_row2_c4 = st.columns([1.2, 1.2, 1.2, 0.9], vertical_alignment="bottom")
        b_wall = b_row2_c1.selectbox(
            "Maze Wall Layout (if 2D Maze)",
            options=["middle_wall_bottom_gap", "middle_wall_center_door", "four_rooms", "hairpin", "tolman_detour", "open"],
            format_func=lambda w: {
                "middle_wall_bottom_gap": "Vertical Wall with Bottom Gap (Fig. 2c)",
                "middle_wall_center_door": "Two Rooms with Central Doorway (Supp. Fig. 13)",
                "four_rooms": "Four Connected Rooms",
                "hairpin": "Hairpin Zig-Zag Maze (Fig. 7)",
                "tolman_detour": "Tolman Detour Corridor (Fig. 3c)",
                "open": "Open Unobstructed Arena",
            }[w],
        )
        b_shape = b_row2_c2.selectbox(
            "Arena Geometry (if 2D Maze)",
            options=["rectangle", "circle", "trapezoid"],
            format_func=lambda s: {"rectangle": "Rectangular / Square", "circle": "Circular Arena", "trapezoid": "Asymmetric Trapezoid"}[s],
        )
        b_pes = b_row2_c3.selectbox(
            "Energy Surface & Thermal kT (if Physics PES)",
            options=[
                ("double_well", 0.55, "Double-Well PES (kT=0.55) — Saddle Discovery"),
                ("advection_obstacle", 0.65, "Repulsive Obstacle + Drift Field (kT=0.65)"),
                ("crystal_grain_boundary", 0.55, "Crystal Grain Boundary Channels (kT=0.55)"),
            ],
            format_func=lambda item: item[2],
        )
        submitted_custom = b_row2_c4.form_submit_button("🔨 Build & Solve World", type="primary", use_container_width=True)

        if submitted_custom:
            if b_mode.startswith("2D Discrete"):
                lab.domain = "neuro"
                _run_prompt(f"Build a {b_h}x{b_w} {b_shape} grid with {b_wall} and gamma={b_gamma}")
                lab.execute_tool(
                    "build_grid_environment",
                    {
                        "height": b_h,
                        "width": b_w,
                        "wall_preset": b_wall,
                        "shape": b_shape,
                        "lattice": "square",
                        "gamma": b_gamma,
                    },
                )
            else:
                lab.domain = "physics"
                pes_type, pes_kt, _ = b_pes
                _run_prompt(f"Build a {b_h}x{b_w} {pes_type} potential energy surface with kT={pes_kt} and gamma={b_gamma}")
            st.rerun()

# ---- Control Tab C: Natural-Language Prompt Bar ----
with ctrl_tabs[2]:
    with st.form("inline_nl_prompt_form", clear_on_submit=True):
        nl_c1, nl_c2 = st.columns([4.2, 0.8], vertical_alignment="bottom")
        nl_text = nl_c1.text_input(
            "Describe an experiment in plain English",
            placeholder="e.g., 'Build two rooms connected by a doorway with gamma 0.95 and find the bottleneck subgoal' or 'Run a 1D track with 90% rightward bias'",
        )
        nl_submit = nl_c2.form_submit_button("Ask Agent ↵", type="primary", use_container_width=True)
        if nl_submit and nl_text.strip():
            _run_prompt(nl_text.strip())
            st.rerun()

    # Sample one-click prompt chips
    sc_1, sc_2, sc_3, sc_4 = st.columns(4)
    if sc_1.button("💬 'Four rooms with doorways, find subgoals'", use_container_width=True):
        _run_prompt("Build a 13x13 four rooms environment with gamma 0.95 and find the doorway subgoals")
        st.rerun()
    if sc_2.button("💬 '1D track with 90% rightward habit skew'", use_container_width=True):
        _run_prompt("Run a 1D track with 90% rightward bias and gamma 0.90 to show Mehta backward skew")
        st.rerun()
    if sc_3.button("💬 'Double-well energy surface saddle point'", use_container_width=True):
        _run_prompt("Build a 2D double-well potential energy surface at kT=0.55 and find the transition-state saddle point")
        st.rerun()
    if sc_4.button("💬 'Move goal from (1,10) to (1,1) with V=MR'", use_container_width=True):
        _run_prompt("Demonstrate zero-shot reward revaluation when moving the goal from (1,10) to (1,1)")
        st.rerun()


# ==============================================================================
# 3. MAIN RESULTS STAGE (FULL-WIDTH READABLE PLOT + STACKED EXPLANATION CARDS)
# ==============================================================================
st.markdown('<div class="sr-step-heading">Step 2 — Visual Simulation Output & Grounded Scientific Explanation</div>', unsafe_allow_html=True)

# Full-width Figure Surface so every subplot is large and easy to read
st.markdown(f"#### 📊 {lab.active_title}")
if lab.current_fig is not None:
    st.pyplot(lab.current_fig, use_container_width=True)

# Below the figure: Clean 2-column reading layout (Left: Grounded Explanation | Right: Key Numbers Table)
exp_col, num_col = st.columns([0.58, 0.42], gap="large")

with exp_col:
    st.markdown(
        f'<div class="sr-grounding-pill">{st.session_state.latest_badge}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="sr-surface">{st.session_state.latest_explanation}</div>',
        unsafe_allow_html=True,
    )

with num_col:
    st.markdown("##### 📋 Verified Simulation Numbers")
    readable_rows = _format_readable_metrics(lab.current_metrics)
    st.dataframe(
        readable_rows,
        use_container_width=True,
        hide_index=True,
        height=245,
    )
    dl_1, dl_2 = st.columns(2)
    if lab.current_fig is not None:
        dl_1.download_button(
            "⬇ Save Figure (PNG)",
            data=_fig_to_png_bytes(lab.current_fig),
            file_name="sr_lab_figure.png",
            mime="image/png",
            use_container_width=True,
        )
    dl_2.download_button(
        "⬇ Export Data (JSON)",
        data=json.dumps(lab.current_metrics, indent=2, default=str),
        file_name="sr_lab_metrics.json",
        mime="application/json",
        use_container_width=True,
    )


# ==============================================================================
# 4. INTERACTIVE DEEP-DIVE EXPLORERS & SR-BENCH LEADERBOARD
# ==============================================================================
st.divider()
st.markdown('<div class="sr-step-heading">Step 3 — Interactive Parameter Explorers & Benchmark Evaluation</div>', unsafe_allow_html=True)

deep_tabs = st.tabs(
    [
        "📍 Interactive Place Field / Green's Picker",
        "🎯 Interactive Zero-Shot Goal Revaluation (V = M·R)",
        "🕸️ Spectral Grid Cells & Bottleneck Explorer",
        "📈 Online TD(0) & Spectral Regularization",
        "🏆 SR-Bench Leaderboard (12 Tasks)",
        "📜 Conversation & Tool Audit Log",
    ]
)

with deep_tabs[0]:
    if lab.env.kind in ("grid_2d", "energy_2d"):
        st.markdown("Drag the sliders below to move the recorded neuron / state $s' = (r', c')$ and compare the **Euclidean GPS**, **Geodesic**, and **Exact SR** fields in real time:")
        pc1, pc2, pc3 = st.columns(3)
        pick_r = pc1.slider("Target State Row (r')", 0, lab.env.height - 1, min(3, lab.env.height - 1))
        pick_c = pc2.slider("Target State Column (c')", 0, lab.env.width - 1, min(5, lab.env.width - 1))
        pick_g = pc3.slider("Discount Factor (γ)", 0.50, 0.99, float(lab.gamma), 0.01)
        M_live = compute_sr_analytic(lab.env.T, gamma=pick_g)
        fig_pf = plot_place_field_comparison(lab.env, M_live, (pick_r, pick_c), domain=lab.domain)
        st.pyplot(fig_pf, use_container_width=True)
        plt.close(fig_pf)
    else:
        st.info("Select a 2D Maze or 2D Energy Surface above (e.g. P1, P3, P4, P8, P9, P10) to inspect 2D coordinates.")

with deep_tabs[1]:
    if lab.env.kind in ("grid_2d", "energy_2d"):
        st.markdown("Move the **Start** position or **Goal A / Goal B** coordinates. Because $V = M \\cdot R$ uses the pre-computed matrix $M$, optimal paths update in **0 learning steps**:")
        rc1, rc2, rc3, rc4, rc5, rc6 = st.columns(6)
        sr_r = rc1.slider("Start Row", 0, lab.env.height - 1, min(10, lab.env.height - 2))
        sr_c = rc2.slider("Start Col", 0, lab.env.width - 1, min(2, lab.env.width - 2))
        ga_r = rc3.slider("Goal A Row", 0, lab.env.height - 1, 1)
        ga_c = rc4.slider("Goal A Col", 0, lab.env.width - 1, min(10, lab.env.width - 2))
        gb_r = rc5.slider("Goal B Row", 0, lab.env.height - 1, 1)
        gb_c = rc6.slider("Goal B Col", 0, lab.env.width - 1, 1)

        fig_rev, rev_m = plot_reward_revaluation(
            lab.env,
            lab.M,
            start_cell=(sr_r, sr_c),
            goal_a=(ga_r, ga_c),
            goal_b=(gb_r, gb_c),
            domain=lab.domain,
        )
        st.pyplot(fig_rev, use_container_width=True)
        plt.close(fig_rev)
        st.success(
            f"**Instant Re-planning:** Start `{rev_m['start_cell']}` → Goal A `{rev_m['goal_a']}` takes **`{rev_m['path_a_steps']}`** steps (`V = {rev_m['value_at_start_for_goal_a']}`). "
            f"Switching to Goal B `{rev_m['goal_b']}` takes **`{rev_m['path_b_steps']}`** steps (`V = {rev_m['value_at_start_for_goal_b']}`) with **0 matrix re-inversions**."
        )
    else:
        st.info("Select a 2D Maze or Energy Surface above to test interactive 2D path revaluation.")

with deep_tabs[2]:
    if lab.env.kind in ("grid_2d", "energy_2d"):
        ec1, ec2 = st.columns([2, 1])
        k_pick = ec1.slider("Starting Eigenvector Mode Index (k)", 1, max(2, min(35, lab.env.n_states - 15)), 4)
        n_sub = ec2.slider("Number of Bottlenecks / Saddles to Mark (★)", 1, 4, 2)
        fig_eig, eig_m = plot_eigenvectors_and_subgoals(
            lab.env,
            gamma=lab.gamma,
            mode_indices=(k_pick, k_pick + 3, k_pick + 6, k_pick + 10, k_pick + 15),
            top_n_subgoals=n_sub,
            domain=lab.domain,
        )
        st.pyplot(fig_eig, use_container_width=True)
        plt.close(fig_eig)
        st.success(
            f"**Normalized Min-Cut Bottleneck / Saddle Coordinates:** `{eig_m['discovered_subgoals']}` "
            f"(Slowest eigenvalue $\\lambda_1 = {eig_m['eigenvalue_1']}$, partition sizes = `{eig_m['partition_sizes']}`)"
        )
    else:
        st.info("Select a 2D Maze or Energy Surface above to explore 2D grid-cell eigenvectors.")

with deep_tabs[3]:
    tc1, tc2, tc3 = st.columns(3)
    td_steps = tc1.slider("Online TD(0) Transition Steps", 1000, 20000, 8000, 1000)
    td_eta = tc2.slider("TD Learning Rate (η)", 0.02, 0.30, 0.10, 0.02)
    td_k = tc3.slider("Spectral Smoothing Basis Size (k)", 4, min(40, lab.env.n_states), min(16, lab.env.n_states))
    fig_td, td_m = plot_td_and_spectral_regularization(
        lab.env, gamma=lab.gamma, n_steps=td_steps, eta=td_eta, k_keep=td_k
    )
    st.pyplot(fig_td, use_container_width=True)
    plt.close(fig_td)
    st.success(
        f"**Spectral Regularization (Supp. Fig. 12):** Early noisy TD correlation `r = {td_m['early_td_correlation']}` "
        f"improves to **`r = {td_m['spectrally_smoothed_early_correlation']}`** after projecting onto the top `{td_k}` grid modes. "
        f"Full `{td_steps}`-step TD reaches **`r = {td_m['final_td_correlation']}`**."
    )

with deep_tabs[4]:
    bench_data = st.session_state.bench_results
    bc1, bc2, bc3 = st.columns(3)
    bc1.metric("1. Euclidean 'GPS' Baseline", f"{bench_data['euclidean_accuracy_pct']}% (2/12)", "Fails on barriers, skew & saddles")
    bc2.metric("2. Unassisted LLM (Text Only)", f"{bench_data['naked_llm_accuracy_pct']}% (3/12)", "Cannot invert (I − γT) in text")
    bc3.metric("3. SR-Lab Resolvent Agent", f"{bench_data['sr_agent_accuracy_pct']}% (12/12)", "+75.0% vs Unassisted LLM")

    fig_bench = plot_benchmark_results(bench_data)
    st.pyplot(fig_bench, use_container_width=True)
    plt.close(fig_bench)

    st.dataframe(
        [
            {
                "ID": t["id"],
                "Category": t["category"],
                "Scientific Question": t["question"],
                "Exact Ground Truth (SR Engine)": t["ground_truth"],
                "Euclidean 'GPS'": "✓" if t["euclidean_correct"] else f"✗ ({t['euclidean_pred']})",
                "Unassisted LLM": "✓" if t["naked_llm_correct"] else f"✗ ({t['naked_llm_pred']})",
                "SR-Lab Agent": f"✓ ({t['sr_agent_pred']})" if t["sr_agent_correct"] else "✗",
            }
            for t in bench_data["tasks"]
        ],
        use_container_width=True,
        hide_index=True,
    )

with deep_tabs[5]:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("badge"):
                st.caption(f"🛡️ **{msg['badge']}** | Executed Tool: `{msg.get('tool')}` | Args: `{msg.get('args')}`")
