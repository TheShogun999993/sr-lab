# 🧠 ⚛️ SR-Lab: Hippocampal Predictive Maps & Lattice Green's Functions

**Open Scientific Intelligence (OSI) Hackathon 2026 Submission**

**SR-Lab** unifies **Reinforcement Learning / Computational Neuroscience** ([Stachenfeld, Botvinick & Gershman, *Nature Neuroscience* 2017](https://doi.org/10.1038/nn.4650)) with **Statistical Physics, Physical Chemistry, and Spectral Graph Theory** by exposing their shared mathematical foundation: the **Markov Resolvent / Lattice Green's Function**

$$M = (I - \gamma T)^{-1} = \sum_{t=0}^{\infty} \gamma^t T^t$$

Standard Large Language Models fail at multi-step spatial navigation, barrier detours, driven advection-diffusion, and energy-landscape transition-state discovery because they tokenize 2D manifolds into 1D text and cannot invert $(I - \gamma T)$ internally. **SR-Lab** equips a scientific LLM agent with an exact **Successor Representation & Spectral Transfer Operator Engine**, plus a **Numeric Grounding Verifier** that checks every numerical claim in the agent's output against the simulation JSON.

---

## ✨ Key Capabilities

1. **Dual-Domain Engine (Neuroscience RL $\leftrightarrow$ Statistical Physics & Chemistry)**:
   - **Track A — Hippocampal Predictive Maps (Presets P1–P7)**: Reproduces Stachenfeld et al. (2017): obstacle-warped place fields (Fig. 2c), backward-skewed place fields on directed 1D tracks (Mehta et al. 2000 / Fig. 3b), zero-shot reward revaluation $V = M \cdot R$ (Fig. 1), entorhinal grid-cell eigenvectors & normalized min-cut doorway subgoals (Figs 6–8, Supp. Fig. 13), boundary geometry distortions across Square/Circle/Trapezoid arenas (Krupic et al. 2015 / Fig. 6), local remapping in Tolman detour mazes (Alvernhe et al. 2011 / Fig. 3c–h), and non-spatial community graphs (Schapiro et al. 2016 / Fig. 4).
   - **Track B — Boltzmann Potential Energy Surfaces & Lattice Transport (Presets P8–P10)**: Computes Metropolis-Arrhenius transition matrices $T(s \to s') \propto \exp(-\Delta E_+ / k_B T)$ on 2D Potential Energy Surfaces $E(x,y)$. Automatically identifies **mountain-pass Transition States (Saddle Points)** via spectral partitioning of $U_1$, models **upstream advection-diffusion wakes** under external electric drift fields $\vec{F}$, and evaluates **zero-shot multi-sink defect capture yields** $V = M \cdot R$ in crystal grain boundaries.
2. **Grounded Tool-Calling Agent (`agent/`)**:
   - Translates plain-English experiment descriptions into validated Pydantic tool calls (`build_grid_environment`, `build_energy_landscape`, `inspect_place_field`, `analyze_1d_track_skew`, `evaluate_reward_revaluation`, `compute_grid_and_subgoals`, `compare_td_vs_analytic`, `run_paper_preset`).
   - Works out-of-the-box with a **Built-in Grounded Scientific NL Parser (zero API keys needed)** and supports **Anthropic Claude, OpenAI, Google Gemini, and Local Ollama**.
   - Includes an automated **Anti-Hallucination Numeric Verifier (`agent/grounding.py`)**.
3. **Executable Benchmark (`SR-Bench`, `bench/sr_bench.py`)**:
   - Evaluates 12 executable tasks across 4 categories (Barrier Warping, Advection & Habit Skew, Spectral Bottlenecks & Saddles, Zero-Shot Revaluation), comparing Euclidean "GPS" baselines (16.7%), Unassisted LLMs (25.0%), and the **SR-Lab Resolvent Agent (100.0%)**.

---

## 🚀 Quickstart

```powershell
cd c:\Users\Sanjay\Documents\antigravity\modest-newton\sr-lab

# 1. Run the automated mathematical, agentic, and benchmark verification suite
python tests/test_sr_lab.py

# 2. Launch the interactive Streamlit Scientific Workbench
streamlit run app.py
```
