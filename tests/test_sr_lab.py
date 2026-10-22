"""Comprehensive Mathematical, Agentic & Benchmark Test Suite for SR-Lab."""
from __future__ import annotations

import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sr.env import build_energy_landscape_env, build_grid_env, build_schapiro_graph, build_track_env
from sr.core import compute_sr_analytic, compute_sr_td, discover_subgoals, spectral_regularize_sr
from sr.metrics import check_row_sums, track_skew_metrics
from sr.presets import PRESETS, run_preset
from agent.tools import LabSession
from agent.llm import run_agent_turn
from agent.grounding import verify_response_numbers
from bench.sr_bench import plot_benchmark_results, run_sr_benchmark


def run_all_tests() -> None:
    fig_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("=== 1. Testing Resolvent Row-Sum Invariant M = (I - gamma*T)^-1 ===")
    for g in (0.50, 0.85, 0.95):
        env = build_grid_env(10, 12, wall_preset="four_rooms")
        M = compute_sr_analytic(env.T, gamma=g)
        chk = check_row_sums(M, gamma=g)
        assert chk["max_abs_error"] < 1e-9, f"Row sum failed for gamma={g}: {chk}"
    print("  [PASS] Row sums equal 1/(1-gamma) to < 1e-9 across all discount factors.")

    print("=== 2. Testing 1D Directed Habit Backward Skew (Mehta 2000) ===")
    trk = build_track_env(41, p_right=0.90)
    M_trk = compute_sr_analytic(trk.T, gamma=0.90)
    sk = track_skew_metrics(M_trk[:, 25], 25)
    assert sk["backward_to_forward_ratio"] > 50.0, f"Expected >50x backward skew, got {sk}"
    assert sk["com_shift_from_target"] < -2.0, f"Expected negative CoM shift, got {sk}"
    print(f"  [PASS] Backward/Forward ratio = {sk['backward_to_forward_ratio']}x, CoM shift = {sk['com_shift_from_target']}.")

    print("=== 3. Testing Doorway Subgoal & Double-Well Saddle-Point Discovery ===")
    env_2r = build_grid_env(11, 23, wall_preset="middle_wall_center_door")
    sub_2r = discover_subgoals(env_2r, gamma=0.95, top_n=2)
    assert (5, 11) in sub_2r["subgoals"] or (5, 10) in sub_2r["subgoals"], f"Doorway missed: {sub_2r['subgoals']}"

    env_pes = build_energy_landscape_env(15, 21, "double_well", kT=0.55, barrier_height=3.6)
    sub_pes = discover_subgoals(env_pes, gamma=0.96, top_n=2)
    saddle = sub_pes["subgoals"][0]
    assert abs(saddle[0] - 7) <= 1 and abs(saddle[1] - 10) <= 1, f"Saddle missed: {saddle}"
    print(f"  [PASS] Doorway subgoals = {sub_2r['subgoals']}; Double-Well Saddle = {saddle}.")

    print("=== 4. Testing Online TD(0) Convergence & Spectral Regularization ===")
    env_td = build_grid_env(8, 8, wall_preset="open")
    td_early = compute_sr_td(env_td, gamma=0.90, eta=0.12, n_steps=800, seed=42)
    reg = spectral_regularize_sr(env_td, gamma=0.90, k_keep=12, M_noisy=td_early["M_hat"])
    td_full = compute_sr_td(env_td, gamma=0.90, eta=0.12, n_steps=12000, seed=42)
    assert reg["corr_recon"] > reg["corr_noisy"], f"Spectral smoothing did not improve early TD: {reg}"
    assert td_full["final_correlation"] > 0.93, f"Full TD correlation too low: {td_full['final_correlation']}"
    print(
        f"  [PASS] Early TD r={reg['corr_noisy']} -> Spectrally Smoothed r={reg['corr_recon']} -> Full TD r={td_full['final_correlation']}."
    )

    print("=== 5. Testing All 10 Flagship Presets (P1-P10) ===")
    for pid in PRESETS:
        env_p, fig_p, metrics_p = run_preset(pid)
        assert env_p.n_states > 0 and metrics_p.get("preset_id") == pid
        if pid in ("P8_double_well_saddle", "P6_tolman_detour", "P7_schapiro_graph"):
            out_path = os.path.join(fig_dir, f"{pid}.png")
            fig_p.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig_p)
    print(f"  [PASS] All {len(PRESETS)} presets executed and generated figures cleanly.")

    print("=== 6. Testing Grounded Scientific Agent & Numeric Anti-Hallucination Verifier ===")
    session = LabSession()
    sample_prompts = [
        "Build two rooms connected by a doorway with gamma 0.95 and find the bottleneck subgoal",
        "Run a 1D track with 90% rightward bias and show backward skew",
        "Build a 2D double-well potential energy surface at kT=0.55 and find the transition state saddle point",
        "Show zero-shot reward revaluation when moving the goal from (1,10) to (1,1)",
    ]
    for pr in sample_prompts:
        turn = run_agent_turn(pr, session, provider="builtin")
        rep = turn["grounding_report"]
        assert rep.passed, f"Grounding failed on prompt '{pr}': unverified={rep.unverified_numbers}\nText={turn['explanation']}"
        plt.close(session.current_fig)

    # Also verify that a deliberately hallucinated number gets caught!
    fake_rep = verify_response_numbers("The peak firing is 987.654 at coordinate (99, 88).", {"peak_value": 2.45})
    assert not fake_rep.passed and 987.654 in fake_rep.unverified_numbers
    print("  [PASS] All agent turns 100% grounded; hallucinated number detector verified.")

    print("=== 7. Running 12-Task Executable Benchmark (SR-Bench) ===")
    bench = run_sr_benchmark()
    fig_b = plot_benchmark_results(bench)
    fig_b.savefig(os.path.join(fig_dir, "figE_sr_bench.png"), dpi=120, bbox_inches="tight")
    plt.close(fig_b)
    assert bench["sr_agent_accuracy_pct"] == 100.0, f"SR Agent did not score 100%: {bench}"
    print(
        f"  [PASS] SR-Bench Scores: Euclidean={bench['euclidean_accuracy_pct']}%, "
        f"Unassisted LLM={bench['naked_llm_accuracy_pct']}%, SR-Lab Agent={bench['sr_agent_accuracy_pct']}% (12/12)."
    )
    print("\nALL SR-LAB TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_all_tests()
