# Changes to Table II

This note records how Table II of the paper (comparison of reachability tools on the 16 benchmarks) changed between the previous version of the paper and the current one, and why.

- **Previous version:** the table of the previous version of the paper, from which the current draft started. Its verdicts are the same as in [arXiv:2411.05554](https://arxiv.org/abs/2411.05554); its times were not compared with the arXiv version.
- **Current version:** the table written by `python -m scripts.paper_table` from `results/summary/` (`results/summary/paper_table.tex`). The current draft of the paper was checked against it cell by cell on 2026-11-08: all 112 cells agree.

In the draft, every changed cell is shown in blue. Until 2026-11-08 the 10 cells whose verdict changed were shown in red.

## Summary

| Cells                      | Count | Meaning                                                    |
| -------------------------- | ----- | ---------------------------------------------------------- |
| Unchanged                  | 45    | Same text as before (mostly N/A, OOM, T/O, Analytical).    |
| New values, same verdict   | 57    | Only the computation time changed.                         |
| Changed verdict            | 10    | The label changed; see [Changed verdicts](#changed-verdicts). |

## What changed in the experiments

These changes apply to the whole table. They explain the new computation times and, together with the reasons given for each cell, the changed verdicts.

- **One machine and one setting per tool.** All tools were re-run on one computer (Apple M5, 32 GB RAM; dReach in Docker under x86-64 emulation).
  - Each tool runs in one setting with its native capabilities: a runner does not split sets or translate a problem for a tool. See `configs/settings/`.
  - The time limit is 1800 s for every tool. HJ also has a memory limit of 8 GB.
  - Times count each tool's core computation, without JIT compilation, MATLAB start-up, CORA derivative generation or container start-up.
- **KRTB.**
  - The sample counts are those at which the sampled extrema of the eigenfunctions have converged.
  - The two KRTB columns time the principal eigenpairs and the reach-time bounds separately.
- **dReach.** The models are generated from the benchmark configs (`tools/dreach/gen_drh.py`). The hand-written `.drh` files used before had errors; they remain in `benchmarks/` on this branch.
- **Runner bugs fixed.** In CORA, a field test was always false. JuliaReach swallowed exceptions. SOSTOOLS wrote a union of sets as an intersection.
- **Example 2 (cart-pole) re-specified.** The changes are marked in blue in `Benchmark_Specifications.pdf`:
  - Tedrake's equations as printed, with the pole angle measured from the hanging vertical.
  - A new LQR gain.
  - Horizon [0, 12] for CP-LQR_REA_CVG.
  - Walls at 1.3 and 3.7, restricted to a box inside the basin of attraction.
- **Label δ-Rea.** dReach's δ-sat is written δ-Rea (footnote c), because δ-sat is not a proof of reachability.

## Changed verdicts

| Benchmark         | Tool       | Before      | Now               | Reason |
| ----------------- | ---------- | ----------- | ----------------- | ------ |
| DUFF_REA_NCVX_LS  | HJ         | N/S         | Rea (16.19)       | The footnote of the previous version said that the toolkit does not support general level sets. That was not accurate: hj_reachability represents any set by its values at the grid nodes, so convexity makes no difference. Only the previous runner was limited, because it recognised boxes and balls only. The runner now puts the target's defining function on the grid. The verdict agrees with the simulation, and the footnote was removed. |
| DUFF_REA_DSJT_BOX | dReach     | Unr (0.12)  | δ-Rea (49.06)     | In the hand-written model, the variable bounds [−1, 1.5]² excluded the initial set (x₂ ∈ [1.8, 1.9]). The initial set was therefore empty and unsat was trivial. The second target referred to a mode that does not exist and was ignored. The benchmark is reachable, so the previous Unr was wrong. The generated model gives δ-sat. |
| DUFF_UNR_CVX_LS   | dReach     | T/O         | Unr (1725.68)     | The time limit is now 1800 s for every tool; the previous runner scripts mostly used 300 s. dReach proves unsat after 1726 s, close to the limit. |
| CP-LQR_REA_CVG    | CORA       | × (4.67)    | Rea (14.71)       | The previous runner could only report "unreachable" (its field test was always false). With the previous horizon of 10, the reachable sets had not entered the target yet. With the fixed runner and the horizon of 12, the reachable sets lie entirely inside the target from t ≈ 9.94 on. This proves reachability. |
| CP-LQR_REA_CVG    | JuliaReach | × (35.60)   | Rea (14.85)       | Same as CORA: the previous runner swallowed exceptions, and the horizon was 10. Now the flowpipe segments lie entirely inside the target from t ≈ 9.94 on. |
| CP-LQR_REA_CVG    | dReach     | T/O         | δ-Rea (9.46)      | Generated model of the re-specified benchmark. The precision is δ = 1e-5, because the target half-width is 1e-3. |
| CP-LQR_REA_USF    | dReach     | Rea (0.43)  | δ-Rea (2.95)      | Only the label changed: δ-sat is now written δ-Rea. |
| CP-LQR_UNR_SAF    | dReach     | T/O         | Unr (0.55)        | Generated model of the re-specified benchmark (wall at 3.7 inside the box X). dReach proves unsat quickly. |
| MAS-CON_UNR_SAF_0 | dReach     | T/O         | Unr (8.21)        | The generated model proves unsat in about 8 s. A run this short means that the previous time-out was not caused by the time limit. It came from the previous hand-written model; which part of that model caused it has not been isolated. |
| MAS-CON_UNR_SAF_1 | dReach     | T/O         | Unr (8.41)        | As for MAS-CON_UNR_SAF_0. |

Every new verdict agrees with the simulation baseline (`python -m scripts.evaluate_results` reports no contradiction). The online report shows each of these results next to the simulated trajectories.

## Note on MAS-CON_REA_BOX

Between 2026-11-06 and 2026-11-08, the draft showed 57.89 s for the KRTB eigenpairs of MAS-CON_REA_BOX. That value came from a run of 2026-11-06. The benchmark was re-run on 2026-11-08, which gave 56.93 s with the same bounds [4.949, 7]. The draft now shows 56.93.

## All cells

In the table below:

- "= x" means the cell is unchanged;
- "a → b" means a new value with the same verdict;
- **bold** means the verdict changed.

The KRTB eigenpairs column has times only, or "Analytical" when the eigenfunctions are known in closed form. Times are in seconds.

| Benchmark | KRTB eigenpairs | KRTB bounds | CORA | JuliaReach | dReach | HJ | SOSTOOLS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DUFF_REA_BOX | 13.42 → 4.81 | Rea (0.07) → Rea (0.01) | Ukn (11.71) → Ukn (9.19) | Ukn (13.70) → Ukn (1.27) | δ-Rea (93.12) → δ-Rea (446.86) | Rea (57.54) → Rea (19.81) | Ukn (1.06) → Ukn (1.70) |
| DUFF_REA_CVX_LS | 15.07 → 4.90 | Rea (0.06) → Rea (0.01) | = N/A | = N/A | δ-Rea (3.13) → δ-Rea (14.31) | Rea (102.41) → Rea (21.61) | Ukn (0.89) → Ukn (1.43) |
| DUFF_REA_NCVX_LS | 13.49 → 4.78 | Rea (0.07) → Rea (0.01) | = N/A | = N/A | δ-Rea (291.29) → δ-Rea (1589.39) | **N/S → Rea (16.19)** | = N/A |
| DUFF_REA_DSJT_BOX | 17.83 → 7.37 | Rea (0.08) → Rea (0.01) | = N/A | = N/A | **Unr (0.12) → δ-Rea (49.06)** | Rea (89.86) → Rea (17.60) | Ukn (1.09) → Ukn (1.56) |
| DUFF_UNR_BOX | 13.95 → 4.87 | Inc (0.07) → Inc (0.01) | Ukn (16.29) → Ukn (9.97) | Ukn (14.32) → Ukn (1.40) | = T/O | Unr (190.15) → Unr (65.02) | Unr (1.52) → Unr (1.33) |
| DUFF_UNR_CVX_LS | 15.10 → 4.84 | Inc (0.08) → Inc (0.01) | = N/A | = N/A | **T/O → Unr (1725.68)** | Unr (103.87) → Unr (22.03) | Unr (0.88) → Unr (1.24) |
| NL-SLC_REA_BOX | = Analytical | PRea (0.04) → PRea (0.01) | = S/E | Ukn (35.25) → Ukn (2.51) | = T/O | Rea (1348.18) → Rea (268.87) | = N/A |
| NL-SLC_UNR_BOX | = Analytical | Unr (0.04) → Unr (0.01) | = S/E | Unr (37.72) → Unr (2.48) | = T/O | Unr (1375.69) → Unr (267.15) | = N/A |
| NL-EIG_BWD_REA_BOX | = Analytical | Rea (0.03) → Rea (0.01) | = N/A | = N/A | = T/O | Rea (33.76) → Rea (6.95) | = N/A |
| NL-EIG_BWD_UNR_BOX | = Analytical | Unr (0.03) → Unr (0.01) | = N/A | = N/A | = T/O | Unr (33.54) → Unr (7.87) | = N/A |
| CP-LQR_REA_CVG | 64.86 → 6.82 | Rea (0.08) → Rea (0.02) | **× (4.67) → Rea (14.71)** | **× (35.60) → Rea (14.85)** | **T/O → δ-Rea (9.46)** | = OOM | = N/A |
| CP-LQR_REA_USF | 54.76 → 55.91 | Rea (0.11) → Rea (0.01) | = N/A | = N/A | **Rea (0.43) → δ-Rea (2.95)** | = OOM | = N/A |
| CP-LQR_UNR_SAF | 21.16 → 54.49 | Unr (0.04) → Unr (0.01) | = N/A | = N/A | **T/O → Unr (0.55)** | = OOM | = N/A |
| MAS-CON_REA_BOX | 801.86 → 56.93 | Rea (0.06) → Rea (0.01) | Rea (77.26) → Rea (52.18) | Rea (1110.07) → Rea (47.37) | = T/O | = OOM | = N/A |
| MAS-CON_UNR_SAF_0 | 782.91 → 60.40 | Unr (0.05) → Unr (0.01) | Unr (76.03) → Unr (51.76) | Unr (1110.22) → Unr (43.81) | **T/O → Unr (8.21)** | = OOM | = N/A |
| MAS-CON_UNR_SAF_1 | 787.40 → 60.45 | Unr (0.06) → Unr (0.01) | Unr (76.38) → Unr (52.18) | Unr (1099.77) → Unr (43.32) | **T/O → Unr (8.41)** | = OOM | = N/A |
