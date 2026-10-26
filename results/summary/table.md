| benchmark | expected | KRTB (setting_1) | HJ (setting_0) | JuliaReach (setting_1) | CORA (setting_1) | SOSBC (setting_1) | dReach (setting_0) |
|---|---|---|---|---|---|---|---|
| CP_LQR_FIN_REA_CONVERGE | reachable | REA witness (eigenpairs 2.56 s + bounds 0.0202 s + witness 0.0192 s) | n/a | REA proven (14.6 s) | REA proven (4.16 s) | n/a | REA possible (10.1 s) |
| CP_LQR_FIN_REA_UNSAFE | reachable | REA witness (eigenpairs 3.81 s + bounds 0.00847 s + witness 0.00393 s) | unknown (393 s) | REA proven (14.8 s) | REA possible (2.8 s) | n/a | REA possible (1.72 s) |
| CP_LQR_FIN_UNR_SAFE | unreachable | UNR est. (eigenpairs 3.87 s + bounds 0.00593 s + witness 1.5e-06 s) | UNR est. (399 s) | UNR proven (13.9 s) | UNR proven (2.73 s) | n/a | UNR proven (0.53 s) |
| DUFF_FIN_REA_BALL | reachable | REA witness (eigenpairs 4.68 s + bounds 0.0082 s + witness 1.53 s) | unknown (22.6 s) | REA possible (6.88 s) | REA possible (183 s) | unknown (1.9 s) | REA possible (14.3 s) |
| DUFF_FIN_REA_BOX | reachable | REA witness (eigenpairs 4.57 s + bounds 0.00823 s + witness 2.54 s) | unknown (20.5 s) | REA possible (7.32 s) | REA possible (481 s) | unknown (2.24 s) | REA possible (447 s) |
| DUFF_FIN_REA_DISJOINT_BOX | reachable | REA witness (eigenpairs 6.98 s + bounds 0.0086 s + witness 0.00432 s) | REA est. (18.3 s) | REA proven (8.33 s) | REA proven (302 s) | unknown (2.13 s) | REA possible (49.1 s) |
| DUFF_FIN_REA_NONCONVEX_LEVEL_SET | reachable | REA witness (eigenpairs 4.62 s + bounds 0.0084 s + witness 0.00636 s) | REA est. (16.3 s) | REA possible (2.69 s) | REA possible (27.6 s) | unknown (1.68 s) | REA possible (1.59e+03 s) |
| DUFF_FIN_UNR_BALL | unreachable | unknown (eigenpairs 4.65 s + bounds 0.00819 s + witness 0.224 s) | UNR est. (23.3 s) | REA possible (6.68 s) | REA possible (182 s) | UNR proven (1.54 s) | UNR proven (1.73e+03 s) |
| DUFF_FIN_UNR_BOX | unreachable | unknown (eigenpairs 4.55 s + bounds 0.00819 s + witness 0.333 s) | UNR est. (70.1 s) | UNR proven (13.5 s) | UNR proven (802 s) | UNR proven (1.96 s) | timeout (1.8e+03 s) |
| MAS_CON_FIN_REA_BOX | reachable | REA witness (eigenpairs 19.3 s + bounds 0.0111 s + witness 0.0269 s) | n/a | REA proven (45.1 s) | REA proven (50.4 s) | n/a | timeout (1.8e+03 s) |
| MAS_CON_FIN_UNR_SAFE_0 | unreachable | UNR est. (eigenpairs 19.7 s + bounds 0.00833 s + witness 1.58e-06 s) | n/a | UNR proven (48 s) | UNR proven (49.7 s) | n/a | UNR proven (8.21 s) |
| MAS_CON_FIN_UNR_SAFE_1 | unreachable | UNR est. (eigenpairs 19.6 s + bounds 0.00781 s + witness 6.25e-07 s) | n/a | UNR proven (44.3 s) | UNR proven (51.8 s) | n/a | UNR proven (8.41 s) |
| NL_EIG_FIN_BACKWARD_REA_BOX | reachable | REA witness (eigenpairs 0.00558 s + bounds 0.0062 s + witness 0.228 s) | REA est. (7 s) | REA proven (129 s) | REA proven (492 s) | n/a | timeout (1.8e+03 s) |
| NL_EIG_FIN_BACKWARD_UNR_BOX | unreachable | UNR est. (eigenpairs 0.004 s + bounds 0.00574 s + witness 1.21e-06 s) | UNR est. (7.51 s) | UNR proven (114 s) | UNR proven (492 s) | n/a | timeout (1.8e+03 s) |
| NL_SLC_FIN_REA_BOX | reachable | REA witness (eigenpairs 0.0144 s + bounds 0.00956 s + witness 0.00621 s) | REA est. (29.1 s) | REA proven (91.1 s) | REA proven (817 s) | unknown (1.83 s) | timeout (1.8e+03 s) |
| NL_SLC_FIN_UNR_BOX | unreachable | UNR est. (eigenpairs 0.00556 s + bounds 0.00939 s + witness 1.38e-06 s) | UNR est. (29.4 s) | UNR proven (87.8 s) | UNR proven (768 s) | UNR proven (1.67 s) | timeout (1.8e+03 s) |

| tool | UNR proven | REA proven | REA witness | REA possible | REA est. | UNR est. | unknown | n/a | set explosion | timeout | error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| KRTB (setting_1) | 0 | 0 | 9 | 0 | 0 | 5 | 2 | 0 | 0 | 0 | 0 |
| HJ (setting_0) | 0 | 0 | 0 | 0 | 4 | 5 | 3 | 4 | 0 | 0 | 0 |
| JuliaReach (setting_1) | 6 | 6 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| CORA (setting_1) | 6 | 5 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| SOSBC (setting_1) | 3 | 0 | 0 | 0 | 0 | 0 | 5 | 8 | 0 | 0 | 0 |
| dReach (setting_0) | 4 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 6 | 0 |

Times are each tool's core computation in seconds: everything needed for the verdict, excluding
JIT compilation, MATLAB start-up, CORA derivative generation and container start-up. KRTB's is
split into the principal eigenpairs (eigenvalues and eigenfunction values on the samples), the
reach-time bounds and the witness simulation. Measured on: macOS-27.0.1-arm64-arm-64bit.
Only timings from the dedicated x86_64 Linux machine are final.
