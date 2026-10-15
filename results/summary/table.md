| benchmark | expected | KRTB (setting_1) | HJ (setting_0) | JuliaReach (setting_1) | CORA (setting_1) | SOSBC (setting_1) | dReach (setting_0) |
|---|---|---|---|---|---|---|---|
| CP_LQR_FIN_REA_CONVERGE | reachable | REA witness (2.67 s) | n/a | REA proven (15 s) | REA proven (5.13 s) | n/a | - |
| CP_LQR_FIN_REA_UNSAFE | reachable | REA witness (3.83 s) | unknown (468 s) | REA proven (14.2 s) | REA possible (3.49 s) | n/a | - |
| CP_LQR_FIN_UNR_SAFE | unreachable | UNR est. (3.98 s) | UNR est. (465 s) | UNR proven (14 s) | UNR proven (3.34 s) | n/a | - |
| DUFF_FIN_REA_BALL | reachable | REA witness (6.51 s) | unknown (28.2 s) | REA possible (6.85 s) | REA possible (198 s) | unknown (1.9 s) | - |
| DUFF_FIN_REA_BOX | reachable | REA witness (7.39 s) | unknown (25.2 s) | REA possible (6.74 s) | REA possible (536 s) | unknown (2.24 s) | - |
| DUFF_FIN_REA_DISJOINT_BOX | reachable | REA witness (7.26 s) | REA est. (22.5 s) | REA proven (7.35 s) | REA proven (309 s) | unknown (2.13 s) | - |
| DUFF_FIN_REA_NONCONVEX_LEVEL_SET | reachable | REA witness (4.8 s) | REA est. (19.7 s) | REA possible (2.59 s) | REA possible (46.2 s) | unknown (1.68 s) | - |
| DUFF_FIN_UNR_BALL | unreachable | unknown (5.27 s) | UNR est. (27.7 s) | REA possible (6.81 s) | REA possible (215 s) | UNR proven (1.54 s) | - |
| DUFF_FIN_UNR_BOX | unreachable | unknown (5.68 s) | UNR est. (83 s) | UNR proven (13.3 s) | UNR proven (846 s) | UNR proven (1.96 s) | - |
| MAS_CON_FIN_REA_BOX | reachable | REA witness (22.3 s) | n/a | REA proven (44.6 s) | REA proven (56.2 s) | n/a | - |
| MAS_CON_FIN_UNR_SAFE_0 | unreachable | UNR est. (22.9 s) | n/a | UNR proven (46.4 s) | UNR proven (54.2 s) | n/a | - |
| MAS_CON_FIN_UNR_SAFE_1 | unreachable | UNR est. (22.4 s) | n/a | UNR proven (44.5 s) | UNR proven (54.2 s) | n/a | - |
| NL_EIG_FIN_BACKWARD_REA_BOX | reachable | REA witness (0.289 s) | REA est. (8.8 s) | REA proven (129 s) | REA proven (505 s) | n/a | - |
| NL_EIG_FIN_BACKWARD_UNR_BOX | unreachable | UNR est. (0.00615 s) | UNR est. (8.81 s) | UNR proven (113 s) | UNR proven (493 s) | n/a | - |
| NL_SLC_FIN_REA_BOX | reachable | REA witness (0.0191 s) | REA est. (33.2 s) | REA proven (87.4 s) | REA proven (866 s) | unknown (1.83 s) | - |
| NL_SLC_FIN_UNR_BOX | unreachable | UNR est. (0.0141 s) | UNR est. (33.1 s) | UNR proven (87.2 s) | UNR proven (824 s) | UNR proven (1.67 s) | - |

| tool | UNR proven | REA proven | REA witness | REA possible | REA est. | UNR est. | unknown | n/a | set explosion | timeout | error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| KRTB (setting_1) | 0 | 0 | 9 | 0 | 0 | 5 | 2 | 0 | 0 | 0 | 0 |
| HJ (setting_0) | 0 | 0 | 0 | 0 | 4 | 5 | 3 | 4 | 0 | 0 | 0 |
| JuliaReach (setting_1) | 6 | 6 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| CORA (setting_1) | 6 | 5 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| SOSBC (setting_1) | 3 | 0 | 0 | 0 | 0 | 0 | 5 | 8 | 0 | 0 | 0 |
| dReach (setting_0) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Times are each tool's core computation in seconds: everything needed for the verdict (for KRTB
including the witness simulation), excluding JIT compilation, MATLAB start-up, CORA derivative
generation and container start-up. Measured on: macOS-27.0.1-arm64-arm-64bit.
Only timings from the dedicated x86_64 Linux machine are final.
