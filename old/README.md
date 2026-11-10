# Archived optimization experiments

These files were moved with `git mv`, with their contents preserved:

| Archived path | Original path | Reason |
| --- | --- | --- |
| `correction/FA_grid_search.py` | `src/facet/correction/FA_grid_search.py` | Unused experimental AAS/FARM grid implementation with unresolved model references. No package imports, tests, or example callers reference it. |
| `runners/run_grid_search.py` | `run_grid_search.py` | Unreferenced root runner using removed AAS/FARM arguments and obsolete output options. Replaced by the tested folder CLI. |
| `correction/flex_copy.txt` | `src/facet/correction/flex_copy.txt` | Unused text backup of correction code. |

The supported folder optimizer is `examples/run_matrix_optimization.py`, backed
by `facet.optimization`. `facet.correction.grid_search`, `grid_search_pareto`,
`examples/run_kfold.py`, and `examples/run_grid_search_set.py`
remain because they have public API, tested, or example callers. They are legacy
compatibility workflows, not alternative implementations of the new optimizer.
