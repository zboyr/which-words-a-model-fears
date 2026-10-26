# x115: Path A and Path B direction geometry

Parent: [Experiment index](README.md).

The saved Qwen3.5-4B Path A probe axis was compared with Path B's global
lexical shifts and deployed fold-consensus vectors in residual coordinates.
Path A uses state 31; Path B's deployed intervention uses state 30.
The same-state comparison uses Path B's state-31 global shift.
All vectors have 2,560 coordinates. Path A's supervision is sampled
`refusal2` frequency; the measurements here are vector geometry.

At state 31, the signed cosine is 0.188217 at `tbg_minus1` and 0.225778
at `tbg`, with unsigned-axis angles 79.151 and 76.951 degrees. Against
Path B's deployed state-30 fold consensus, cosine is 0.176560 and
0.198556, respectively. Individual-fold cosines span 0.171280–0.180495
and 0.192292–0.203285. Path A's saved probe and adapter axes have cosine
1.000000, as does the saved raw probe with the direction reconstructed
by undoing feature standardization.

Implementation: `code/x115_direction_geometry_temp.py`.
Results and SHA-256 input provenance:
`data/results/x115_path_a_b_direction_geometry.json`.
Run with `python3 code/x115_direction_geometry_temp.py`.
