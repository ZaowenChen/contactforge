# Assets and third-party material

The root MIT license applies to the original project code and documentation.
It does **not** automatically relicense third-party material.

## Vendored collision compatibility code

`src/mjlab_contact_prep/_vendor/` is derived from MuJoCo Warp and remains under
Apache License 2.0. Its upstream revision, changes and complete license are in
that directory.

## Simulation assets

`assets/` contains robot, gripper, peg and hole model files retained from the
research workspace. Names and hashes are preserved for experiment
reproducibility. Rights in vendor or CAD-derived assets remain with their
respective owners; the root MIT license does not grant additional rights to
those files. Users are responsible for confirming that their intended use and
redistribution are permitted and for replacing assets when required.

No model checkpoints, raw trajectories, state caches or local runtime packages
are distributed through Git.
