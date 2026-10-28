# Convex contact compatibility code

`collision_ccd.py` contains the GJK/EPA dependency closure of `ccd` from
[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp/blob/63c21c2c4147995d6e221e6e2586ba12fcf39592/mujoco_warp/_src/collision_gjk.py),
commit `63c21c2c4147995d6e221e6e2586ba12fcf39592`, Apache-2.0.
The upstream copyright header and license are retained.

This is a backport for the existing **3.7.0.1** runtime, not a dependency upgrade.
The full upstream runtime requires newer MuJoCo/Warp APIs and is not installed.

Compatibility changes:

- Retain the 3.7 CCD call signature; replace the newer overflow-report arguments
  with the existing diagnostic print behavior.
- Use the 3.7 `Geom.hfprism` field instead of the newer `Geom.polyvert` field.
- Omit unsupported flex/triangle support branches and unused multicontact code.
- Leave multicontact generation and its supported geometry types in 3.7;
  cylinders do not receive the newer cylinder multicontact feature.
- `collision_kernel.py` retains the installed 3.7 non-heightfield kernel builder,
  translates pairs into a nearby coordinate frame for GJK/EPA, manifold clipping
  **and normal calculation**, then returns only final contact positions to world
  coordinates. This reduces float32 cancellation for micrometre contacts at
  metre-scale world positions. Returning witnesses to world coordinates before
  calculating the normal was tested and rejected because it still produced force
  spikes. Heightfield prisms retain their existing coordinate convention.

The upstream split polytope thresholds, simplex checks, separating-plane tests,
EPA outward-face handling, and convergence rules are retained. We do not lower
all contact thresholds: that experiment produced invalid initial contact forces.

`backend.py` checks the installed version and collision caller hash before
selecting this callable. It does not modify the installed package. Use a fresh
process and `--physics-backend stock` for the original backend. Hashes and the
chosen backend are included in experiment manifests. This compatibility layer
must be revalidated when upgrading the runtime.
