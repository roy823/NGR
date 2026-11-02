# NGR: Nucleate, Grow, Refine

NGR estimates a rigid transformation from tentative 3D correspondences. Its
crystallization-inspired nuclei initialize correspondence phases, and a
total-reflux-inspired update retains each phase's current composition during
multiplicative enrichment. Weighted rigid fitting produces candidate poses.
The CPU estimator does not require training; phase weights are inference states.

The complete surface-assisted method includes a reusable registration
postprocessor: geometric hypothesis screening, two-scale ICP, and
geometry-informed readout under a fixed candidate budget. The correspondence-only
interface provides registration when observed surfaces are unavailable.

## Installation

Use Python 3.11:

```bash
pip install .
```

For complete surface-assisted registration:

```bash
pip install ".[surface]"
```

## Correspondences only

```python
from ngr import register_correspondences

pose = register_correspondences(p, q, inlier_radius=0.04)
aligned = p @ pose["R"].T + pose["t"]
```

`p` and `q` are matched finite NumPy arrays of shape `(N, 3)`, with `N >= 3`.
Coordinates and `inlier_radius` must use the same physical unit. The returned
translation uses that unit, and the pose maps source points into the target
frame. The native radius is `0.04`; setting another radius scales the complete
correspondence estimator, including its compatibility kernel and refitting.
If no admissible nondegenerate nucleus is available, registration raises
`ValueError`. Triangle ranking uses support from additional correspondences;
exactly three mutually consistent matches can leave that ranking empty.

## Correspondences with observed surfaces

```python
from ngr import register_with_surfaces

pose = register_with_surfaces(
    p, q, source_points, target_points, inlier_radius=0.1, budget=16,
)
```

All four point arrays use the same coordinate frame conventions and unit. Supply
raw observed surfaces; surface preparation is performed inside this interface.
For coordinates in metres, use radius `0.1` for indoor scans or `0.6` for outdoor
driving scans. The function keeps the first observation in each voxel of side
`0.5 * inlier_radius`, preserving input order. Supply observed point arrays with
at least six retained points and nonzero extent.

The surface interface supports both pair and triple nuclei. Grown poses form the
primary candidate pool. Direct fits of triple nuclei supplement the auxiliary
pool; pair nuclei supply grown poses only. Surface evidence selects a queue for
point-to-plane ICP and a guarded information-weighted final readout.

The default `budget=16` and optional `budget=32` cap candidate-refinement attempts
on nested queues. Each attempt uses up to two ICP capture radii, with at most ten
iterations per radius. The native winner is retained as the first candidate;
surface verification selects the final output from the refined candidates.
If every primary surface refinement is invalid, the function raises `ValueError`.

## Implementation

- `ngr/_core.py`: compatibility graph, motif ranking, initialization and growth.
- `ngr/_rigid.py`: weighted rigid fitting and correspondence refitting.
- `ngr/_surface.py`: surface support, information weights, mode selection and ICP.
- `ngr/registration.py`: correspondence and observed-surface interfaces.

Equations, scaling and inference steps are described in [the method](docs/method.md).

## Tests

```bash
pip install ".[surface,test]"
python -m pytest
```

Author: Chengrui Xu.
