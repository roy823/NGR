# NGR: Nucleate, Grow, Refine

NGR estimates a rigid transformation from tentative 3D correspondences. It builds
a geometric compatibility graph, initializes motif-conditioned phases, develops
them through multiplicative growth, and selects a fitted rigid pose.

The implementation runs on CPU. Phase weights are inference states; the estimator
does not require training.

## Installation

Use Python 3.11:

```bash
pip install .
```

For observed-surface verification:

```bash
pip install ".[surface]"
```

## Correspondence registration

```python
from ngr import register_correspondences

pose = register_correspondences(p, q)
R, t = pose["R"], pose["t"]
aligned = p @ R.T + t
```

`p` and `q` are matched finite NumPy arrays with shape `(N, 3)`. The transformation
maps source points into the target frame. This interface uses the model's native
coordinate units: correspondence refitting and verification use radius `0.04`.

## Indoor observed-surface registration

```python
from ngr import prepare_indoor_surface, register_indoor

source = prepare_indoor_surface(source_points)
target = prepare_indoor_surface(target_points)
pose = register_indoor(p, q, source, target)
```

All coordinates in this interface are in metres. Surface preparation selects the
first observed point in each 5 cm voxel while retaining input order. The indoor
interface uses three-point nuclei, metric correspondence radius 10 cm,
information-weighted surface evidence, and adaptive refinement of additional
pose modes.

## Implementation

- `ngr/_core.py`: compatibility graph, motif ranking, initialization and growth.
- `ngr/_rigid.py`: weighted rigid fitting and correspondence refitting.
- `ngr/_surface.py`: surface support, information weights, mode selection and ICP.
- `ngr/registration.py`: the coordinate-to-pose interfaces.

The equations and inference sequence are described in [the method](docs/method.md).

## Tests

```bash
pip install ".[surface,test]"
python -m pytest
```

Author: Chengrui Xu.
