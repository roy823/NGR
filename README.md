# NGR: Nucleate, Grow, Refine

NGR is a training-free CPU reference implementation for rigid 3D registration.
Its capacity-aware NGR-LS generator preserves alternative correspondence phases,
retained-composition growth enriches their support, and weighted fitting
produces pose hypotheses. Surface finishing allocates a bounded refinement
budget; complementary release selects the final pose using geometric support
and residual concentration.

The implementation uses NumPy, SciPy and PyTorch for array operations, with
Open3D for point-to-plane ICP.

## Installation

Use Python 3.11:

    pip install .

For observed-surface registration:

    pip install ".[surface]"

## Complete registration

    from ngr import register_with_surfaces

    pose = register_with_surfaces(
        p, q, source_points, target_points,
        inlier_radius=0.1, budget=32,
    )
    aligned = source_points @ pose["R"].T + pose["t"]
    attempts = pose["refinement_count"]

The matched arrays p and q have shape (N, 3), with N at least three.
Surface arrays contain the observed source and target points. All coordinates,
translations and the positive inlier radius use the same physical unit.

The returned R has shape (3, 3), and t has shape (3,). They map source into
target. Surface preparation keeps the first observation in each voxel of side
0.5 times the radius, preserving input order; each surface must retain at least
six observations. Each refinement attempt uses two ICP scales, with at most
ten iterations per scale.

The default budget is 32; a budget of 16 uses the corresponding nested queue
prefix. The budget caps attempts, and refinement_count reports actual use.
If no valid refined pose exists, registration raises ValueError.

## Reuse the verifier with another generator

    from ngr import refine_candidates

    pose = refine_candidates(
        p, q, source_points, target_points,
        candidate_rotations, candidate_translations,
        native_pose={"R": native_rotation, "t": native_translation},
        inlier_radius=0.1, budget=32,
    )

Candidate rotations and translations have shapes (K, 3, 3) and (K, 3).
The native pose is the generator's own output, including its native refinement,
and is retained in the primary shortlist. Invalid raw rigid poses are
discarded; the native pose itself must be valid.

An optional auxiliary argument supplies a pair of rotation and translation
arrays for the additional refinement queue. All candidate translations use
the input coordinate unit. The complete NGR interface automatically supplies
direct triple-nucleus fits as auxiliary proposals; pair nuclei supply grown
poses only.

## Correspondences only

    from ngr import register_correspondences

    pose = register_correspondences(p, q, inlier_radius=0.04)

This interface uses the capacity-aware generator and its native residual
readout. The radius scales the compatibility kernel and correspondence refits
together. Seeding requires admissible geometric support; if no nondegenerate
nucleus exists, registration raises ValueError. Exactly three mutually
consistent matches can leave the triple ranking empty because its score
requires additional graph support.

## Implementation

- ngr/_core.py: compatibility, capacity-aware motif selection and phase growth.
- ngr/_rigid.py: weighted rigid fits and correspondence refitting.
- ngr/_surface.py: candidate allocation, ICP, surface evidence and release.
- ngr/registration.py: unit-aware registration and candidate-pool interfaces.

[Method details](docs/method.md) describe the equations, numerical conventions
and coordinate scaling.

## Tests

    pip install ".[surface,test]"
    python -m pytest

Author: Chengrui Xu.
