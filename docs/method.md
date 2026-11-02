# Inference method

For correspondences `(p_i, q_i)`, NGR seeks a proper rotation `R` and translation
`t` such that `q_i` is close to `R p_i + t` for a coherent subset.

## Compatibility and nuclei

The off-diagonal graph entries are

$$\Lambda_{ij}=\exp\left[-\frac{(\|p_i-p_j\|-\|q_i-q_j\|)^2}{2\sigma_c^2}\right],
\qquad \Lambda_{ii}=0.$$

The native-unit kernel width is `sigma_c = 0.015`. Pair scores combine direct
compatibility with second-order support:

$$P_{ab}=\Lambda_{ab}(\Lambda^2)_{ab}.$$

Let `lambda_bar` be the mean off-diagonal compatibility. The observable load
`B2 = (N - 2) * lambda_bar**2` determines the motif order. Below one, NGR retains
up to 256 pairs. Otherwise, it ranks up to 4096 pivot pairs and evaluates all
third nodes with the weighted four-node score

$$T_{abc}=\Lambda_{ab}\Lambda_{ac}\Lambda_{bc}
            \sum_d\Lambda_{ad}\Lambda_{bd}\Lambda_{cd}.$$

Source and target triangles must have singular-value ratio at least `0.1`.
Blockwise ranking and deduplication retain at most 1024 triples. Phase weights
start from the product of anchor compatibility rows, with a unit contribution at
each anchor, followed by normalization.

## State-retaining growth

Crystallization-inspired motifs provide the initial composition of each phase.
The total-reflux-inspired update retains that composition as it evolves. Phases
evolve independently through ten sequential updates:

$$w_i^+\propto(w_i+10^{-30})\,[1+N(\Lambda w)_i]^2.$$

The current composition remains the multiplicative base. Compatibility matrices
and growth use single precision; distances, triangle geometry, rigid fitting and
final residual scoring use double precision.

## Rigid fitting and readout

Each phase produces a weighted Kabsch fit. Two subsequent fits use the estimated
model's residual-below-`0.04` correspondence mask. Pair phases use support count
minus a small mean-residual penalty. Triple phases use the extensive score

$$M(R,t)=\sum_i\max(1-\|Rp_i+t-q_i\|/\tau,0).$$

The correspondence interface returns the pose with highest score. For physical
residual radius `tau`, matches are multiplied by `0.04 / tau` before this inference,
and translations are divided by the same scale on output. The physical kernel
width is therefore `0.375 * tau`.

## Budgeted geometric postprocessing

The complete surface-assisted estimator includes a reusable pose postprocessor.
It combines geometric screening, allocation of a fixed candidate-refinement
budget, two-scale point-to-plane ICP, and a guarded geometry-informed readout.
Its inputs are candidate poses, a native reference pose, correspondences, and
observed surfaces. Candidate generation and this verification stage are separate
components.

The surface interface uses the same motif inference at residual radius `tau`.
All surface-backend coordinates are normalized by `0.1 / tau`; the following
backend thresholds are stated in that normalized unit. For metre coordinates,
`tau=0.1` is the indoor radius and `tau=0.6` is the outdoor driving radius.

Surface preparation computes `scaled = points * (0.04 / tau)`, keeps the first
observation in each internal `0.02` voxel in input order, divides by `0.04 / tau`,
then applies the backend scale. The physical voxel side is `0.5 * tau`.

Grown hypotheses form the primary pool. A native-winner-preserving shortlist
retains at most 32 pose modes, and an interleaved stable ranking by `M`, `G` and
`M*G` allocates at most 16 primary refinements. Direct three-point fits remain
auxiliary proposals; two-point nuclei have no direct pose proposal.

The observed source is deterministically divided into fitting and query samples.
ICP uses the source fitting sample and the complete observed target. Surface
support uses held-out source queries, sampled target queries, and truncated
nearest-neighbor distances against the observed surfaces. Uniform
forward and reverse means give harmonic support `G`. For a query point with
centred, RMS-normalized coordinate `x` and normal `n`, the Jacobian is

$$J_i=[(x_i\times n_i)^T,n_i^T],\quad
\ell_i=J_i\left(\sum_jJ_j^TJ_j\right)^\dagger J_i^T.$$

Normalized nonnegative leverage values weight the directional supports; their
harmonic mean gives `I`. Normals use at most 20 neighbors. Initial screening uses
the first 128 query points per direction with renormalized weights.

Pose neighborhoods use `5 degrees` and normalized translation distance `0.1`
(physical `tau`). The nondominated unused modes in `(M, I)` supply at most 16
additional refinements. The complete additional queue is stably allocated before
the global budget is applied. The default B16 and optional B32 cap total
candidate-refinement attempts on nested prefixes, including auxiliary poses at
B16 when fewer than 16 primary attempts are available. Each attempt uses
point-to-plane fitting, normalized Tukey scale `0.1`, and up to two ICP passes at
capture radii `0.2` then `0.1` (physical `2*tau` then `tau`), with at most 10
iterations per radius. Invalid intermediate poses terminate the attempt.

A retained reference pose maximizes `M * sqrt(G)`. A challenger must have
nondecreasing `I` relative to that reference and strictly improve
`M * sqrt(I)`. Ties retain the reference. If every primary refinement is invalid, no winner is
returned. The output is one rigid transformation in the input coordinate unit.
