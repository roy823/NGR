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

Ten independent updates enrich each phase:

$$w_i^+\propto(w_i+10^{-30})\,[1+N(\Lambda w)_i]^2.$$

The current composition remains the multiplicative base. Compatibility matrices
and growth use single precision; distances, triangle geometry, rigid fitting and
final residual scoring use double precision.

## Rigid fitting and readout

Each phase produces a weighted Kabsch fit. Two subsequent fits use the estimated
model's residual-below-`0.04` correspondence mask. Pair phases use support count
with a mean-residual tie term. Triple phases use the extensive score

$$M(R,t)=\sum_i\max(1-\|Rp_i+t-q_i\|/\tau,0).$$

The correspondence interface returns the pose with highest score.

## Observed-surface verification

The indoor interface scales metric matches by `0.04 / 0.1` for motif inference.
It retains a native-winner-preserving shortlist of at most 32 pose modes and
allocates at most 16 initial surface refinements.

Each observed surface is deterministically divided into fitting and query
samples. Surface support uses truncated nearest-neighbor distances. Uniform
forward and reverse means give harmonic support `G`. For a query point with
centred, RMS-normalized coordinate `x` and normal `n`, the Jacobian is

$$J_i=[(x_i\times n_i)^T,n_i^T],\quad
\ell_i=J_i\left(\sum_jJ_j^TJ_j\right)^\dagger J_i^T.$$

Normalized nonnegative leverage values weight the directional supports; their
harmonic mean gives `I`. Normals use at most 20 neighbors. Initial screening uses
the first 128 query points per direction with renormalized weights.

Direct fits of the same nuclei supplement the grown hypotheses. Pose
neighborhoods use `5 degrees` and `0.1 metres`. The nondominated unused modes in
`(M, I)` supply at most 16 additional refinements. Total refinement is at most 32.
Every ICP call uses point-to-plane fitting, Tukey scale `0.1 metres`, capture
radii `0.2` then `0.1 metres`, and at most 10 iterations at each radius.

A retained reference pose maximizes `M * sqrt(G)`. A challenger must have
nondecreasing `I` relative to that reference and strictly improve
`M * sqrt(I)`. Ties retain the reference. The output is one rigid transformation.
