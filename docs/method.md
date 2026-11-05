# Inference method

For tentative correspondences (p_i, q_i), NGR estimates a proper rigid motion
(R, t). The generator and the surface verifier are separate components; the
verifier also accepts candidate poses from another generator. Phase weights
are inference states, with no training required.

## Compatibility and motif budgets

The graph is zero-diagonal, with

$$
\Lambda_{ij}=\exp\!\left[
-\frac{(\|p_i-p_j\|-\|q_i-q_j\|)^2}{2\sigma_c^2}
\right], \qquad i\ne j.
$$

In internal generator units, sigma_c = 0.015. Let lambda_bar be the mean
off-diagonal compatibility. The load (N - 2) times lambda_bar squared chooses
256 pair nuclei when below one, or a shortlist of 4096 pivot pairs leading to
at most 1024 triple nuclei otherwise. Requested counts are bounded by the
number of available motifs.

Pair affinity is

$$P_{ab}=\Lambda_{ab}(\Lambda^2)_{ab}.$$

A deeper ranking retains at most 65536 pairs. Half the pair budget keeps the
original affinity ranking. The remainder is selected greedily with a
cumulative vacancy for each correspondence x:

$$
\theta_x=(1+n_x/q)^{-1},\qquad
\widetilde P_{ab}=P_{ab}\theta_a\theta_b,
$$

where n_x counts the selected motifs containing x and q = 8 for pairs.

Triple affinity is

$$
T_{abc}=\Lambda_{ab}\Lambda_{ac}\Lambda_{bc}
        \sum_d\Lambda_{ad}\Lambda_{bd}\Lambda_{cd}.
$$

Both source and target triangles must have singular-value ratio at least 0.1.
Each block of 256 pivots contributes at most 3072 scored triples.
Deduplication and ranking retain up to 16384 candidates. Up to 512 nuclei keep
their affinity order; the remainder of the 1024 budget uses
T_abc times theta_a theta_b theta_c, with q = 4. These vacancies discount
repeated match occupancy without imposing a hard exclusion.

## Retained-composition growth and fitting

Each pair or triple initializes a distribution from the product of its
anchors' compatibility rows plus a unit contribution at each anchor.
Normalization gives a phase of total weight one. Phases evolve independently
for ten stages:

$$
w_i^+\propto (w_i+10^{-30})[1+N(\Lambda w)_i]^2.
$$

The current composition remains the multiplicative base. Compatibility and
growth use single precision; distances, triangle geometry, rigid fitting and
residual readout use double precision.

Each phase gives a weighted Kabsch fit followed by two hard refits on matches
below internal residual radius 0.04. The pair native readout is the support
count minus 0.001 times the mean residual. The triple native readout is

$$M(R,t)=\sum_i[1-\|Rp_i+t-q_i\|/0.04]_+.$$

For a physical radius tau, generator inputs are multiplied by 0.04 / tau and
translations divided by that scale on output. Thus the physical kernel width
is 0.375 times tau. The correspondence-only interface returns the native
winner. The complete interface also supplies direct triple-nucleus fits to
the surface verifier; a pair alone does not define a unique 3D pose.

## Bounded surface finishing

All backend coordinates are normalized by 0.1 / tau. Distances below are in
these internal surface units. Preparation retains the first observed point
per physical voxel of side 0.5 times tau, in input order.

A native-winner-preserving shortlist holds at most 32 distinct pose modes,
using 5 degrees and translation distance 0.1. Interleaved stable rankings by
correspondence support M, surface support G and their product allocate at
most 16 primary refinements. Distinct unused poses on the nondominated
(M, I) frontier provide up to 16 more. Additional candidates are ranked before
applying the overall budget, so budgets 16 and 32 have nested queues.

Each attempt uses point-to-plane ICP at capture radii 0.2 and 0.1, a Tukey
kernel of scale 0.1, and at most ten iterations per scale. All valid refined
poses remain eligible for release, including auxiliary poses when a primary
refinement is invalid.

## Geometric evidence

The observed source is deterministically divided into fitting and query
points. ICP uses the fitting points and the complete observed target.
Surface evidence uses the disjoint source queries and sampled target queries,
with at most 4096 queries in either direction.

Truncated correspondence support uses radius 0.1. The harmonic mean of
forward and reverse truncated nearest-neighbor support gives G. To obtain
information support I, the directional means use normalized leverage weights:

$$
J_i=[(\widetilde x_i\times n_i)^T,n_i^T],\qquad
\ell_i=J_i\left(\sum_jJ_j^TJ_j\right)^\dagger J_i^T.
$$

Coordinates for these rows are centered and RMS-normalized. Normals use at
most 20 neighbors. Initial information screening uses 128 queries per
direction with renormalized weights.

## Complementary release

For every valid refined candidate k, correspondence and information support
give the normalized channel

$$s_k=\frac{M_k\sqrt{I_k}}{\max_j M_j\sqrt{I_j}}.$$

In each nearest-neighbor direction, retain distances d below 0.1 and set
u = d squared / 0.1 squared. Its mean is used in the one-sided concentration
score

$$
\psi(\bar u)=\sup_{\eta\ge0}
\left[-\eta\bar u-\log\frac{1-e^{-\eta}}{\eta}\right].
$$

This is the rate function below the mean of an idealized uniform distribution
on [0, 1]. It is a residual-concentration score, not a calibrated probability
of pose correctness. The score is zero for a mean at least one half.
A monotone bisection solves the positive optimum.

Average the two directional scores into J_k and normalize the shape channel
as h_k = exp(J_k - max_j J_j). The released pose maximizes

$$\min(s_k,h_k).$$

Ties prefer larger support, then larger shape score, then the earlier queue
position. The final decision introduces no additional ICP calls.

A candidate with no accepted queries in either direction has no shape
evidence. If none of the candidates has shape evidence, release uses maximum
support. If all support scores vanish, it uses the shape channel. Infinite
concentration scores, produced by exact zero residuals, share the maximal
shape score. An empty valid refined pool raises ValueError.

## Interfaces and output units

register_correspondences returns R and t from the capacity-aware generator.
register_with_surfaces runs generation, finishing and release. refine_candidates
runs finishing and release on externally supplied poses; the native pose and
all candidate translations must use the same unit as the observed points.

The two surface interfaces return R, t and refinement_count. R maps source to
target as a column-vector rotation, so NumPy row arrays align as
p @ R.T + t. Refinement count measures candidate attempts, not the number of
internal ICP iterations. Coordinate arrays are not modified.
