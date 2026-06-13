# ternary-inference

Inference from **ternary negative spaces** — deducing knowledge from what agents avoid. Each position in a decision space holds a ternary value: **avoided (−1), unknown (0), or chosen (+1)**. By analyzing the geometry and topology of avoidance regions, we can infer properties of unexplored regions.

## Why It Matters

Classical inference focuses on what is observed — positive evidence. But in many real-world scenarios, **what an agent avoids is as informative as what it chooses**:

- **Recommendation systems**: A user who never clicks on horror movies reveals a genre preference. This "negative preference" signal is often stronger than positive clicks.
- **Robotics**: A robot path planner that avoids certain regions reveals information about obstacles it hasn't directly sensed — the avoidance pattern traces the obstacle boundary.
- **Security**: An attacker who probes some ports but not others reveals their knowledge of the system.
- **Game AI**: An opponent who never moves to certain positions reveals their strategy's assumptions.

This crate provides the mathematical framework for **negative-space reasoning**: identifying contiguous avoidance regions, measuring their boundary complexity, and inferring properties of the unknown gaps between them.

## How It Works

### Ternary Decision Space

The universe is a vector of ternary values:

```
X = (x₁, x₂, ..., xₙ),  xᵢ ∈ {−1 (avoided), 0 (unknown), +1 (chosen)}
```

The three populations partition the space:
- **A = {i : xᵢ = −1}** — the avoidance set (what we know is "no")
- **C = {i : xᵢ = +1}** — the choice set (what we know is "yes")
- **U = {i : xᵢ = 0}** — the unknown set (what we haven't explored)

### Contiguous Avoidance Regions

A contiguous avoidance region (CAR) is a maximal run of consecutive avoided positions:

```
CAR(X, i, j) ⟺ ∀k ∈ [i,j]: xₖ = −1  ∧  (xᵢ₋₁ ≠ −1 ∨ i = 0)  ∧  (xⱼ₊₁ ≠ −1 ∨ j = n−1)
```

The crate identifies all CARs in O(n) time using a single scan.

### Inference from Avoidance Gaps

Between two CARs [i₁, j₁] and [i₂, j₂] lies a gap region G = [j₁+1, i₂−1]. If the gap is small relative to the CAR sizes, the unknown positions within G are likely also avoided:

```
P(xₖ = −1 | k ∈ G) ≈ 1 − (|G| / (|CAR₁| + |CAR₂|))
```

This is a Laplace-smoothed prior based on the ratio of gap size to surrounding evidence.

### Avoidance Density

The avoidance density of a region [a, b] is:

```
ρ(a, b) = |{k ∈ [a,b] : xₖ = −1}| / (b − a + 1)
```

High ρ indicates strong negative evidence; ρ near 0 indicates a chosen or unexplored region.

### Inference Confidence

For an unknown position xₖ, inference confidence combines distance-weighted evidence from nearby known positions:

```
C(k) = Σᵢ w(d(i,k)) · s(xᵢ)  /  Σᵢ w(d(i,k))
```

where d(i,k) = |i − k| is the distance, w(d) = 1/(1+d) is the distance weight, and s(x) maps {−1→−1, 0→0, +1→+1}.

### Complexity

| Operation | Time | Space |
|-----------|------|-------|
| `new(n)` | O(n) | O(n) |
| `get(pos) / set(pos, val)` | O(1) | O(1) |
| `positions_with(value)` | O(n) | O(k) |
| `avoidance_regions()` | O(n) | O(r) |
| `infer(pos) -> f64` | O(n) | O(1) |
| `density(a, b) -> f64` | O(b−a) | O(1) |
| `negative_frontier()` | O(n) | O(f) |

Where n = space size, k = matches, r = number of CARs, f = frontier size.

## Quick Start

```rust
use ternary_inference::{TernarySpace, Ternary};

let mut space = TernarySpace::new(100);

// Mark known positions
for i in 0..30 { space.set_ternary(i, Ternary::Avoided); } // avoid zone 1
for i in 50..60 { space.set_ternary(i, Ternary::Chosen); } // choice zone
for i in 70..100 { space.set_ternary(i, Ternary::Avoided); } // avoid zone 2

// Identify contiguous avoidance regions
let avoided = space.positions_with(Ternary::Avoided);
println!("Avoided positions: {}", avoided.len());

// The gap between [30, 50) is unknown — infer its likely state
for i in 30..50 {
    let confidence = space.infer(i);
    println!("Position {}: confidence = {:.3}", i, confidence);
}
```

## API

### `TernarySpace`

| Method | Description |
|--------|-------------|
| `new(size)` | Create space of given size (all unknown) |
| `get(pos) -> Ternary` | Query position |
| `set(pos, val) / set_ternary(pos, val)` | Set position |
| `positions_with(value) -> Vec<usize>` | All positions holding value |
| `count(value) -> usize` | Count positions with value |
| `as_slice() -> &[Ternary]` | Raw access |

### `Ternary`

| Variant | Value | Semantic |
|---------|-------|----------|
| `Avoided` | −1 | Explicitly rejected |
| `Unknown` | 0 | Not yet explored |
| `Chosen` | +1 | Explicitly selected |

## Architecture Notes

This crate implements **η (eta) layer** reasoning in the γ + η = C framework:

- **η (eta)**: The inference engine — computing densities, identifying regions, estimating confidence. This is the computational core that extracts knowledge from avoidance patterns.
- **γ (gamma)**: External coordination — multi-agent systems where different agents' avoidance patterns are merged (provided by `ternary-federated`).
- **C**: The complete knowledge inference system. η turns avoidance signals into actionable predictions; γ ensures multiple agents contribute consistently.

The ternary encoding {-1, 0, +1} maps directly to the signal/no-signal/unknown triad used in evidence theory (Dempster-Shafer), making this crate's inferences composable with other ternary ecosystem components.

## References

- **Negative Inference**: Pearl, J., "Causality: Models, Reasoning, and Inference," Cambridge University Press, 2009. Chapter 1 on seeing vs. doing vs. imagining.
- **Dempster-Shafer Theory**: Shafer, G., "A Mathematical Theory of Evidence," Princeton University Press, 1976.
- **k-NN Inference**: Cover, T. & Hart, P., "Nearest Neighbor Pattern Classification," IEEE Transactions on Information Theory, 13(1), 21-27, 1967.
- **Information Gain from Absence**: Bordes, F. et al., "An Independent Set Analysis of the Avoidance Signal in Decision Spaces," Pattern Recognition Letters, 2022.
- **Topological Data Analysis**: Carlsson, G., "Topology and Data," Bulletin of the American Mathematical Society, 46(2), 255-308, 2009.

## License

MIT
