# Ternary Inference — Deducing Knowledge from Ternary Negative Spaces

**Ternary Inference** deduces knowledge by analyzing what agents *avoid* rather than what they choose. Each position in a decision space holds a ternary value: avoided (-1), unknown (0), or chosen (+1). By mapping contiguous avoidance regions and analyzing gaps between chosen regions, the crate infers properties of unexplored space.

## Why It Matters

Traditional inference focuses on positive evidence — what was chosen, clicked, or selected. But negative evidence (what was avoided) is equally informative and dramatically underutilized. A recommendation system that knows users avoid horror films (-1) and choose comedies (+1) can infer something about their preferences for neutral genres (0) that haven't been tested yet. This crate provides the formal framework: ternary decision spaces where {-1, 0, +1} represent avoidance, uncertainty, and preference, plus algorithms to identify regions, measure contiguity, and make inferences from the pattern of avoidance.

## How It Works

### Ternary Decision Space

A `TernarySpace` is a vector of ternary values. Each position represents a decision point (an item, a feature, a configuration option) with state:

- **Chosen (+1)**: The agent has actively selected this
- **Unknown (0)**: No information — unexplored
- **Avoided (-1)**: The agent has actively rejected this

### Negative Region Detection

Contiguous runs of -1 values form **avoidance regions**. The crate identifies maximal contiguous avoidance segments and measures their size. Large avoidance regions indicate strong negative preferences; isolated -1 values may be noise.

### Gap Analysis

Between avoidance regions lie **gaps** — sequences of unknown (0) or chosen (+1) values. Gap analysis measures:
- Gap length: How many positions between avoidance regions
- Gap composition: Ratio of chosen vs. unknown within the gap
- Boundary sharpness: How quickly the value transitions from -1 to 0/+1

### Inference Rules

From the ternary structure, the crate infers:

1. **Likely preference**: If a gap between two avoidance regions contains mostly +1 values, the entire gap is likely preferred
2. **Likely avoidance**: If an unknown (0) position is surrounded by -1 on both sides, it's likely also avoided
3. **Boundary regions**: Positions at the edge of avoidance regions have uncertain preference and are candidates for exploration

Inference is O(n) for n positions.

## Quick Start

```rust
use ternary_inference::{Ternary, TernarySpace};

let mut space = TernarySpace::new(10);
space.set(0, Ternary::Avoided);
space.set(1, Ternary::Avoided);
space.set(2, Ternary::Unknown);
space.set(3, Ternary::Chosen);
space.set(4, Ternary::Chosen);

// Identify avoidance regions
let regions = space.contiguous_regions(Ternary::Avoided);
println!("{} avoidance regions detected", regions.len());
```

```bash
cargo add ternary-inference
```

## API

| Type / Function | Description |
|---|---|
| `Ternary` | `Avoided(-1)`, `Unknown(0)`, `Chosen(+1)` with Display (✗, ?, ✓) |
| `TernarySpace` | Vector of ternary decisions: `new(size)`, `set()`, `get()`, `contiguous_regions()` |

## Architecture Notes

Negative-space inference is the epistemic layer of **SuperInstance**. Agents record their decisions as ternary spaces; the inference engine identifies what they likely want (γ = growth toward) and likely avoid (η = entropy away from). The γ + η = C conservation manifests spatially: every position is either chosen, avoided, or unknown, and the sum of known positions equals the explored fraction. See [Architecture](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## References

- Pearl, Judea. *Causality*, 2nd ed., Cambridge UP, 2009 — inference from incomplete evidence.
- Bordes, Antoine et al. "Translating Embeddings for Modeling Multi-relational Data," *NeurIPS*, 2013 — negative sampling in knowledge graphs.
- Settles, Burr. "Active Learning Literature Survey," *UW-Madison CS TR-1648*, 2009 — exploring unknown regions.

## License

MIT
