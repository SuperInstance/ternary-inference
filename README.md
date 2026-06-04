# ternary-inference

**Inference from ternary negative spaces** — if agents avoid X and Y, what does the gap between them imply?

## Concept

Traditional inference works with positive knowledge: "A is true, B is true, therefore C." This crate explores a different approach: **negative space inference** — learning from what agents *avoid* rather than what they choose.

### Ternary Decisions

At each position in a decision space, an agent can:

| Value | Meaning |
|-------|---------|
| `-1` | **Avoided** — the agent actively rejected this |
| `0` | **Unknown** — unexplored territory |
| `+1` | **Chosen** — the agent selected this |

### Negative Space Inference

When multiple agents leave contiguous regions of avoidance, the boundaries and gaps between those regions contain implicit information:

1. **Analogy**: If agents avoid A and avoid B, and A shares structure with C, then C might also be worth avoiding.
2. **Interpolation**: If positions 1–5 are avoided and positions 10–15 are avoided, positions 6–9 might form a bridge region.
3. **Exclusion**: If every position in a region is avoided by at least one agent, the region is fully excluded — anything remaining is noteworthy.

## Architecture

```
TernarySpace      — A grid of {-1, 0, +1} decisions
AvoidanceMap      — Records which positions are avoided/chosen/unknown
NegativeRegion    — Identifies contiguous avoidance zones and their boundaries
InferenceRule     — Rules for deducing from gaps between avoidance regions
DeductionEngine   — Applies inference rules to produce new knowledge
FeedbackValidator — Tests deductions against actual outcomes, tracks accuracy
```

## Usage

```rust
use ternary_inference::{TernarySpace, AvoidanceMap, DeductionEngine, FeedbackValidator};

// Create a decision space with 20 positions
let mut space = TernarySpace::new(20);

// Record avoidances
space.set(2, -1);
space.set(3, -1);
space.set(4, -1);
space.set(10, -1);
space.set(11, -1);
space.set(12, -1);

// Record choices
space.set(0, 1);
space.set(1, 1);
space.set(15, 1);

// Build avoidance map and find negative regions
let map = AvoidanceMap::from_space(&space);
let regions = map.find_negative_regions();

// Run inference
let engine = DeductionEngine::new();
let deductions = engine.infer(&regions);

// Validate against outcomes
let mut validator = FeedbackValidator::new();
for deduction in &deductions {
    validator.record(deduction, space.get(deduction.position));
}
println!("Accuracy: {:.1}%", validator.accuracy() * 100.0);
```

## License

MIT
