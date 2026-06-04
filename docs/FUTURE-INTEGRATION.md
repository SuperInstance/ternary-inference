# Future Integration: ternary-inference

## Current State

ternary-inference deduces knowledge from what agents avoid — inference from ternary negative spaces. `TernarySpace` maps positions to `Ternary` values (Avoided/Unknown/Chosen). `AvoidanceMap` finds contiguous negative regions and gaps between them. `NegativeRegion` tracks boundaries and computes distance/shared-boundary analysis. `InferenceRule` provides four deduction types: `Analogy` (shared boundary type implies similar avoidance), `Interpolation` (edge proximity predicts avoidance), `Exclusion` (fully avoided gaps are excluded), and `BoundaryTransition` (avoidance→choice transitions). `DeductionEngine` applies rules with confidence thresholds. `FeedbackValidator` tracks prediction accuracy.

## Integration Opportunities

### construct-core Tiers → Tiered Inference

ternary-inference maps to the construct-core hardware tiers:

- **Layer 0 (ESP32)**: Pre-computed `DeductionEngine` results compiled into lookup tables. No runtime inference — all deductions were made at compile time and stored as `ternary-compiler::CompiledPolicy`. The ESP32 just does `query_lookup("inference:position_7")` → "Avoided".
- **Layer 1 (Pi)**: `DeductionEngine::infer_from_map()` runs on incoming sensor data. `NegativeRegion` detection is O(n) on the ternary space. `InferenceRule::interpolation()` is cheap (just distance calculations). Real-time inference on edge devices.
- **Layer 2 (DGX)**: Full `DeductionEngine` with all rules, `FeedbackValidator` for continuous accuracy tracking, and `structural_similarity()` for cross-agent pattern matching. GPU-accelerated when running on large ternary spaces.

### negative-space-core → Combined Inference Engine

`negative-space-core::InferenceEngine` operates at the population level (across agents), while `ternary-inference::DeductionEngine` operates at the space level (within an agent's decision space). Combined:

1. `negative-space-core` identifies WHICH agents have the most informative avoidance patterns (294:1 ratio conformers)
2. `ternary-inference` identifies WHAT those agents are avoiding (negative regions and gaps)
3. `DeductionEngine::infer()` produces deductions from the combined analysis
4. `FeedbackValidator` validates against actual outcomes

### ternary-cell → Cell-Level Inference

Each `TernaryCell` in a grid has a ternary value (-1/0/+1). The grid IS a `TernarySpace`. `AvoidanceMap::find_negative_regions()` identifies clusters of Suppress cells. `DeductionEngine::infer()` predicts what those avoidance clusters imply about neighboring cells. The `FeedbackValidator` runs against the next tick's actual values, creating a continuous learning loop: predict → tick → validate → refine.

### PLATO Room → Room-Level Gap Analysis

Each PLATO room has a knowledge domain. Rooms that agents avoid (low visit count) are negative spaces in the room-level ternary space. `DeductionEngine` infers WHY agents avoid certain rooms: boundary analysis of neighboring rooms reveals whether avoidance is due to poor ensign quality, wrong hardware tier, or insufficient skill dependencies.

## Potential in Mature Systems

ternary-inference becomes the "reasoning engine" of the ecosystem. Every agent uses it to learn from what it avoids. Every room uses it to understand why agents visit or avoid it. The fleet coordinator uses it to identify underutilized rooms and optimize placement. The `FeedbackValidator` ensures inference quality improves over time. The four inference rules (Analogy, Interpolation, Exclusion, BoundaryTransition) cover the full spectrum from "similar to what we know" to "transitioning from avoidance to choice."

## Cross-Pollination Ideas

- **ternary-attention → InferenceRule::Analogy**: Replace the simple shared-boundary-type check with attention-weighted analogy. Cells that "attend" to each other have higher analogy confidence.
- **ternary-science → Inference validation**: Use `ternary-science`'s 51 tests as the ground truth for `FeedbackValidator`. Run inference on the experiment data and track accuracy.
- **ternary-steganography → Hidden inference**: Embed inference results in the avoidance regions themselves. The inference about a gap IS hidden in the gap.

## Dependencies for Next Steps

1. `TernarySpace` → `CellGrid` bridge (grid cells as inference space)
2. `DeductionEngine` → construct-core tiered execution (Layer 0/1/2)
3. `FeedbackValidator` → PLATO tile store (accuracy as a tile)
4. Combined pipeline with `negative-space-core::InferenceEngine`
5. GPU acceleration for `structural_similarity()` on large spaces
