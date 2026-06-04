//! # ternary-inference
//!
//! Inference from ternary negative spaces — deducing knowledge from what agents avoid.
//!
//! Each position in a decision space holds a ternary value: avoided (-1), unknown (0),
//! or chosen (+1). By identifying contiguous avoidance regions and analyzing the gaps
//! between them, we can infer properties of unexplored regions.

use std::fmt;

/// A ternary value: avoided, unknown, or chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ternary {
    Avoided = -1,
    Unknown = 0,
    Chosen = 1,
}

impl Ternary {
    /// Create from an i8 value. Clamps to valid range.
    pub fn from_i8(v: i8) -> Self {
        match v {
            -1 => Ternary::Avoided,
            0 => Ternary::Unknown,
            1 => Ternary::Chosen,
            _ if v < -1 => Ternary::Avoided,
            _ => Ternary::Chosen,
        }
    }

    /// Convert to i8.
    pub fn as_i8(self) -> i8 {
        self as i8
    }

    /// Is this an avoidance?
    pub fn is_avoided(self) -> bool {
        self == Ternary::Avoided
    }

    /// Is this unknown?
    pub fn is_unknown(self) -> bool {
        self == Ternary::Unknown
    }

    /// Is this a choice?
    pub fn is_chosen(self) -> bool {
        self == Ternary::Chosen
    }
}

impl fmt::Display for Ternary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ternary::Avoided => write!(f, "✗"),
            Ternary::Unknown => write!(f, "?"),
            Ternary::Chosen => write!(f, "✓"),
        }
    }
}

// ── TernarySpace ──────────────────────────────────────────────────────────────

/// A space of ternary decisions, each position has {-1, 0, +1}.
#[derive(Clone, Debug)]
pub struct TernarySpace {
    values: Vec<Ternary>,
}

impl TernarySpace {
    /// Create a new space of the given size, all positions unknown.
    pub fn new(size: usize) -> Self {
        TernarySpace {
            values: vec![Ternary::Unknown; size],
        }
    }

    /// Get the size of the space.
    pub fn size(&self) -> usize {
        self.values.len()
    }

    /// Get the value at a position.
    pub fn get(&self, pos: usize) -> Ternary {
        self.values.get(pos).copied().unwrap_or(Ternary::Unknown)
    }

    /// Set the value at a position from an i8.
    pub fn set(&mut self, pos: usize, value: i8) {
        if pos < self.values.len() {
            self.values[pos] = Ternary::from_i8(value);
        }
    }

    /// Set the value at a position from a Ternary.
    pub fn set_ternary(&mut self, pos: usize, value: Ternary) {
        if pos < self.values.len() {
            self.values[pos] = value;
        }
    }

    /// Get all positions with a given value.
    pub fn positions_with(&self, value: Ternary) -> Vec<usize> {
        self.values
            .iter()
            .enumerate()
            .filter(|(_, &v)| v == value)
            .map(|(i, _)| i)
            .collect()
    }

    /// Count positions with a given value.
    pub fn count(&self, value: Ternary) -> usize {
        self.values.iter().filter(|&&v| v == value).count()
    }

    /// Get a slice of the underlying values.
    pub fn as_slice(&self) -> &[Ternary] {
        &self.values
    }

    /// Check if two regions of the space are structurally similar.
    pub fn structural_similarity(&self, start_a: usize, end_a: usize, start_b: usize, end_b: usize) -> f64 {
        let len_a = end_a.saturating_sub(start_a);
        let len_b = end_b.saturating_sub(start_b);
        if len_a == 0 || len_b == 0 {
            return 0.0;
        }
        let min_len = len_a.min(len_b);
        let mut matches = 0usize;
        for i in 0..min_len {
            if self.get(start_a + i) == self.get(start_b + i) {
                matches += 1;
            }
        }
        matches as f64 / min_len as f64
    }
}

impl fmt::Display for TernarySpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for v in &self.values {
            write!(f, "{}", v)?;
        }
        Ok(())
    }
}

// ── AvoidanceMap ──────────────────────────────────────────────────────────────

/// Records which positions are avoided (-1), unknown (0), or chosen (+1).
/// Provides methods to find contiguous negative (avoidance) regions.
#[derive(Clone, Debug)]
pub struct AvoidanceMap {
    space: TernarySpace,
}

impl AvoidanceMap {
    /// Create from a TernarySpace.
    pub fn from_space(space: &TernarySpace) -> Self {
        AvoidanceMap { space: space.clone() }
    }

    /// Create a new empty map of the given size.
    pub fn new(size: usize) -> Self {
        AvoidanceMap { space: TernarySpace::new(size) }
    }

    /// Get the underlying space.
    pub fn space(&self) -> &TernarySpace {
        &self.space
    }

    /// Set a position.
    pub fn set(&mut self, pos: usize, value: i8) {
        self.space.set(pos, value);
    }

    /// Find all contiguous negative (avoidance) regions.
    pub fn find_negative_regions(&self) -> Vec<NegativeRegion> {
        let mut regions = Vec::new();
        let mut i = 0;
        while i < self.space.size() {
            if self.space.get(i).is_avoided() {
                let start = i;
                while i < self.space.size() && self.space.get(i).is_avoided() {
                    i += 1;
                }
                let end = i; // exclusive
                regions.push(NegativeRegion::new(start, end, &self.space));
            } else {
                i += 1;
            }
        }
        regions
    }

    /// Find all gaps (contiguous unknown regions) between avoidance regions.
    pub fn find_gaps(&self) -> Vec<(usize, usize)> {
        let mut gaps = Vec::new();
        let mut i = 0;
        while i < self.space.size() {
            if self.space.get(i).is_unknown() {
                let start = i;
                while i < self.space.size() && self.space.get(i).is_unknown() {
                    i += 1;
                }
                gaps.push((start, i));
            } else {
                i += 1;
            }
        }
        gaps
    }

    /// Find gaps that lie between two avoidance regions.
    pub fn find_gaps_between_regions(&self, regions: &[NegativeRegion]) -> Vec<Gap> {
        if regions.len() < 2 {
            return Vec::new();
        }
        let mut gaps = Vec::new();
        for i in 0..regions.len() - 1 {
            let gap_start = regions[i].end;
            let gap_end = regions[i + 1].start;
            if gap_start < gap_end {
                gaps.push(Gap {
                    start: gap_start,
                    end: gap_end,
                    left_region: i,
                    right_region: i + 1,
                });
            }
        }
        gaps
    }
}

// ── NegativeRegion ────────────────────────────────────────────────────────────

/// A contiguous region of avoidances with computed boundaries.
#[derive(Clone, Debug)]
pub struct NegativeRegion {
    /// Start position (inclusive).
    pub start: usize,
    /// End position (exclusive).
    pub end: usize,
    /// The boundary values just outside this region.
    pub left_boundary: Option<Ternary>,
    pub right_boundary: Option<Ternary>,
    /// Size of the region.
    pub size: usize,
}

impl NegativeRegion {
    /// Create a new negative region from a span in a space.
    pub fn new(start: usize, end: usize, space: &TernarySpace) -> Self {
        let left_boundary = if start > 0 {
            Some(space.get(start - 1))
        } else {
            None
        };
        let right_boundary = if end < space.size() {
            Some(space.get(end))
        } else {
            None
        };
        NegativeRegion {
            start,
            end,
            left_boundary,
            right_boundary,
            size: end - start,
        }
    }

    /// Does this region share a boundary type with another?
    pub fn shares_boundary_type(&self, other: &NegativeRegion) -> bool {
        match (self.right_boundary, other.left_boundary) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        }
    }

    /// Distance to another region (gap between them).
    pub fn distance_to(&self, other: &NegativeRegion) -> usize {
        if self.end <= other.start {
            other.start - self.end
        } else if other.end <= self.start {
            self.start - other.end
        } else {
            0 // overlapping
        }
    }

    /// Check if position is within this region.
    pub fn contains(&self, pos: usize) -> bool {
        pos >= self.start && pos < self.end
    }
}

// ── Gap ───────────────────────────────────────────────────────────────────────

/// A gap between two avoidance regions.
#[derive(Clone, Debug)]
pub struct Gap {
    pub start: usize,
    pub end: usize,
    pub left_region: usize,
    pub right_region: usize,
}

impl Gap {
    /// Size of the gap.
    pub fn size(&self) -> usize {
        self.end - self.start
    }

    /// Is the gap empty?
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// Positions in the gap.
    pub fn positions(&self) -> Vec<usize> {
        (self.start..self.end).collect()
    }
}

// ── InferenceRule ─────────────────────────────────────────────────────────────

/// The type of inference applied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InferenceType {
    /// If A avoids X and B avoids Y, and X~Y structurally, the gap implies similar avoidance.
    Analogy,
    /// If avoidance at both ends, the gap likely bridges — positions near the gap center are unknown.
    Interpolation,
    /// Everything in the gap is excluded — no agent chose it.
    Exclusion,
    /// The gap boundaries suggest a transition from avoidance to choice.
    BoundaryTransition,
}

/// A single inference/deduction about a position.
#[derive(Clone, Debug)]
pub struct Inference {
    /// Position this inference is about.
    pub position: usize,
    /// Predicted value.
    pub predicted: Ternary,
    /// Type of inference used.
    pub inference_type: InferenceType,
    /// Confidence (0.0 to 1.0).
    pub confidence: f64,
    /// Human-readable reason.
    pub reason: String,
}

/// Inference rules for deducing from gaps.
pub struct InferenceRule;

impl InferenceRule {
    /// Analogy: if regions share boundary types, infer similar avoidance in the gap.
    pub fn analogy(
        regions: &[NegativeRegion],
        gap: &Gap,
    ) -> Option<Inference> {
        let left = &regions[gap.left_region];
        let right = &regions[gap.right_region];
        if left.shares_boundary_type(right) && gap.size() <= left.size + right.size {
            let mid = gap.start + gap.size() / 2;
            let confidence = 0.6 + 0.2 * (1.0 / (gap.size() as f64 + 1.0));
            Some(Inference {
                position: mid,
                predicted: Ternary::Avoided,
                inference_type: InferenceType::Analogy,
                confidence: confidence.min(0.95),
                reason: format!(
                    "Regions [{},{}) and [{},{}) share boundary type; gap center {} likely avoided",
                    left.start, left.end, right.start, right.end, mid
                ),
            })
        } else {
            None
        }
    }

    /// Interpolation: positions near avoidance boundaries may follow the boundary's value.
    pub fn interpolation(
        regions: &[NegativeRegion],
        gap: &Gap,
    ) -> Vec<Inference> {
        let mut inferences = Vec::new();
        if gap.size() == 0 {
            return inferences;
        }

        let gap_size = gap.size();

        // Edge positions: close to avoidance, likely avoidance too
        if gap_size >= 2 {
            inferences.push(Inference {
                position: gap.start,
                predicted: Ternary::Avoided,
                inference_type: InferenceType::Interpolation,
                confidence: 0.7,
                reason: format!("Position {} adjacent to left avoidance region", gap.start),
            });
            inferences.push(Inference {
                position: gap.end - 1,
                predicted: Ternary::Avoided,
                inference_type: InferenceType::Interpolation,
                confidence: 0.7,
                reason: format!("Position {} adjacent to right avoidance region", gap.end - 1),
            });
        }

        // Center positions: further from avoidance, lower confidence
        if gap_size >= 4 {
            for pos in gap.start + 1..gap.end - 1 {
                let dist_from_edge = (pos as i64 - gap.start as i64)
                    .min(gap.end as i64 - 1 - pos as i64) as usize;
                let confidence = 0.3 + 0.1 * dist_from_edge as f64;
                inferences.push(Inference {
                    position: pos,
                    predicted: Ternary::Unknown,
                    inference_type: InferenceType::Interpolation,
                    confidence: confidence.min(0.6),
                    reason: format!("Position {} in gap interior, distance {} from nearest avoidance", pos, dist_from_edge),
                });
            }
        }

        inferences
    }

    /// Exclusion: if no agent chose anything in the gap, it's fully excluded.
    pub fn exclusion(space: &TernarySpace, gap: &Gap) -> Option<Inference> {
        let any_chosen = (gap.start..gap.end).any(|p| space.get(p).is_chosen());
        if !any_chosen && gap.size() > 0 {
            let all_avoided = (gap.start..gap.end).all(|p| space.get(p).is_avoided());
            if all_avoided {
                Some(Inference {
                    position: gap.start + gap.size() / 2,
                    predicted: Ternary::Avoided,
                    inference_type: InferenceType::Exclusion,
                    confidence: 0.95,
                    reason: format!(
                        "Gap [{},{}) fully excluded — all positions avoided or unknown",
                        gap.start, gap.end
                    ),
                })
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Boundary transition: detect transitions at gap edges.
    pub fn boundary_transition(
        regions: &[NegativeRegion],
        gap: &Gap,
    ) -> Vec<Inference> {
        let mut inferences = Vec::new();
        let left = &regions[gap.left_region];

        if let Some(lb) = left.right_boundary {
            if lb.is_chosen() {
                // Transition from avoidance to choice
                inferences.push(Inference {
                    position: gap.start,
                    predicted: Ternary::Chosen,
                    inference_type: InferenceType::BoundaryTransition,
                    confidence: 0.8,
                    reason: format!(
                        "Position {} at boundary transition (avoidance→choice)",
                        gap.start
                    ),
                });
            }
        }

        if gap.size() > 0 {
            let right = &regions[gap.right_region];
            if let Some(rb) = right.left_boundary {
                if rb.is_chosen() {
                    inferences.push(Inference {
                        position: gap.end - 1,
                        predicted: Ternary::Chosen,
                        inference_type: InferenceType::BoundaryTransition,
                        confidence: 0.8,
                        reason: format!(
                            "Position {} at boundary transition (choice→avoidance)",
                            gap.end - 1
                        ),
                    });
                }
            }
        }

        inferences
    }
}

// ── DeductionEngine ───────────────────────────────────────────────────────────

/// Applies inference rules to produce deductions about unexplored regions.
pub struct DeductionEngine {
    /// Minimum confidence threshold for deductions.
    pub min_confidence: f64,
}

impl DeductionEngine {
    /// Create a new engine with default settings.
    pub fn new() -> Self {
        DeductionEngine { min_confidence: 0.3 }
    }

    /// Create with a custom confidence threshold.
    pub fn with_confidence(min_confidence: f64) -> Self {
        DeductionEngine { min_confidence }
    }

    /// Run all inference rules on negative regions within an avoidance map.
    pub fn infer_from_map(&self, map: &AvoidanceMap) -> Vec<Inference> {
        let regions = map.find_negative_regions();
        self.infer(&regions, map.space())
    }

    /// Run all inference rules on pre-computed regions.
    pub fn infer(&self, regions: &[NegativeRegion], space: &TernarySpace) -> Vec<Inference> {
        let mut deductions = Vec::new();

        if regions.len() < 2 {
            // Need at least two regions for gap-based inference
            // But we can still do single-region boundary analysis
            if regions.len() == 1 {
                let r = &regions[0];
                if let Some(lb) = r.left_boundary {
                    if lb.is_unknown() && r.start > 0 {
                        deductions.push(Inference {
                            position: r.start - 1,
                            predicted: Ternary::Avoided,
                            inference_type: InferenceType::BoundaryTransition,
                            confidence: 0.5,
                            reason: format!("Position {} just before solitary avoidance region", r.start - 1),
                        });
                    }
                }
            }
            return deductions;
        }

        // Build a temporary AvoidanceMap to find gaps
        let map = AvoidanceMap::from_space(space);
        let gaps = map.find_gaps_between_regions(regions);

        for gap in &gaps {
            // Analogy
            if let Some(inf) = InferenceRule::analogy(regions, gap) {
                if inf.confidence >= self.min_confidence {
                    deductions.push(inf);
                }
            }

            // Interpolation
            for inf in InferenceRule::interpolation(regions, gap) {
                if inf.confidence >= self.min_confidence {
                    deductions.push(inf);
                }
            }

            // Exclusion
            if let Some(inf) = InferenceRule::exclusion(space, gap) {
                if inf.confidence >= self.min_confidence {
                    deductions.push(inf);
                }
            }

            // Boundary transition
            for inf in InferenceRule::boundary_transition(regions, gap) {
                if inf.confidence >= self.min_confidence {
                    deductions.push(inf);
                }
            }
        }

        // Deduplicate by position (keep highest confidence)
        deductions.sort_by(|a, b| {
            a.position
                .cmp(&b.position)
                .then(b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal))
        });
        deductions.dedup_by(|a, b| a.position == b.position);

        deductions
    }
}

impl Default for DeductionEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ── FeedbackValidator ─────────────────────────────────────────────────────────

/// Tests deductions against actual outcomes and tracks accuracy.
#[derive(Clone, Debug)]
pub struct FeedbackValidator {
    /// Total deductions validated.
    total: usize,
    /// Correct predictions.
    correct: usize,
    /// Detailed records: (position, predicted, actual, correct).
    records: Vec<(usize, Ternary, Ternary, bool)>,
}

impl FeedbackValidator {
    /// Create a new validator.
    pub fn new() -> Self {
        FeedbackValidator {
            total: 0,
            correct: 0,
            records: Vec::new(),
        }
    }

    /// Record a deduction result against the actual value.
    pub fn record(&mut self, inference: &Inference, actual: Ternary) {
        let is_correct = inference.predicted == actual;
        self.total += 1;
        if is_correct {
            self.correct += 1;
        }
        self.records.push((inference.position, inference.predicted, actual, is_correct));
    }

    /// Record a prediction directly.
    pub fn record_prediction(&mut self, position: usize, predicted: Ternary, actual: Ternary) {
        let is_correct = predicted == actual;
        self.total += 1;
        if is_correct {
            self.correct += 1;
        }
        self.records.push((position, predicted, actual, is_correct));
    }

    /// Get accuracy (0.0 to 1.0). Returns 0.0 if no records.
    pub fn accuracy(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.correct as f64 / self.total as f64
        }
    }

    /// Total number of validations.
    pub fn total(&self) -> usize {
        self.total
    }

    /// Number of correct predictions.
    pub fn correct(&self) -> usize {
        self.correct
    }

    /// Get all records.
    pub fn records(&self) -> &[(usize, Ternary, Ternary, bool)] {
        &self.records
    }

    /// Accuracy for a specific inference type — not stored per-type, so this
    /// returns overall accuracy. Provided for API symmetry.
    pub fn accuracy_for_positions(&self, positions: &[usize]) -> f64 {
        let pos_set: Vec<usize> = positions.to_vec();
        let filtered: Vec<_> = self
            .records
            .iter()
            .filter(|(p, _, _, _)| pos_set.contains(p))
            .collect();
        if filtered.is_empty() {
            return 0.0;
        }
        let correct = filtered.iter().filter(|(_, _, _, ok)| *ok).count();
        correct as f64 / filtered.len() as f64
    }

    /// Reset the validator.
    pub fn reset(&mut self) {
        self.total = 0;
        self.correct = 0;
        self.records.clear();
    }
}

impl Default for FeedbackValidator {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Ternary from_i8
    #[test]
    fn test_ternary_from_i8() {
        assert_eq!(Ternary::from_i8(-1), Ternary::Avoided);
        assert_eq!(Ternary::from_i8(0), Ternary::Unknown);
        assert_eq!(Ternary::from_i8(1), Ternary::Chosen);
        assert_eq!(Ternary::from_i8(-5), Ternary::Avoided);
        assert_eq!(Ternary::from_i8(5), Ternary::Chosen);
    }

    // 2. Ternary predicates
    #[test]
    fn test_ternary_predicates() {
        assert!(Ternary::Avoided.is_avoided());
        assert!(!Ternary::Avoided.is_unknown());
        assert!(Ternary::Unknown.is_unknown());
        assert!(Ternary::Chosen.is_chosen());
    }

    // 3. Ternary display
    #[test]
    fn test_ternary_display() {
        assert_eq!(format!("{}", Ternary::Avoided), "✗");
        assert_eq!(format!("{}", Ternary::Unknown), "?");
        assert_eq!(format!("{}", Ternary::Chosen), "✓");
    }

    // 4. TernarySpace creation and defaults
    #[test]
    fn test_space_new() {
        let s = TernarySpace::new(10);
        assert_eq!(s.size(), 10);
        assert_eq!(s.count(Ternary::Unknown), 10);
        assert_eq!(s.count(Ternary::Avoided), 0);
    }

    // 5. TernarySpace set and get
    #[test]
    fn test_space_set_get() {
        let mut s = TernarySpace::new(5);
        s.set(0, -1);
        s.set(2, 1);
        s.set(4, 0);
        assert_eq!(s.get(0), Ternary::Avoided);
        assert_eq!(s.get(1), Ternary::Unknown);
        assert_eq!(s.get(2), Ternary::Chosen);
        assert_eq!(s.get(3), Ternary::Unknown);
        assert_eq!(s.get(4), Ternary::Unknown);
    }

    // 6. TernarySpace out-of-bounds
    #[test]
    fn test_space_out_of_bounds() {
        let mut s = TernarySpace::new(3);
        s.set(10, 1);
        assert_eq!(s.get(10), Ternary::Unknown);
    }

    // 7. TernarySpace positions_with
    #[test]
    fn test_positions_with() {
        let mut s = TernarySpace::new(6);
        s.set(1, -1);
        s.set(2, -1);
        s.set(4, 1);
        assert_eq!(s.positions_with(Ternary::Avoided), vec![1, 2]);
        assert_eq!(s.positions_with(Ternary::Chosen), vec![4]);
    }

    // 8. AvoidanceMap finds negative regions
    #[test]
    fn test_find_negative_regions() {
        let mut s = TernarySpace::new(10);
        s.set(1, -1);
        s.set(2, -1);
        s.set(5, -1);
        s.set(6, -1);
        s.set(7, -1);
        let map = AvoidanceMap::from_space(&s);
        let regions = map.find_negative_regions();
        assert_eq!(regions.len(), 2);
        assert_eq!(regions[0].start, 1);
        assert_eq!(regions[0].end, 3);
        assert_eq!(regions[1].start, 5);
        assert_eq!(regions[1].end, 8);
    }

    // 9. NegativeRegion boundaries
    #[test]
    fn test_negative_region_boundaries() {
        let mut s = TernarySpace::new(10);
        s.set(0, 1); // left boundary of region
        s.set(1, -1);
        s.set(2, -1);
        s.set(3, 1); // right boundary
        let map = AvoidanceMap::from_space(&s);
        let regions = map.find_negative_regions();
        assert_eq!(regions.len(), 1);
        let r = &regions[0];
        assert_eq!(r.left_boundary, Some(Ternary::Chosen));
        assert_eq!(r.right_boundary, Some(Ternary::Chosen));
        assert_eq!(r.size, 2);
    }

    // 10. NegativeRegion at edges
    #[test]
    fn test_region_at_edge() {
        let mut s = TernarySpace::new(5);
        s.set(0, -1);
        s.set(4, -1);
        let map = AvoidanceMap::from_space(&s);
        let regions = map.find_negative_regions();
        assert_eq!(regions.len(), 2);
        assert_eq!(regions[0].left_boundary, None); // position -1 doesn't exist
        assert_eq!(regions[0].right_boundary, Some(Ternary::Unknown));
        assert_eq!(regions[1].left_boundary, Some(Ternary::Unknown));
        assert_eq!(regions[1].right_boundary, None); // position 5 doesn't exist
    }

    // 11. NegativeRegion distance
    #[test]
    fn test_region_distance() {
        let mut s = TernarySpace::new(10);
        s.set(0, -1);
        s.set(1, -1);
        s.set(5, -1);
        s.set(6, -1);
        let map = AvoidanceMap::from_space(&s);
        let regions = map.find_negative_regions();
        assert_eq!(regions[0].distance_to(&regions[1]), 3);
    }

    // 12. NegativeRegion contains
    #[test]
    fn test_region_contains() {
        let mut s = TernarySpace::new(5);
        s.set(1, -1);
        s.set(2, -1);
        let map = AvoidanceMap::from_space(&s);
        let regions = map.find_negative_regions();
        assert!(regions[0].contains(1));
        assert!(regions[0].contains(2));
        assert!(!regions[0].contains(0));
        assert!(!regions[0].contains(3));
    }

    // 13. AvoidanceMap find gaps
    #[test]
    fn test_find_gaps() {
        let mut s = TernarySpace::new(6);
        s.set(0, -1);
        s.set(1, 1);
        s.set(2, 0);
        s.set(3, 0);
        s.set(4, 0);
        s.set(5, -1);
        let map = AvoidanceMap::from_space(&s);
        let gaps = map.find_gaps();
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0], (2, 5));
    }

    // 14. AvoidanceMap find gaps between regions
    #[test]
    fn test_gaps_between_regions() {
        let mut s = TernarySpace::new(10);
        s.set(0, -1);
        s.set(1, -1);
        // 2,3,4 are unknown
        s.set(5, -1);
        s.set(6, -1);
        let map = AvoidanceMap::from_space(&s);
        let regions = map.find_negative_regions();
        let gaps = map.find_gaps_between_regions(&regions);
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].start, 2);
        assert_eq!(gaps[0].end, 5);
        assert_eq!(gaps[0].size(), 3);
    }

    // 15. Gap positions
    #[test]
    fn test_gap_positions() {
        let gap = Gap { start: 3, end: 7, left_region: 0, right_region: 1 };
        assert_eq!(gap.positions(), vec![3, 4, 5, 6]);
        assert_eq!(gap.size(), 4);
        assert!(!gap.is_empty());
    }

    // 16. Shares boundary type
    #[test]
    fn test_shares_boundary_type() {
        let mut s = TernarySpace::new(15);
        s.set(0, 1); // chosen before first region
        s.set(1, -1);
        s.set(2, -1);
        s.set(3, 1); // right boundary of first region = chosen
        s.set(4, 1); // left boundary of second region = chosen
        s.set(5, -1);
        s.set(6, -1);
        s.set(7, 1); // chosen after second region
        let map = AvoidanceMap::from_space(&s);
        let regions = map.find_negative_regions();
        assert_eq!(regions.len(), 2);
        // Region 0: [1,3), right_boundary = space[3] = Chosen
        // Region 1: [5,7), left_boundary = space[4] = Chosen
        assert!(regions[0].shares_boundary_type(&regions[1]));
    }

    // 17. DeductionEngine with two regions
    #[test]
    fn test_engine_infer() {
        let mut s = TernarySpace::new(15);
        s.set(0, -1);
        s.set(1, -1);
        s.set(2, -1);
        // 3-7 unknown
        s.set(8, -1);
        s.set(9, -1);
        s.set(10, -1);
        let map = AvoidanceMap::from_space(&s);
        let engine = DeductionEngine::new();
        let deductions = engine.infer_from_map(&map);
        assert!(!deductions.is_empty(), "Should produce some deductions");
        // All deductions should have positions within the space
        for d in &deductions {
            assert!(d.position < s.size());
        }
    }

    // 18. DeductionEngine with single region
    #[test]
    fn test_engine_single_region() {
        let mut s = TernarySpace::new(10);
        s.set(0, 0);
        s.set(1, -1);
        s.set(2, -1);
        let map = AvoidanceMap::from_space(&s);
        let engine = DeductionEngine::new();
        let deductions = engine.infer_from_map(&map);
        // Single region: should produce a boundary deduction
        assert_eq!(deductions.len(), 1);
        assert_eq!(deductions[0].position, 0);
    }

    // 19. DeductionEngine with no regions
    #[test]
    fn test_engine_no_regions() {
        let s = TernarySpace::new(10);
        let map = AvoidanceMap::from_space(&s);
        let engine = DeductionEngine::new();
        let deductions = engine.infer_from_map(&map);
        assert!(deductions.is_empty());
    }

    // 20. FeedbackValidator accuracy
    #[test]
    fn test_validator_accuracy() {
        let mut v = FeedbackValidator::new();
        let inf = Inference {
            position: 0,
            predicted: Ternary::Avoided,
            inference_type: InferenceType::Analogy,
            confidence: 0.8,
            reason: "test".into(),
        };
        v.record(&inf, Ternary::Avoided); // correct
        v.record(&inf, Ternary::Chosen);  // wrong
        v.record(&inf, Ternary::Avoided); // correct
        assert_eq!(v.total(), 3);
        assert_eq!(v.correct(), 2);
        assert!((v.accuracy() - 2.0 / 3.0).abs() < 1e-9);
    }

    // 21. FeedbackValidator empty
    #[test]
    fn test_validator_empty() {
        let v = FeedbackValidator::new();
        assert_eq!(v.accuracy(), 0.0);
        assert_eq!(v.total(), 0);
    }

    // 22. FeedbackValidator reset
    #[test]
    fn test_validator_reset() {
        let mut v = FeedbackValidator::new();
        v.record_prediction(0, Ternary::Avoided, Ternary::Avoided);
        assert_eq!(v.total(), 1);
        v.reset();
        assert_eq!(v.total(), 0);
        assert_eq!(v.accuracy(), 0.0);
    }

    // 23. FeedbackValidator record_prediction
    #[test]
    fn test_record_prediction() {
        let mut v = FeedbackValidator::new();
        v.record_prediction(0, Ternary::Chosen, Ternary::Chosen);
        v.record_prediction(1, Ternary::Avoided, Ternary::Unknown);
        assert_eq!(v.correct(), 1);
        assert_eq!(v.total(), 2);
    }

    // 24. InferenceRule analogy
    #[test]
    fn test_analogy_rule() {
        let mut s = TernarySpace::new(15);
        s.set(0, 1);
        s.set(1, -1);
        s.set(2, -1);
        s.set(3, 1); // shared boundary: chosen
        s.set(4, 0);
        s.set(5, 0);
        s.set(6, 1); // shared boundary: chosen
        s.set(7, -1);
        s.set(8, -1);
        s.set(9, 1);
        let map = AvoidanceMap::from_space(&s);
        let regions = map.find_negative_regions();
        let gaps = map.find_gaps_between_regions(&regions);
        assert_eq!(gaps.len(), 1);
        let result = InferenceRule::analogy(&regions, &gaps[0]);
        assert!(result.is_some());
        let inf = result.unwrap();
        assert_eq!(inf.predicted, Ternary::Avoided);
        assert_eq!(inf.inference_type, InferenceType::Analogy);
    }

    // 25. Exclusion rule
    #[test]
    fn test_exclusion_rule() {
        let mut s = TernarySpace::new(10);
        s.set(0, -1);
        s.set(1, -1);
        s.set(2, -1); // all avoided in gap
        s.set(3, -1);
        s.set(4, -1);
        s.set(5, -1);
        let gap = Gap { start: 2, end: 4, left_region: 0, right_region: 1 };
        let result = InferenceRule::exclusion(&s, &gap);
        assert!(result.is_some());
        assert_eq!(result.unwrap().inference_type, InferenceType::Exclusion);
    }

    // 26. TernarySpace display
    #[test]
    fn test_space_display() {
        let mut s = TernarySpace::new(5);
        s.set(0, -1);
        s.set(2, 1);
        assert_eq!(format!("{}", s), "✗?✓??");
    }

    // 27. Structural similarity
    #[test]
    fn test_structural_similarity() {
        let mut s = TernarySpace::new(10);
        s.set(0, -1);
        s.set(1, -1);
        s.set(2, 0);
        s.set(5, -1);
        s.set(6, -1);
        s.set(7, 0);
        let sim = s.structural_similarity(0, 3, 5, 8);
        assert!((sim - 1.0).abs() < 1e-9);
    }

    // 28. Confidence threshold filtering
    #[test]
    fn test_confidence_threshold() {
        let mut s = TernarySpace::new(20);
        s.set(0, -1);
        s.set(1, -1);
        s.set(2, -1);
        s.set(10, -1);
        s.set(11, -1);
        s.set(12, -1);
        let map = AvoidanceMap::from_space(&s);

        let high_engine = DeductionEngine::with_confidence(0.9);
        let low_engine = DeductionEngine::with_confidence(0.1);

        let high = high_engine.infer_from_map(&map);
        let low = low_engine.infer_from_map(&map);
        assert!(low.len() >= high.len(), "Lower threshold should produce >= deductions");
    }

    // 29. Full pipeline integration
    #[test]
    fn test_full_pipeline() {
        let mut space = TernarySpace::new(20);
        // Two avoidance regions
        space.set(1, -1);
        space.set(2, -1);
        space.set(3, -1);
        space.set(12, -1);
        space.set(13, -1);
        space.set(14, -1);
        // Some choices
        space.set(0, 1);
        space.set(18, 1);

        let map = AvoidanceMap::from_space(&space);
        let regions = map.find_negative_regions();
        assert_eq!(regions.len(), 2);

        let engine = DeductionEngine::new();
        let deductions = engine.infer_from_map(&map);

        let mut validator = FeedbackValidator::new();
        for d in &deductions {
            validator.record(d, space.get(d.position));
        }

        assert!(validator.total() > 0);
        assert!(validator.accuracy() >= 0.0);
    }

    // 30. Validator accuracy_for_positions
    #[test]
    fn test_accuracy_for_positions() {
        let mut v = FeedbackValidator::new();
        v.record_prediction(0, Ternary::Avoided, Ternary::Avoided);
        v.record_prediction(1, Ternary::Chosen, Ternary::Unknown);
        v.record_prediction(2, Ternary::Avoided, Ternary::Avoided);
        assert!((v.accuracy_for_positions(&[0, 2]) - 1.0).abs() < 1e-9);
        assert!((v.accuracy_for_positions(&[1]) - 0.0).abs() < 1e-9);
    }
}
