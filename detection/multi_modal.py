from __future__ import annotations

from typing import Iterable, List, Dict, Any, Optional

from .base import ChangeDetector, DetectionResult


class MultiModalDetector(ChangeDetector):
    """
    Combines multiple change detectors into an Ensemble.

    This class aggregates the outputs of several detectors to make a more robust 
    decision. It supports two voting strategies:
    
    1. Discrete Voting (use_confidence_vote=False):
       - Counts how many detectors flagged a change.
       - Logic: (Count of detections / Total detectors) >= threshold.
       
    2. Continuous/Confidence Voting (use_confidence_vote=True):
       - Averages the confidence scores of all detectors.
       - Logic: Average Confidence >= threshold.
    """

    def __init__(
        self,
        detectors: Iterable[ChangeDetector],
        vote_threshold: float = 0.4,
        use_confidence_vote: bool = True,
    ):
        """
        Args:
            detectors: A list of ChangeDetector instances.
            vote_threshold: The threshold required to trigger a combined detection.
            use_confidence_vote: Boolean flag to switch between voting modes.
        """
        self.detectors: List[ChangeDetector] = list(detectors)
        self.vote_threshold = float(vote_threshold)
        self.use_confidence_vote = bool(use_confidence_vote)
        
        # specific_id: Create a composite name based on sub-detectors
        name = "+".join(detector.name for detector in self.detectors)
        super().__init__(name=name or "multi_modal")

    def reset(self) -> None:
        """Reset all sub-detectors."""
        for detector in self.detectors:
            detector.reset()

    def _get_confidence(self, res: DetectionResult) -> float:
        """
        Extract a normalized confidence score [0.0, 1.0] from a result.
        
        Priority:
        1. Metadata 'confidence' field.
        2. Raw 'score' field.
        """
        # prefer explicit metadata confidence, fallback to score, clamp to [0,1]
        conf = None
        if isinstance(res.metadata, dict):
            conf = res.metadata.get("confidence")
        if conf is None:
            conf = res.score
        try:
            conf = float(conf)
        except Exception:
            conf = 0.0
        return float(max(0.0, min(1.0, conf)))

    def update(self, state, action, reward, next_state, done, info=None) -> DetectionResult:
        # Update all detectors independently
        results: List[DetectionResult] = [
            detector.update(state, action, reward, next_state, done, info)
            for detector in self.detectors
        ]

        individual: Dict[int, Dict[str, Any]] = {}
        confidences: List[float] = []
        discrete_votes = 0
        
        # Process results
        for i, res in enumerate(results):
            conf = self._get_confidence(res)
            confidences.append(conf)
            detected_flag = bool(res.detected)
            
            if detected_flag:
                discrete_votes += 1
                
            individual[i] = {
                "detected": detected_flag,
                "confidence": conf,
                "raw_score": float(res.score) if res.score is not None else 0.0,
                "metadata": res.metadata,
            }

        num_detectors = max(len(results), 1)
        avg_confidence = float(sum(confidences) / num_detectors)
        discrete_vote_ratio = float(discrete_votes / num_detectors)

        # Apply Voting Logic
        if self.use_confidence_vote:
            vote_metric = avg_confidence
            detected = vote_metric >= self.vote_threshold
            vote_type = "confidence_avg"
        else:
            vote_metric = discrete_vote_ratio
            detected = vote_metric >= self.vote_threshold
            vote_type = "discrete_ratio"

        combined_score = avg_confidence

        metadata = {
            "vote_type": vote_type,
            "vote_metric": vote_metric,
            "discrete_vote_ratio": discrete_vote_ratio,
            "avg_confidence": avg_confidence,
            "individual": individual,
        }

        # top-level consistent confidence field
        metadata["confidence"] = combined_score

        return DetectionResult(detected=bool(detected), score=combined_score, metadata=metadata)


class WeightedMultiModalDetector(ChangeDetector):
    """
    Combines multiple detectors with specific weights.
    
    This is useful when certain detectors (e.g., Latent Space) are known to be 
    more reliable or sensitive than others (e.g., simple Reward monitoring).
    """
    
    def __init__(
        self,
        detectors: Iterable[ChangeDetector],
        vote_threshold: float = 0.5,
        detector_weights: Optional[Iterable[float]] = None,
    ):
        self.detectors: List[ChangeDetector] = list(detectors)
        self.vote_threshold = float(vote_threshold)

        # Fix: If weights are not provided, use uniform weights
        if detector_weights is None:
            self.detector_weights = [1.0 for _ in self.detectors]  # Uniform weights
        else:
            self.detector_weights = [float(w) for w in detector_weights]

        # Ensure the number of weights matches the number of detectors
        if len(self.detector_weights) != len(self.detectors):
            print(f"Warning: Weight count mismatch. Using uniform weights.")
            self.detector_weights = [1.0 for _ in self.detectors]

        # Normalize weights so that the sum of weights equals the number of detectors.
        # This keeps the scale of 'vote_ratio' consistent with the unweighted version.
        total_weight = sum(self.detector_weights)
        self.detector_weights = [w / total_weight * len(self.detectors) for w in self.detector_weights]
        
        name = "+".join(detector.name for detector in self.detectors)
        super().__init__(name=f"weighted_{name}")

    def reset(self) -> None:
        for detector in self.detectors:
            detector.reset()

    def _get_confidence(self, res: DetectionResult) -> float:
        """Extract confidence from detection result."""
        conf = None
        if isinstance(res.metadata, dict):
            conf = res.metadata.get("confidence")
        if conf is None:
            # If no explicit confidence, use normalized score
            # Assumption: raw score is roughly in range 0-10
            conf = min(1.0, max(0.0, res.score / 10.0)) 
        try:
            conf = float(conf)
        except Exception:
            conf = 0.0
        return float(max(0.0, min(1.0, conf)))

    def update(self, state, action, reward, next_state, done, info=None) -> DetectionResult:
        results: List[DetectionResult] = [
            detector.update(state, action, reward, next_state, done, info)
            for detector in self.detectors
        ]

        individual: Dict[int, Dict[str, Any]] = {}
        weighted_confidence_sum = 0.0
        weighted_detection_sum = 0.0
        total_weight = sum(self.detector_weights)

        for i, (res, weight) in enumerate(zip(results, self.detector_weights)):
            conf = self._get_confidence(res)
            detected = bool(res.detected)
            
            # Weighted Confidence
            weighted_confidence_sum += conf * weight
            
            # Weighted Detection (Add weight only if detector fired)
            if detected:
                weighted_detection_sum += weight

            individual[i] = {
                "detected": detected,
                "confidence": conf,
                "weight": float(weight),
                "raw_score": float(res.score) if res.score is not None else 0.0,
                "metadata": res.metadata,
            }

        # Calculate weighted vote ratio
        vote_ratio = weighted_detection_sum / total_weight if total_weight > 0 else 0.0
        
        # Calculate weighted average confidence
        combined_confidence = weighted_confidence_sum / total_weight if total_weight > 0 else 0.0

        # Fix: Use weighted vote ratio to decide if change is detected
        detected = vote_ratio >= self.vote_threshold

        metadata = {
            "vote_type": "weighted_detection",
            "vote_ratio": vote_ratio,
            "combined_confidence": combined_confidence,
            "weighted_detection_sum": weighted_detection_sum,
            "total_weight": total_weight,
            "individual": individual,
            "weights": self.detector_weights,
        }
        metadata["confidence"] = combined_confidence

        return DetectionResult(
            detected=bool(detected), 
            score=combined_confidence, 
            metadata=metadata
        )