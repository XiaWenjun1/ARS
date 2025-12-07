from __future__ import annotations

from typing import Iterable, List, Dict, Any, Optional

from .base import ChangeDetector, DetectionResult


class MultiModalDetector(ChangeDetector):
    """
    Combines multiple individual ChangeDetectors and aggregates their results
    to make a more robust detection decision. It supports two voting modes:
    by discrete detection count or by average confidence.
    """

    def __init__(
        self,
        detectors: Iterable[ChangeDetector],
        vote_threshold: float = 0.4,
        use_confidence_vote: bool = True,
    ):
        """
        Initializes the MultiModalDetector.

        Args:
            detectors (Iterable[ChangeDetector]): A collection of individual detectors to combine.
            vote_threshold (float): The threshold for the voting metric (either discrete ratio or avg confidence)
                                    above which a change is detected.
            use_confidence_vote (bool): If True, the detector votes based on the average confidence
                                        across all detectors. If False, it votes based on the ratio
                                        of detectors that reported a discrete 'detected' flag.
        """
        self.detectors: List[ChangeDetector] = list(detectors)
        self.vote_threshold = float(vote_threshold)
        self.use_confidence_vote = bool(use_confidence_vote)
        # Construct a name from the combined detectors.
        name = "+".join(detector.name for detector in self.detectors)
        super().__init__(name=name or "multi_modal")

    def reset(self) -> None:
        """
        Resets all individual detectors combined within this multi-modal detector.
        """
        for detector in self.detectors:
            detector.reset()

    def _get_confidence(self, res: DetectionResult) -> float:
        """
        Extracts and normalizes the confidence score from a DetectionResult.
        Prefers explicit 'confidence' in metadata, falls back to 'score',
        and clamps the value to the range [0, 1].

        Args:
            res (DetectionResult): The result from an individual detector.

        Returns:
            float: The normalized confidence score (0.0 to 1.0).
        """
        # Prefer explicit metadata confidence, fallback to score, clamp to [0,1]
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
        """
        Updates all individual detectors and aggregates their results to determine
        if a change is detected.

        Args:
            state: The current state observation.
            action: The action taken.
            reward: The reward received.
            next_state: The state after the action.
            done (bool): Whether the episode has terminated.
            info (Optional[Dict[str, Any]]): Additional information from the environment.

        Returns:
            DetectionResult: The aggregated result of the multi-modal detection.
        """
        # Update each individual detector and collect their results.
        results: List[DetectionResult] = [
            detector.update(state, action, reward, next_state, done, info)
            for detector in self.detectors
        ]

        individual: Dict[int, Dict[str, Any]] = {}
        confidences: List[float] = []
        discrete_votes = 0 # Count of detectors that discretely reported a change.
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

        # Determine the final detection based on the chosen voting mode.
        if self.use_confidence_vote:
            vote_metric = avg_confidence
            detected = vote_metric >= self.vote_threshold
            vote_type = "confidence_avg"
        else:
            vote_metric = discrete_vote_ratio
            detected = vote_metric >= self.vote_threshold
            vote_type = "discrete_ratio"

        combined_score = avg_confidence # The score of the multi-modal detector is its average confidence.

        metadata = {
            "vote_type": vote_type,
            "vote_metric": vote_metric,
            "discrete_vote_ratio": discrete_vote_ratio,
            "avg_confidence": avg_confidence,
            "individual": individual,
        }

        # Top-level consistent confidence field for easier access.
        metadata["confidence"] = combined_score

        return DetectionResult(detected=bool(detected), score=combined_score, metadata=metadata)


class WeightedMultiModalDetector(ChangeDetector):
    """
    Combines multiple individual ChangeDetectors, applying specified weights to each
    detector's contribution to the overall detection decision. This allows for
    prioritizing more reliable or relevant detectors.
    """
    def __init__(
        self,
        detectors: Iterable[ChangeDetector],
        vote_threshold: float = 0.5,
        detector_weights: Optional[Iterable[float]] = None,
    ):
        """
        Initializes the WeightedMultiModalDetector.

        Args:
            detectors (Iterable[ChangeDetector]): A collection of individual detectors to combine.
            vote_threshold (float): The threshold for the weighted vote ratio, above which a change is detected.
            detector_weights (Optional[Iterable[float]]): A list of weights corresponding to each detector.
                                                         If None, all detectors are given uniform weights.
        """
        self.detectors: List[ChangeDetector] = list(detectors)
        self.vote_threshold = float(vote_threshold)

        # Handle weights: if not provided, assign uniform weights.
        if detector_weights is None:
            self.detector_weights = [1.0 for _ in self.detectors]  # Uniform weights
        else:
            self.detector_weights = [float(w) for w in detector_weights]

        # Ensure the number of weights matches the number of detectors.
        if len(self.detector_weights) != len(self.detectors):
            print(f"Warning: Number of weights ({len(self.detector_weights)}) does not match "
                  f"number of detectors ({len(self.detectors)}). Using uniform weights instead.")
            self.detector_weights = [1.0 for _ in self.detectors]

        # Normalize weights so that their sum equals the number of detectors.
        # This makes the interpretation of vote_threshold more consistent.
        total_weight = sum(self.detector_weights)
        # Avoid division by zero if total_weight is 0 (should not happen with default 1.0 weights).
        if total_weight > 0:
            self.detector_weights = [w / total_weight * len(self.detectors) for w in self.detector_weights]
        else: # Fallback for safety, though uniform weights should prevent this.
             self.detector_weights = [1.0 for _ in self.detectors]

        name = "+".join(detector.name for detector in self.detectors)
        super().__init__(name=f"weighted_{name}")

    def reset(self) -> None:
        """
        Resets all individual detectors combined within this weighted multi-modal detector.
        """
        for detector in self.detectors:
            detector.reset()

    def _get_confidence(self, res: DetectionResult) -> float:
        """
        Extracts and normalizes the confidence score from a DetectionResult,
        considering potential score ranges.

        Args:
            res (DetectionResult): The result from an individual detector.

        Returns:
            float: The normalized confidence score (0.0 to 1.0).
        """
        conf = None
        if isinstance(res.metadata, dict):
            conf = res.metadata.get("confidence")
        if conf is None:
            # If no explicit confidence, use a normalized score.
            # Assuming score might be in a range, e.g., 0-10, normalize it to 0-1.
            # This is a heuristic and might need adjustment based on detector scores.
            conf = min(1.0, max(0.0, res.score / 10.0))  
        try:
            conf = float(conf)
        except Exception:
            conf = 0.0
        return float(max(0.0, min(1.0, conf)))

    def update(self, state, action, reward, next_state, done, info=None) -> DetectionResult:
        """
        Updates all individual detectors and aggregates their results using weights
        to determine if a change is detected.

        Args:
            state: The current state observation.
            action: The action taken.
            reward: The reward received.
            next_state: The state after the action.
            done (bool): Whether the episode has terminated.
            info (Optional[Dict[str, Any]]): Additional information from the environment.

        Returns:
            DetectionResult: The aggregated result of the weighted multi-modal detection.
        """
        # Update each individual detector and collect their results.
        results: List[DetectionResult] = [
            detector.update(state, action, reward, next_state, done, info)
            for detector in self.detectors
        ]

        individual: Dict[int, Dict[str, Any]] = {}
        weighted_confidence_sum = 0.0
        weighted_detection_sum = 0.0
        total_weight = sum(self.detector_weights) # Recalculate total_weight for consistency

        for i, (res, weight) in enumerate(zip(results, self.detector_weights)):
            conf = self._get_confidence(res)
            detected = bool(res.detected)
            
            # Accumulate weighted confidence.
            weighted_confidence_sum += conf * weight
            
            # Accumulate weighted detection: if a detector fired, add its weight.
            if detected:
                weighted_detection_sum += weight

            individual[i] = {
                "detected": detected,
                "confidence": conf,
                "weight": float(weight),
                "raw_score": float(res.score) if res.score is not None else 0.0,
                "metadata": res.metadata,
            }

        # Calculate the weighted vote ratio based on detectors that reported 'detected'.
        vote_ratio = weighted_detection_sum / total_weight if total_weight > 0 else 0.0
        
        # Calculate the combined average confidence, weighted by detector importance.
        combined_confidence = weighted_confidence_sum / total_weight if total_weight > 0 else 0.0

        # Determine overall detection based on the weighted vote ratio and threshold.
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