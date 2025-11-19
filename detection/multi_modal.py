from __future__ import annotations

from typing import Iterable, List, Dict, Any, Optional

from .base import ChangeDetector, DetectionResult


class MultiModalDetector(ChangeDetector):
    """
    Combines multiple detectors.

    Two voting modes:
      - use_confidence_vote=False (default): vote by discrete detections count (ratio of detectors that fired).
      - use_confidence_vote=True: vote by average confidence across detectors (continuous).
    """

    def __init__(
        self,
        detectors: Iterable[ChangeDetector],
        vote_threshold: float = 0.4,
        use_confidence_vote: bool = True,
    ):
        self.detectors: List[ChangeDetector] = list(detectors)
        self.vote_threshold = float(vote_threshold)
        self.use_confidence_vote = bool(use_confidence_vote)
        name = "+".join(detector.name for detector in self.detectors)
        super().__init__(name=name or "multi_modal")

    def reset(self) -> None:
        for detector in self.detectors:
            detector.reset()

    def _get_confidence(self, res: DetectionResult) -> float:
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
        results: List[DetectionResult] = [
            detector.update(state, action, reward, next_state, done, info)
            for detector in self.detectors
        ]

        individual: Dict[int, Dict[str, Any]] = {}
        confidences: List[float] = []
        discrete_votes = 0
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
    def __init__(
        self,
        detectors: Iterable[ChangeDetector],
        vote_threshold: float = 0.5,
        detector_weights: Optional[Iterable[float]] = None,
    ):
        self.detectors: List[ChangeDetector] = list(detectors)
        self.vote_threshold = float(vote_threshold)

        # 修复：如果未提供权重，使用均匀权重
        if detector_weights is None:
            self.detector_weights = [1.0 for _ in self.detectors]  # 均匀权重
        else:
            self.detector_weights = [float(w) for w in detector_weights]

        # 确保权重数量匹配
        if len(self.detector_weights) != len(self.detectors):
            print(f"警告: 权重数量不匹配，使用均匀权重")
            self.detector_weights = [1.0 for _ in self.detectors]

        # 归一化权重，使总权重=检测器数量
        total_weight = sum(self.detector_weights)
        self.detector_weights = [w / total_weight * len(self.detectors) for w in self.detector_weights]
        
        name = "+".join(detector.name for detector in self.detectors)
        super().__init__(name=f"weighted_{name}")

    def reset(self) -> None:
        for detector in self.detectors:
            detector.reset()

    def _get_confidence(self, res: DetectionResult) -> float:
        """从检测结果中提取置信度"""
        conf = None
        if isinstance(res.metadata, dict):
            conf = res.metadata.get("confidence")
        if conf is None:
            # 如果没有显式置信度，使用归一化的score
            conf = min(1.0, max(0.0, res.score / 10.0))  # 假设score范围0-10
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
            
            # 加权置信度
            weighted_confidence_sum += conf * weight
            
            # 加权检测（如果检测到，加上权重）
            if detected:
                weighted_detection_sum += weight

            individual[i] = {
                "detected": detected,
                "confidence": conf,
                "weight": float(weight),
                "raw_score": float(res.score) if res.score is not None else 0.0,
                "metadata": res.metadata,
            }

        # 计算加权投票比例
        vote_ratio = weighted_detection_sum / total_weight if total_weight > 0 else 0.0
        
        # 计算加权平均置信度
        combined_confidence = weighted_confidence_sum / total_weight if total_weight > 0 else 0.0

        # 修复：使用加权投票比例来决定是否检测到变化
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