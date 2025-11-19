# RQ3 Experimental Design: Hybrid Latent-Space Replay with World Models

## Research Question

**Can integrating adiabatic replay with world models enhance continual learning under memory constraints?**

This question addresses the limitations of both pure replay-based methods (high memory) and pure world-model methods (potential instability), proposing a hybrid approach that combines:
1. **Latent compression** (from adiabatic replay principles)
2. **World model synthesis** (from Dreamer-style approaches)
3. **Uncertainty-guided sampling** (novel contribution)
4. **Knowledge distillation** (for long-term retention)

---

## Experimental Design Overview

### Motivation

- **Problem 1**: Traditional replay buffers require significant memory (O(state_dim × buffer_size))
- **Problem 2**: World models can generate infinite data but may produce unrealistic samples
- **Solution**: Compress real experiences to latent space + use world model to synthesize additional experiences + guide sampling with uncertainty estimates

### Innovation Points

1. **Latent Compression with Learned Encoder**: Compress state-action pairs to 16-dimensional latent space (4-8x memory reduction)
2. **Uncertainty-Guided Prioritization**: Combine epistemic (prediction error) and aleatoric (world model std) uncertainty
3. **Hybrid Replay Strategy**: Mix compressed real experiences with synthetic world model samples
4. **Knowledge Distillation**: Preserve old task knowledge when learning new tasks

---

## Experimental Conditions

### 8 Core Conditions

| Condition | Description | Purpose |
|-----------|-------------|---------|
| `baseline_no_replay` | DQN without any replay | Lower bound baseline |
| `standard_replay` | Unlimited experience replay | Upper bound baseline (no memory constraint) |
| `limited_replay` | Constrained buffer (2000 samples) | Fair memory-constrained baseline |
| `latent_replay` | Latent compression only | Test compression effectiveness alone |
| `world_model_only` | World model synthesis only | Test world model alone |
| `hybrid_basic` | Latent + WM (equal sampling) | Basic hybrid approach |
| `hybrid_uncertainty` | Latent + WM (uncertainty-guided) | **Main innovation** |
| `hybrid_distill` | Full system + distillation | Complete system |

### Ablation Study Design

The conditions form a systematic ablation:
- **No replay** vs **With replay**: Tests necessity of memory
- **Unlimited** vs **Limited**: Establishes memory constraint impact
- **Latent only** vs **WM only** vs **Hybrid**: Tests synergy
- **Basic hybrid** vs **Uncertainty-guided**: Tests sampling strategy
- **With distillation** vs **Without**: Tests forgetting mitigation

---

## Implementation Components

### 1. Latent Encoder Architecture

```python
Encoder: [state+action] → 64 → 32 → 16 (latent)
Decoder: 16 (latent) → 32 → 64 → [state+action]
```

- **Training**: Online updates on recent experiences
- **Loss**: MSE reconstruction loss for state and action
- **Compression**: ~16 / (state_dim + action_dim) ≈ 3-4x for CartPole

### 2. Uncertainty Estimator

**Epistemic Uncertainty** (model uncertainty):
- Track prediction errors over sliding window
- Compute ratio: recent_error / baseline_error
- High ratio → high epistemic uncertainty

**Aleatoric Uncertainty** (data uncertainty):
- Use world model ensemble standard deviation
- `next_std_mean` from ensemble predictions
- High std → high aleatoric uncertainty

**Combined Score**:
```
uncertainty = 0.6 × epistemic + 0.4 × aleatoric
priority = uncertainty × reward_scale × terminal_bonus
```

### 3. Hybrid Sampling Strategy

**Temperature-Scaled Priority Sampling**:
```python
priorities = [sample.priority for sample in buffer]
scaled = priorities ** (1.0 / temperature)
probs = scaled / sum(scaled)
```

- Temperature = 0.8 for uncertainty-guided (more prioritized)
- Temperature = 1.0 for basic hybrid (more uniform)

**Mixing Real and Synthetic**:
- Real experiences: High priority, stored in latent buffer
- Synthetic experiences: Lower priority (0.85x), generated periodically
- Max synthetic ratio: 20% of buffer

### 4. Knowledge Distillation

**Teacher-Student Framework**:
- Teacher: Policy snapshot after completing task T
- Student: Current policy learning task T+1
- Loss: `α × CE(labels) + (1-α) × KL(teacher || student)`

**Temperature Scaling**:
- Use T=2.0 to soften probability distributions
- Helps transfer "dark knowledge" about action preferences

---

## Evaluation Metrics

### Primary Metrics

1. **Average Performance**: Mean reward across all tasks
   ```
   avg_perf = mean([eval_reward(task_i) for i in all_tasks])
   ```

2. **Memory Efficiency**: Performance per megabyte
   ```
   mem_eff = avg_performance / (memory_MB + 0.1)
   ```

3. **Forward Transfer**: Learning speed on new tasks
   ```
   FT = mean([perf_curve(task_i) - perf_curve(task_0) for i > 0])
   ```
   - Positive → faster learning on later tasks
   - Indicates positive knowledge transfer

4. **Backward Transfer**: Forgetting on old tasks
   ```
   BT = mean([final_perf(task_i) - peak_perf(task_i) for i < current])
   ```
   - Negative → catastrophic forgetting
   - Positive → improvement or no forgetting

### Secondary Metrics

5. **Compression Ratio**: Memory reduction factor
   ```
   ratio = uncompressed_size / compressed_size
   ```

6. **Sample Efficiency**: Episodes to convergence per task

7. **Stability**: Variance in performance across seeds

---

## Experimental Protocol

### Training Procedure

1. **Task Sequence**: 4 tasks × 100 episodes × 2 cycles = 800 total episodes
2. **Task Boundaries**: 
   - Evaluate all tasks before switching
   - Save teacher network (for distillation)
   - Update buffer task boundary markers
   
3. **Per-Episode Flow**:
   ```
   for each episode:
       1. Interact with environment
       2. Store transition in buffer (with uncertainty)
       3. Update world model (if applicable)
       4. Update policy from replay
       5. Generate synthetic samples (if hybrid)
       6. Train latent encoder (if using latent)
   ```

4. **Evaluation Protocol**:
   - Every 20 episodes: Evaluate all tasks (ε=0)
   - 10 episodes per task evaluation
   - Track task-specific performance curves

### Hyperparameters

**Policy Network**:
- Hidden dim: 64
- Learning rate: 1e-3
- Batch size: 64
- Target update: every 200 steps
- γ: 0.99

**World Model**:
- Hidden dim: 64
- Ensemble size: 3
- Learning rate: 1e-3

**Latent Encoder**:
- Latent dim: 16
- Update frequency: every 10 steps
- Batch size: 32

**Buffer Sizes**:
- Standard replay: 50,000 samples
- Limited replay: 2,000 samples
- Latent buffer: 2,000 latent + 200 raw samples
- Synthetic generation: 20 samples per 5 episodes

---

## Expected Results & Hypotheses

### Hypothesis 1: Memory Efficiency
**H1**: Hybrid methods achieve higher performance per MB than baselines

Expected ranking:
```
hybrid_distill > hybrid_uncertainty > latent_replay > limited_replay
```

**Reasoning**: Latent compression reduces memory, world model amplifies limited samples

### Hypothesis 2: Performance
**H2**: Hybrid uncertainty-guided sampling outperforms basic hybrid

Expected:
```
hybrid_uncertainty > hybrid_basic
```

**Reasoning**: Prioritizing uncertain samples focuses learning on informative transitions

### Hypothesis 3: Forward Transfer
**H3**: World model conditions show positive forward transfer

Expected:
```
FT(hybrid_*) > FT(latent_only) > FT(limited_replay)
```

**Reasoning**: World model learns reusable transition dynamics across tasks

### Hypothesis 4: Backward Transfer
**H4**: Distillation reduces forgetting

Expected:
```
BT(hybrid_distill) > BT(hybrid_uncertainty) > BT(limited_replay)
```

**Reasoning**: Knowledge distillation explicitly preserves old task knowledge

### Hypothesis 5: Compression
**H5**: Latent compression achieves 4-8x ratio without major performance loss

Expected:
```
latent_replay ≈ 90-95% of standard_replay performance
at ~12-25% memory usage
```

---

## Analysis Plan

### Quantitative Analysis

1. **ANOVA Test**: Compare mean performance across conditions
2. **Paired t-tests**: 
   - Hybrid vs limited_replay
   - Uncertainty-guided vs basic hybrid
   - With distillation vs without

3. **Regression Analysis**:
   - Performance ~ memory_usage + forward_transfer + backward_transfer
   - Identify which factors best predict overall performance

4. **Efficiency Frontier**:
   - Plot performance vs memory scatter
   - Identify Pareto-optimal solutions

### Qualitative Analysis

1. **Learning Curve Inspection**: Identify when each method plateaus
2. **Task-Specific Performance**: Which tasks benefit most from hybrid approach?
3. **Failure Mode Analysis**: When does world model generate poor samples?
4. **Uncertainty Calibration**: Are uncertainty estimates well-calibrated?

---

## Visualization Suite

### Core Plots

1. **Overall Performance Bar Chart**: Mean ± std for all conditions
2. **Memory Efficiency Scatter**: Performance vs memory usage
3. **Transfer Metrics**: Side-by-side forward and backward transfer
4. **Compression Ratios**: Bar chart for latent-based methods
5. **Learning Curves**: Smoothed reward curves for key conditions
6. **Memory Efficiency Metric**: Performance/MB bar chart
7. **Task Performance Heatmap**: Performance breakdown by task
8. **Memory Usage Over Time**: Track buffer growth during training

### Advanced Visualizations

9. **Uncertainty Calibration Curve**: Predicted vs actual error
10. **Sample Distribution**: Real vs synthetic sample ratios over time
11. **Forgetting Timeline**: Track old task performance degradation
12. **Ablation Waterfall**: Sequential impact of each component

---

## Integration with Existing Codebase

### File Structure
```
experiments/
├── cartpole_rq3_hybrid_replay.py     # Main experiment script
├── LatentReplayBuffer.py              # Latent buffer implementation
└── visualizations/
    └── rq3/
        ├── overall_performance.png
        ├── memory_efficiency.png
        ├── transfer_metrics.png
        └── rq3_summary.json

detection/                              # Reuse from RQ1
├── base.py
└── ...

AdaptiveWorldModel.py                   # Reuse from RQ2
AdaptiveExplorationController.py        # Reuse from RQ2

configs/
└── cartpole_config.py                  # Same tasks as RQ1/RQ2

environments/
└── cartpole_cl.py                      # Same CL formulation
```

### Running Experiments

**Full experiment**:
```bash
python cartpole_rq3_hybrid_replay.py --seeds 0 1 2 --episodes-per-task 100 --cycles 2
```

**Quick test**:
```bash
python cartpole_rq3_hybrid_replay.py --quick-test
```

**Single condition**:
```python
from cartpole_rq3_hybrid_replay import run_rq3_experiment, CartPoleConfig

cfg = CartPoleConfig()
result = run_rq3_experiment(cfg, "hybrid_uncertainty", seed=0, episodes_per_task=100, cycles=2)
```

---

## Expected Timeline

### Phase 1: Implementation (1-2 days)
- [x] LatentReplayBuffer class
- [x] Uncertainty estimation
- [x] RQ3 agent wrapper
- [x] Experimental runner
- [x] Visualization suite

### Phase 2: Debugging (1 day)
- [ ] Test each condition independently
- [ ] Verify latent encoder training
- [ ] Check world model integration
- [ ] Validate metrics computation

### Phase 3: Experimentation (1-2 days)
- [ ] Run all conditions × 3 seeds
- [ ] Monitor for issues (NaN, divergence)
- [ ] Collect results

### Phase 4: Analysis (1 day)
- [ ] Generate all visualizations
- [ ] Perform statistical tests
- [ ] Write interpretation
- [ ] Answer research question

---

## Key Insights Expected

### Technical Contributions

1. **Latent compression is viable**: 4-8x memory reduction without major performance loss
2. **Uncertainty guides effective sampling**: Prioritizing uncertain samples improves learning
3. **World models amplify limited data**: Synthetic samples compensate for storage constraints
4. **Distillation mitigates forgetting**: Explicit knowledge preservation helps long-term retention

### Practical Implications

1. **Memory-constrained deployment**: Hybrid approach enables CL on resource-limited devices
2. **Sample efficiency**: Reduces need for massive replay buffers
3. **Modularity**: Components (latent, WM, distillation) can be mixed-and-matched
4. **Scalability**: Compression ratio improves with higher-dimensional state spaces

### Theoretical Insights

1. **Replay is about information, not raw data**: Latent representations preserve task-relevant information
2. **Uncertainty quantification matters**: Knowing what we don't know guides effective learning
3. **Generative models complement discriminative models**: WM reasoning + policy optimization = powerful combination
4. **Multi-faceted approach to forgetting**: Both architectural (replay) and algorithmic (distillation) solutions needed

---

## Fallback Plans

### If hybrid doesn't outperform baselines:

**Diagnosis checklist**:
1. Is latent encoder learning good representations? (Check reconstruction loss)
2. Is world model generating realistic samples? (Inspect predicted states)
3. Is uncertainty estimate well-calibrated? (Plot predicted vs actual error)
4. Is sampling temperature appropriate? (Try different values)
5. Is synthetic sample ratio too high? (Reduce to 10-15%)

**Potential fixes**:
- Increase latent dimension (16 → 32)
- Add variational component (VAE instead of AE)
- Use ensemble world model uncertainty more heavily
- Adjust mixing ratio (more real, less synthetic)
- Train encoder for longer before using

### If memory efficiency is poor:

- Reduce latent dimension further (16 → 8)
- Eliminate raw buffer entirely
- Use fixed-size circular buffer for encoder training
- Compress rewards/done flags separately

### If forgetting is severe:

- Increase distillation weight (α = 0.7 → 0.8)
- Add EWC or other forgetting mitigation
- Store more samples per task
- Use task-specific replay buffers

---

## Success Criteria

### Minimum Viable Result
- Hybrid method achieves ≥ 90% of standard replay performance
- Memory usage ≤ 30% of standard replay
- Forward transfer ≥ 0 (no negative transfer)
- Backward transfer ≥ -20 (limited forgetting)

### Strong Result
- Hybrid method achieves ≥ 95% of standard replay performance
- Memory usage ≤ 20% of standard replay
- Forward transfer > +5 (clear positive transfer)
- Backward transfer > -10 (minimal forgetting)
- Memory efficiency metric highest among all conditions

### Excellent Result
- Hybrid method achieves ≥ 100% of standard replay (matches or exceeds)
- Memory usage ≤ 15% of standard replay
- Forward transfer > +10
- Backward transfer > -5
- Clear statistical significance (p < 0.05) vs all baselines
- Compression ratio > 6x

---

## Connection to Literature

### Addresses Gaps in:

1. **Adiabatic Replay (AR)**: 
   - AR relies on task boundaries (ours is continuous)
   - AR uses simple importance sampling (ours uses uncertainty)
   
2. **Dreamer**:
   - Dreamer lacks memory efficiency mechanisms
   - Dreamer doesn't prioritize samples by informativeness
   
3. **MoReFree**:
   - MoReFree focuses on modular policies
   - Ours focuses on memory-efficient replay

### Builds Upon:

- **Experience Replay**: Core idea but with compression
- **World Models**: Use for data augmentation
- **Continual Learning**: Address catastrophic forgetting
- **Active Learning**: Uncertainty-guided sampling

---

## Contribution Summary

### Novelty (4 I's)

1. **Informedness**: Built on solid understanding of replay, world models, and CL literature
2. **Innovativeness**: Novel combination of latent compression + world models + uncertainty
3. **Inventiveness**: Non-trivial technical implementation with multiple components
4. **Impactfulness**: Practical solution to memory constraints in CL; generalizable insights

### Expected Publications

- Main claim: "Hybrid latent-space replay achieves X% of full replay at Y% memory"
- Secondary: "Uncertainty-guided sampling improves sample efficiency by Z%"
- Tertiary: "Knowledge distillation reduces forgetting in hybrid systems"

---

## Summary

RQ3 investigates a comprehensive solution to memory-constrained continual learning by combining:
- **Latent compression** (reduce memory 4-8x)
- **World model synthesis** (augment limited data)
- **Uncertainty-guided sampling** (focus on informative transitions)
- **Knowledge distillation** (preserve old knowledge)

The experimental design includes 8 conditions systematically ablating each component, with rigorous evaluation on performance, memory efficiency, forward transfer, and backward transfer metrics. Expected result: hybrid approaches break the memory-performance trade-off, enabling effective continual learning under resource constraints.