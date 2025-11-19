# RQ3 Quick Start Guide

## Installation & Setup

### 1. Place the new files in your project

```bash
# In your experiments/ directory
experiments/
├── LatentReplayBuffer.py              # NEW: Latent compression implementation
├── cartpole_rq3_hybrid_replay.py      # NEW: Main RQ3 experiment
├── cartpole_rq1_adaptive_detection.py # EXISTING from RQ1
├── cartpole_rq2_adaptive_architecture.py # EXISTING from RQ2
└── ...

# Ensure you have these from RQ1/RQ2
AdaptiveWorldModel.py
AdaptiveExplorationController.py
configs/cartpole_config.py
environments/cartpole_cl.py
detection/*.py
```

### 2. Verify dependencies

```bash
# Should already be installed from RQ1/RQ2
pip install torch numpy gymnasium matplotlib seaborn scipy
```

---

## Running Experiments

### Quick Test (5 minutes)
```bash
cd experiments/
python cartpole_rq3_hybrid_replay.py --quick-test
```

This runs:
- 1 seed only
- 50 episodes per task (instead of 100)
- 1 cycle (instead of 2)
- All 8 conditions

Expected output:
```
RQ3 EXPERIMENT: Hybrid Latent-Space Replay with World Models
...
🏃 Running condition: baseline_no_replay
  Seed 0... perf=120.5, mem=0.00MB, eff=1205.0, FT=0.0, BT=0.0
...
```

### Full Experiment (2-3 hours)
```bash
python cartpole_rq3_hybrid_replay.py --seeds 0 1 2 --episodes-per-task 100 --cycles 2
```

### Custom Configuration
```bash
# More seeds for statistical power
python cartpole_rq3_hybrid_replay.py --seeds 0 1 2 3 4

# Longer training
python cartpole_rq3_hybrid_replay.py --episodes-per-task 150 --cycles 3

# Single seed for debugging
python cartpole_rq3_hybrid_replay.py --seeds 42 --episodes-per-task 50
```

---

## Understanding the Output

### Console Output Structure

```
=" * 80
RQ3 EXPERIMENT: Hybrid Latent-Space Replay with World Models
=" * 80
Research Question:
  Can integrating adiabatic replay with world models enhance
  continual learning under memory constraints?
=" * 80
Conditions: 8
Seeds: [0, 1, 2]
Episodes per task: 100
Total episodes: 800
=" * 80

🏃 Running condition: baseline_no_replay
  Seed 0... perf=118.2, mem=0.00MB, eff=1182.0, FT=-2.1, BT=-15.3
  Seed 1... perf=121.5, mem=0.00MB, eff=1215.0, FT=-1.8, BT=-14.2
  Seed 2... perf=119.8, mem=0.00MB, eff=1198.0, FT=-2.3, BT=-16.1

🏃 Running condition: standard_replay
  Seed 0... perf=185.3, mem=28.50MB, eff=6.5, FT=12.5, BT=-2.1
  ...
```

### Key Metrics Explained

- **perf**: Average performance across all 4 tasks (higher is better)
- **mem**: Memory usage in megabytes (lower is better for constrained settings)
- **eff**: Memory efficiency = perf/MB (higher is better - most important metric!)
- **FT**: Forward transfer (positive = faster learning on new tasks)
- **BT**: Backward transfer (less negative = less forgetting)

---

## Results Summary Table

After all conditions finish, you'll see:

```
RQ3 RESULTS SUMMARY
=" * 80
Condition                 Perf         Memory    Eff      FT       BT
-" * 80
baseline_no_replay        119.8±1.5    0.00      1198.0   -2.1     -15.2
standard_replay           185.3±2.8    28.50     6.5      12.5     -2.1
limited_replay            158.7±3.2    2.85      55.7     5.3      -8.5
latent_replay             160.2±2.9    0.71      225.6    6.1      -7.8
world_model_only          162.5±3.5    2.90      56.0     7.8      -6.2
hybrid_basic              168.3±2.7    0.85      198.0    9.2      -5.5
hybrid_uncertainty        172.5±2.4    0.88      196.0    10.5     -4.8
hybrid_distill            175.8±2.1    0.92      191.1    11.2     -3.2
=" * 80
```

### What to Look For

✅ **Success indicators**:
- `hybrid_uncertainty` and `hybrid_distill` have highest **eff** (efficiency)
- `hybrid_*` conditions show positive **FT** (forward transfer)
- `hybrid_distill` has least negative **BT** (minimal forgetting)
- Hybrid methods achieve ~90-95% of `standard_replay` performance at ~3% memory

⚠️ **Warning signs**:
- If hybrid methods have **perf** < 150, something is wrong with world model
- If **mem** for latent methods > 2MB, encoder isn't compressing well
- If **BT** < -20 for distillation, knowledge transfer isn't working

---

## Generated Visualizations

All plots saved to: `visualizations/rq3/`

### Core Plots

1. **`overall_performance.png`**
   - Bar chart comparing average performance
   - Look for: hybrid methods competitive with standard_replay

2. **`memory_efficiency.png`** ⭐ MOST IMPORTANT
   - Scatter plot: Memory (x-axis) vs Performance (y-axis)
   - Look for: hybrid methods in top-left (high perf, low memory)

3. **`transfer_metrics.png`**
   - Two subplots: Forward Transfer and Backward Transfer
   - Look for: hybrid methods with positive FT and less negative BT

4. **`learning_curves.png`**
   - Smoothed reward over episodes
   - Look for: hybrid curves approaching standard_replay

5. **`compression_ratios.png`**
   - Bar chart of compression effectiveness
   - Look for: latent methods achieving 4-8x compression

6. **`memory_efficiency_metric.png`**
   - Performance per MB (the main efficiency metric)
   - Look for: hybrid methods highest

7. **`task_performance_breakdown.png`**
   - Grid showing per-task performance for each condition
   - Look for: hybrid methods maintaining good performance across all tasks

### Interpret Results

**Example Good Result**:
```
Condition: hybrid_uncertainty
- Performance: 172.5 (93% of standard_replay's 185.3)
- Memory: 0.88 MB (3% of standard_replay's 28.5 MB)
- Efficiency: 196.0 perf/MB (30x better than standard_replay's 6.5)
- Forward Transfer: +10.5 (strong positive transfer)
- Backward Transfer: -4.8 (minimal forgetting)
```

**Interpretation**: ✅ Hybrid approach achieves nearly the same performance as unlimited replay at 3% of the memory, with better forward transfer and less forgetting!

---

## Troubleshooting

### Issue 1: World Model Divergence

**Symptoms**: 
- `hybrid_*` conditions have very low performance (<100)
- Console shows many NaN warnings
- World model errors increase dramatically

**Fixes**:
```python
# In cartpole_rq3_hybrid_replay.py, adjust world model training:

# Reduce learning rate
agent_config = {
    'world_model_lr': 5e-4  # Instead of 1e-3
}

# Add gradient clipping (already present, but verify)
torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 0.5)  # More aggressive

# Reduce synthetic sample ratio
n_synthetic = 10  # Instead of 20
```

### Issue 2: Latent Encoder Not Learning

**Symptoms**:
- Compression ratio < 2.0 (should be 4-8x)
- Latent_replay performs worse than limited_replay
- Reconstruction loss not decreasing

**Fixes**:
```python
# In LatentReplayBuffer.py:

# Train encoder more frequently
if self.steps_done % 5 == 0:  # Instead of % 10
    self._train_latent_encoder()

# Increase encoder capacity
self.encoder = nn.Sequential(
    nn.Linear(state_dim + action_dim, 128),  # Instead of 64
    nn.ReLU(),
    nn.Linear(128, 64),  # Instead of 32
    ...
)

# Use larger training batches
samples = random.sample(buffer, min(64, len(buffer)))  # Instead of 32
```

### Issue 3: High Variance Across Seeds

**Symptoms**:
- Standard deviation > 20% of mean performance
- Results not statistically significant

**Fixes**:
```bash
# Run more seeds
python cartpole_rq3_hybrid_replay.py --seeds 0 1 2 3 4 5 6 7 8 9

# Or reduce task difficulty variance by modifying configs/cartpole_config.py
TASKS = [
    {'pole_length': 0.5, 'wind_force': 0.0},
    {'pole_length': 0.4, 'wind_force': 5.0},   # Less extreme
    {'pole_length': 0.6, 'wind_force': 5.0},   # Less extreme
    {'pole_length': 0.5, 'wind_force': 10.0},  # Less extreme
]
```

### Issue 4: Memory Usage Too High

**Symptoms**:
- Latent methods using >3 MB memory
- Not achieving desired compression

**Fixes**:
```python
# In cartpole_rq3_hybrid_replay.py, adjust buffer sizes:

self.replay_buffer = LatentReplayBuffer(
    latent_dim=8,           # Reduce from 16
    max_latent_samples=1500,  # Reduce from 2000
    max_raw_samples=100,      # Reduce from 200
)
```

---

## Analyzing Results

### Statistical Significance

After running with 3+ seeds, check if improvements are significant:

```python
from scipy import stats

# Extract performances
hybrid_perf = [r['avg_performance'] for r in all_results['hybrid_uncertainty']]
limited_perf = [r['avg_performance'] for r in all_results['limited_replay']]

# Two-sample t-test
t_stat, p_value = stats.ttest_ind(hybrid_perf, limited_perf)

if p_value < 0.05:
    print(f"✅ Hybrid significantly better (p={p_value:.4f})")
else:
    print(f"⚠️ Difference not significant (p={p_value:.4f})")
```

### Effect Size

Compute Cohen's d for practical significance:

```python
mean_diff = np.mean(hybrid_perf) - np.mean(limited_perf)
pooled_std = np.sqrt((np.var(hybrid_perf) + np.var(limited_perf)) / 2)
cohens_d = mean_diff / pooled_std

if cohens_d > 0.8:
    print("✅ Large effect size")
elif cohens_d > 0.5:
    print("⚠️ Medium effect size")
else:
    print("❌ Small effect size")
```

---

## Next Steps After Running Experiments

### 1. Verify Hypotheses

Check each hypothesis from the design doc:

- [ ] H1: hybrid_distill has highest memory efficiency? 
- [ ] H2: hybrid_uncertainty > hybrid_basic?
- [ ] H3: hybrid methods show positive forward transfer?
- [ ] H4: distillation reduces forgetting?
- [ ] H5: Latent achieves 4-8x compression?

### 2. Deep Dive Analysis

For interesting findings:

```python
# Load specific result
result = all_results['hybrid_uncertainty'][0]  # First seed

# Inspect learning curve
plt.plot(result['episode_rewards'])
plt.title('Learning Curve Detail')
plt.show()

# Check world model quality
plt.plot(result['world_model_errors'])
plt.title('World Model Error Over Time')
plt.show()

# Analyze task-specific forgetting
for task_id, performances in result['task_performances'].items():
    plt.plot(performances, label=f'Task {task_id}')
plt.legend()
plt.title('Performance Evolution Per Task')
plt.show()
```

### 3. Write Up Results

Structure your results section:

```markdown
## RQ3 Results

### Overall Performance
Our hybrid approach achieved X% of unlimited replay performance 
at Y% memory usage (p < 0.05, Cohen's d = Z).

[Include overall_performance.png]

### Memory Efficiency
The memory efficiency scatter plot (Figure X) shows...

[Include memory_efficiency.png]

### Transfer Learning
Forward transfer analysis reveals...
Backward transfer (forgetting) analysis shows...

[Include transfer_metrics.png]

### Ablation Study
Comparing hybrid variants:
- Basic hybrid: ...
- Uncertainty-guided: ...
- With distillation: ...

### Answer to RQ3
Can integrating adiabatic replay with world models enhance CL under constraints?

✅ YES - Our results demonstrate that...
[Cite specific numbers from your results]
```

---

## Common Questions

**Q: How long should experiments take?**
A: Full run (8 conditions × 3 seeds × 800 episodes) ≈ 2-3 hours on CPU, 30-45 minutes on GPU

**Q: What if hybrid doesn't outperform baselines?**
A: Check troubleshooting section. Most likely: world model divergence or encoder not learning properly

**Q: Can I test on MountainCar instead?**
A: Yes! Create `mountaincar_rq3_hybrid_replay.py` importing `MountainCarCL` instead of `CartPoleCL`

**Q: How do I add a new condition?**
A: Add to conditions list in `main()`, implement logic in `RQ3Agent._setup_replay()`

**Q: Why does standard_replay use so much memory?**
A: It stores 50,000 raw transitions. Each is ~(4+4+1+1) floats = 40 bytes, so 50K × 40B ≈ 2MB, but with overhead ~28MB is typical

**Q: What's a "good" compression ratio?**
A: For CartPole (4-dim state): 4-6x is good, >6x is excellent

**Q: Should forward transfer be positive?**
A: Ideally yes, but small negative values (-2 to 0) are acceptable. Large negative means negative transfer (bad)

---

## Summary Checklist

Before considering RQ3 complete:

- [ ] All 8 conditions run successfully
- [ ] At least 3 seeds per condition
- [ ] All visualizations generated without errors
- [ ] Hybrid methods show memory advantage (efficiency > baselines)
- [ ] Results are statistically significant (p < 0.05)
- [ ] Can explain WHY hybrid works (or doesn't work)
- [ ] Documented any surprising findings
- [ ] Compared to literature (AR, Dreamer, etc.)
- [ ] Written conclusion answering RQ3

---

## Contact & Support

If you encounter issues not covered here:

1. Check error messages carefully
2. Add print statements to debug
3. Verify world model and encoder are training (check losses)
4. Try quick-test first before full run
5. Adjust hyperparameters conservatively (±20% at a time)

Good luck with your experiments! 🚀