# ARS
autonomous robotic system

## How to Run Experiments

To run the experiments, first navigate to the `experiments` directory from the project's root folder:

```bash
cd experiments
```

Then, you can execute any of the research question (RQ) scripts using Python. Below are the specific commands for each experiment.

### RQ1: Adaptive Detection

This experiment evaluates different methods for detecting environmental changes.

**CartPole:**
```bash
python cartpole_rq1_adaptive_detection.py --seeds 0 --episodes-per-task 200 --cycles 1 --warmup-episodes 50
```

**MountainCar:**
```bash
python mountaincar_rq1_adaptive_detection.py --seeds 0 --episodes-per-task 100 --cycles 1 --warmup-episodes 20
```

### RQ2: Adaptive Architecture

This experiment investigates adaptive architectures for the agent's policy and world model.

**CartPole:**
```bash
python cartpole_rq2_adaptive_architecture.py --seeds 0 --episodes-per-task 200 --cycles 1 --warmup-episodes 50
```

**MountainCar:**
```bash
python mountaincar_rq2_adaptive_architecture.py --seeds 0 --episodes-per-task 100 --cycles 1 --warmup-episodes 20
```

### RQ3: Hybrid Replay

This experiment explores a hybrid replay strategy that combines latent and conventional replay.

**CartPole:**
```bash
python cartpole_rq3_hybrid_replay.py --seeds 0 --episodes-per-task 200 --cycles 1 --warmup-episodes 50
```

**MountainCar:**
```bash
python mountaincar_rq3_hybrid_replay.py --seeds 0 --episodes-per-task 100 --cycles 1 --warmup-episodes 20
```

### Command-line Arguments

*   `--seeds`: Specify one or more random seeds for the experiment. To run with multiple seeds, separate the numbers with spaces (e.g., `--seeds 0 1 2`).
*   `--episodes-per-task`: Set the number of episodes to run for each continual learning task.
*   `--cycles`: Define the number of continual learning cycles.
*   `--warmup-episodes`: Set the number of episodes for the initial warm-up phase before the main training begins.
