# Proactive Error Detection with NVIDIA Cosmos Reason 2

A video analytics AI agent that detects procedural errors and tracks objects in cooking videos using NVIDIA Cosmos Reason 2. Built for the [NVIDIA Cosmos Cookoff](https://luma.com/nvidia-cosmos-cookoff).

## Overview

Can vision-language models proactively identify mistakes during procedural tasks? I benchmark Cosmos Reason 2 on two challenges:

1. **Proactive Error Recognition:** Detect when someone deviates from recipe instructions
2. **Episodic Memory:** Track objects when they leave the camera view

**Why it matters:** True AI assistants should warn users about mistakes in progress—not just answer questions after the fact. This requires physical common sense, temporal reasoning, and proactive awareness.

---

## Experiments

### Dataset: CaptainCook4D

94.5 hours of egocentric cooking videos with intentional errors and recipe annotations.

### Experiment 1: Recipe-Conditioned Error Detection

**Question:** Can Cosmos Reason 2 detect procedural errors by comparing video to recipe text?

**Setup:**

- Input: Recipe instructions + video clip
- Output: Error detected (Yes/No) + explanation + timestamp
- Test set: 20-30 videos (mix of correct/error executions)

**Metrics:**

- Error detection accuracy
- Temporal localization error (seconds)
- False positive rate

### Experiment 2: Episodic Memory (Object Tracking)

**Question:** Can Cosmos Reason 2 remember where objects are after they leave the frame?

**Setup:**

- Input: Video where object goes out of view
- Output: Object's last known location
- Test objects: Knife, bowl, cutting board, containers

**Metrics:**

- Location recall accuracy
- Temporal precision ("when was it last visible?")

---

## Model Configuration

- **Model:** NVIDIA Cosmos Reason 2B
- **Video sampling:** 4 FPS
- **Max tokens:** 1000

---

## Results

[Coming soon after experiments complete]

---

## Applications

- Manufacturing quality control
- Assistive robotics for cooking/assembly
- Real-time training feedback systems
- Autonomous vehicle inspection

---

## Installation & Usage

[Coming soon]

---

## Acknowledgments

Built for the NVIDIA Cosmos Cookoff. Dataset: [CaptainCook4D](https://github.com/CaptainCook4D).

---

## License

MIT