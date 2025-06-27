# DLLMs Benchmark & Visualization Guide

Subject: Experiments
Motivation: 規劃 Benchmarking Fast-dLLM 的計畫
Sub Program: Phison 2025.26 Proposal

## 1 . Metrics

| Metric | Unit | Goal | Notes |
| --- | --- | --- | --- |
| **Accuracy** | ratio (0–1) or % | ↑ | Task-specific (exact match, BLEU, perplexity …). For example GSM8K Accuracy. |
| **Throughput** | token/s | ↑ | End-to-end token/s. |
| **Score** | 0–100 | ↑ | Composite index: `Score = w_acc × Acc_norm + w_thr × Thr_norm` with `w_acc + w_thr = 1`. Both metrics are **min-max-normalised** to 0–100, so the weights reflect *importance* rather than *magnitude*. |

> Keep the raw values; compute normalisation & scoring in the analysis notebook so future engineers can quickly experiment with different weight splits (e.g. 70 / 30, 30 / 70).
> 

---

## 2 . Control Variables

| Variable | Typical Range | Hypothesised Effect |
| --- | --- | --- |
| **denoising_step** | 16 - 256 | More steps ⇒ ↑ accuracy, ↓ throughput |
| **block_size** | 4 - 256 | Larger block ⇒ ↑ throughput, but may slightly ↓ accuracy |

Other factors (decoding strategy, GPUs, generation length=256?, threshold=0.9? precision, …) must be fixed (can be fixed by paper recommend).

---

## 3 . Experimental Design

| Stage | Action | Why? |
| --- | --- | --- |
| 3.1 | **Full grid search** – all `(step, block)` pairs. | Provides a ground-truth surface. |
| 3.2 (Optional) | **Replication** – ≥ 3 runs per pair or timed ≥ 30 s. | Capture variance. |

CSV schema:

```

test_name,denoising_step,block_size,accuracy,throughput
```

---

## 4 . Normalisation & Composite Score

```python
acc_min, acc_max = df["accuracy"].agg(["min", "max"])
thr_min, thr_max = df["throughput"].agg(["min", "max"])
df["Acc_norm"] = 100 * (df["accuracy"]   - acc_min) / (acc_max - acc_min)
df["Thr_norm"] = 100 * (df["throughput"] - thr_min) / (thr_max - thr_min)
w_acc, w_thr = 0.5, 0.5        # change to 0.7 / 0.3 etc.df["Score"]  = w_acc * df["Acc_norm"] + w_thr * df["Thr_norm"] or set a argument parser to change ratio manually
```

---

## 5 . Visualisation Cookbook

| ID | Plot Type | Axes | Purpose / Insight |
| --- | --- | --- | --- |
| **L1** | Dual-axis line – vary **step** | X = `step`, left Y = Accuracy, right Y = Throughput (block fixed) | Visualise opposing trends of step. |
| **L2** | Dual-axis line – vary **block** | X = `block`, left Y = Accuracy, right Y = Throughput (step fixed) | Visualise block impact. |
| **H1** | Heatmap – **Accuracy** | X = `step`, Y = `block`, Color = Accuracy | Locate the highest-quality region. |
| **H2** | Heatmap – **Throughput** | same axes | Find fastest configurations. |
| **H3** | Heatmap – **Score** *(parametric)* | same axes | Show best overall trade-off for the *current* weights. |
| **S1** | **Pareto Scatter** | X = Throughput, Y = Accuracy | Reveal efficiency frontier. |

### 5.1 Heatmaps & Lines

*(See existing descriptions.)*

### 5.2 Pareto Scatter (Trade-off Plot) – **S1**

| Aspect | Specification |
| --- | --- |
| **Data points** | Each `(<Throughput>, <Accuracy>)` pair represents one `(step, block)` configuration. |
| **Colour encoding** | Map colour to **Throughput** (e.g. viridis). |
| **Size encoding** | Map marker size to **block_size** so larger blocks are visually prominent. |
| **Hull / Pareto front** | Compute the **upper convex hull** in (Throughput, Accuracy) space. Highlight it with a distinct line or marker style. These points maximise Throughput **without lowering Accuracy** – ideal candidates when you must decide between equal-accuracy options. |
| **Interpretation** | Points on the Pareto front are *non-dominated*. Everything below/left is strictly worse on at least one metric. |