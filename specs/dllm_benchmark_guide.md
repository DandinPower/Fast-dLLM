# Benchmark & Visualisation Guide

## 1 · Metrics

| Metric         | Unit              | Goal | Notes                                                                                                           |
| -------------- | ----------------- | ---- | --------------------------------------------------------------------------------------------------------------- |
| **Accuracy**   | ratio (0 – 1) / % | ↑    | Task-specific (e.g. GSM8K exact-match).                                                                         |
| **Throughput** | tokens per seconds     | ↑    | End-to-end tokens per second.                                                                                   |

---

## 2 · Control Parameters

| Parameter              | Typical Values                         | Expected Trend                        |
| ---------------------- | -------------------------------------- | ------------------------------------- |
| **block\_length B**    | 4, 8, 16, 32, 64, 128, 256             | ↑ B ⇒ ↑ throughput, slight ↓ accuracy |
| **denoising\_steps S** | num\_blocks for confidence parallel, but it can maximum set to 256 | ↑ S ⇒ ↑ accuracy, ↓ throughput        |

All other factors (GPU type, gen\_length = 256, sampling threshold = 0.9, precision, etc.) **must stay fixed**.

---

## 3 · Experimental Design

| Stage                     | What you run                                                                           | Purpose                                                        |
| ------------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| **Stage 1 – Block sweep** | Fix one step per block for enabling fully confidence parallel | Measures the pure effect of *parallel width*.  |
| **Stage 2 – Step sweep**  | Lock **B = 32** (best trade-off from Stage 1) and run S ∈ {8, 16, 32, 64, 128, 256}.   | Measures the quality-vs-speed for best denoising step. |

### Runnable configurations

| Stage | B                          | S                       |
| ----- | -------------------------- | ----------------------- |
| 1     | 4, 8, 16, 32, 64, 128, 256 | **num_block**                   |
| 2     | **32**                     | 8, 16, 32, 64, 128, 256 |

### CSV schema

```
run_name,block_length,steps,accuracy,throughput
```

---

## 4 · Visualisation Cookbook

| ID      | Plot                          | Axes                                                           | Insight                                                                            |
| ------- | ----------------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| **L-B** | Dual-axis line: *block sweep* | **X = block\_length (log₂)**<br>Y₁ = Accuracy, Y₂ = Throughput | Shows how widening the parallel block changes speed and quality at fixed steps per block (1). |
| **L-S** | Dual-axis line: *step sweep*  | **X = steps (log₂)**<br>Y₁ = Accuracy, Y₂ = Throughput         | Shows the quality-vs-speed frontier for B = 32.                                    |

Implementation tips:

* Use identical colour/order for Accuracy (left Y-axis) and Throughput (right Y-axis) across both charts so the story is obvious.
* Log-scale the X-axis to keep spacing even (4 → 256).
* Annotate each point with its `(Accuracy %, Throughput)` if the lines overlap.