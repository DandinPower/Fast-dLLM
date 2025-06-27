# **LLaDA Decoding Strategy Guide (Refactored)**

## 1  Quick Overview

LLaDA decodes by **reverse diffusion**: the model starts from a fully-masked sequence and iteratively unmasks tokens.  By tuning *steps* and *block\_length* you sweep from fully sequential to highly parallel generation, trading quality for speed.

**Key specs**

| Item           | Value                       |
| -------------- | --------------------------- |
| Context length | 1 024 tokens                |
| Mask token ID  | `126336`                    |
| Base model     | `GSAI-ML/LLaDA-8B-Instruct` |

---

## 2  Decoding Hyper-parameters

| Parameter                    | Purpose                                           | Typical Range             |
| ---------------------------- | ------------------------------------------------- | ------------------------- |
| **gen\_length**              | Output length (tokens)                            | 32 – 512                  |
| **block\_length**            | Tokens per parallel block (≤ *gen\_length*), enable Semi-autoregressive set `block_length < gen_length` to mix serial inter-block and parallel intra-block decoding.      | 16, 32, 64                |
| **steps**                    | Diffusion steps; lower = faster                   | `num_blocks - gen_length` |
| **threshold**                | Confidence cut-off in parallel mode               | 0 - 1                     |
| **use\_cache / dual\_cache** | Reuse KV states; dual cache speeds suffix tokens  | on / off                  |
| **temperature**              | Softmax temperature                               | 0 - 1                     |
| **remasking**                | Optional re-mask after low-confidence predictions | random / low-conf         |

---

## 3  Decoding Modes

> Let `gen_length = 256`, `block_length = 32`, so `num_blocks = 8`.

| Mode                      | steps | Unmask pattern                         | Typical speed-up |
| ------------------------- | ----- | -------------------------------------- | ---------------- |
| **Sequential (baseline)** | 256   | 1 token/step                           | 1×               |
| **Prefix cache**          | 256   | 1 token/step, KV reused                | 1.5–2×           |
| **Parallel**              | 8     | 32 tokens/step filtered by *threshold* | 2–5×             |
| **Dual-cache + Parallel** | 8     | As above, plus cache                   | up to 11×        |

---

## 4  Tuning Guidelines

### 4.1  Speed

1. Set `steps = num_blocks` (one pass per block).
2. Turn on `use_cache` and `dual_cache`.
3. Lower `threshold` (0.8 – 0.9).
4. Prefer `block_length = 32` (empirically fastest).

### 4.2  Quality

1. Use `steps = gen_length`.
2. Raise or disable `threshold`.
3. Consider `block_length = 16` for stronger left-to-right dependence.

---

## 5  Confidence-Aware Parallel Decoding

In parallel decoding mode (`threshold ≠ None`), **confidence threshold overrides step-based token counts**, acting as a safeguard for output quality.

### Priority Logic

1. **Step-based selection**
   Selects top-*k* tokens per block using confidence:

   ```python
   _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
   transfer_index[j, select_index] = True
   ```

   * Respects the `steps` parameter.
   * Guarantees fixed token count per block (in non-parallel mode).

2. **Threshold override (if enabled)**
   Filters low-confidence predictions:

   ```python
   if threshold is not None:
       for idx in select_index:
           if confidence[j, idx] < threshold:
               transfer_index[j, idx] = False
   ```

   * Can reduce unmasked tokens below the step-defined count.
   * Prioritizes reliability over quantity.

### Behavioral Differences

| Mode                | Token Count      | Based on   |
| ------------------- | ---------------- | ---------- |
| Non-parallel        | Fixed            | `steps`    |
| Parallel (thr≠None) | Variable (≤ max) | Confidence |

### Example

Given `threshold = 0.9`, `num_transfer_tokens = 4`, and confidence scores `[0.95, 0.87, 0.82, 0.75]`:

* All 4 selected by step.
* Only 1 survives threshold filter → 1 token unmasked.

### Why Threshold Takes Priority

* **Safety**: Prevents committing to uncertain tokens.
* **Adaptivity**: Skips unmasking if confidence is low.
* **Control**: Maintains generation quality even under aggressive settings.

> The threshold acts as a **dynamic brake**—you may request fast decoding via fewer steps, but the model slows down if it's unsure.

> Note: Actual **NFE (forward passes)** may exceed `steps × num_blocks` due to threshold filtering and re-tries.