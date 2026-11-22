<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset=".github/assets/banner-dark.svg">
    <img alt="inkling-2-mini 12gb: 51-64 tok/s on one 12 GB GPU, 64K context" src=".github/assets/banner-light.svg" width="100%">
  </picture>
</p>

<h3 align="center">Run Inkling 2 Mini at 51–64 tok/s with 64K context on a single 12 GB GPU</h3>

<p align="center">
  <strong>~2× faster decode</strong> · <strong>~17× faster prefill</strong> · <strong>10.3 GiB GGUF</strong> · <strong>90.9% top-1 match vs Q8_0</strong>
</p>

<p align="center">
  <a href="https://huggingface.co/hridya423/Inkling-2-Mini-12GB-GGUF"><strong>Hugging Face Model</strong></a> ·
  <a href="https://github.com/hridaya423/llama.cpp/tree/inkling-fast"><strong>llama.cpp Fork</strong></a> ·
  <a href="https://github.com/hridaya423/llama.cpp/releases/tag/inkling-fast-4d46c81b8"><strong>Windows Release</strong></a> ·
  <a href="https://github.com/hridaya423/llama.cpp/blob/inkling-fast/docs/inkling2-fast.md"><strong>Technical Notes</strong></a>
</p>

---

[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) doesn't fit on a 12 GB GPU at usable quality with upstream `llama.cpp`—you either spill experts to the CPU and crawl at ~30 tok/s (88 tok/s prefill), or run out of VRAM trying to allocate a 64K context window.

This project pairs a custom **`inkling-fast`** `llama.cpp` fork with a loop-aware mixed 2–3 bit quant (**`mixB`**) to fit the model + 64K context into **11.6 GB of VRAM** (including the Windows desktop), doubling decode speed and boosting prompt processing by ~17× while staying within **4.8 percentage points of `IQ4_NL`'s top-1 token agreement** against `Q8_0`.

## Benchmarks

Measured on an **RTX 4070 Ti 12 GB** (i9-13900K, Windows, CUDA 12.4) at **64K context** (`-c 65536`).

| Setup | Model Size | Decode (short) | Decode (22K ctx) | Prefill (22K) | Top-1 vs Q8_0 | KLD | PPL Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Upstream `llama.cpp` + `ggml-org` `IQ4_NL` (CPU offload) | 16.4 GiB | 29–34 tok/s | 29–34 tok/s | 88 tok/s | 95.7% | 0.009 | 1.002 |
| **`inkling-fast` + `mixB` GGUF** | **10.3 GiB** | **61–64 tok/s** | **51–53 tok/s** | **~1,500 tok/s** | **90.9%** | **0.048** | **1.034** |
| **Delta** | **−37%** | **~2.0×** | **~1.7×** | **~17×** | **−4.8 pts** | — | **+3.2%** |

> Quality metrics are from `llama-perplexity --kl-divergence` against `Q8_0` across 50 chunks × 512 tokens of `wikitext-2`. Speed metrics are from `llama-server` on Windows (CUDA 12.4).
>
> On `llama-bench` (`pp512`, `tg128`, 6 runs), the graph and CUDA optimizations alone lift the exact same `mixB` GGUF from **54.1–54.5 → 63.5–63.8 tok/s** decode and **1,673–1,686 → 2,183–2,191 tok/s** prefill.

## Requirements

- **GPU:** NVIDIA GPU with **12 GB VRAM** (close other GPU-heavy apps if driving your desktop on the same card)
- **Disk:** ~12 GB free space (`10.3 GiB` model + `0.6 GiB` vision projector)

## Quick Start

> [!IMPORTANT]
> Upstream `llama.cpp` **cannot** load `mixB` because its expert down-projection tensors are zero-padded from 1152 → 1280 columns to unlock IQ2/IQ3 block quantization. Use the `inkling-fast` build below.

### 1. Get `llama.cpp` (`inkling-fast`)

**Windows (RTX 40 series / `sm_89`):**
Download both zip files from the [latest release](https://github.com/hridaya423/llama.cpp/releases/tag/inkling-fast-4d46c81b8) and extract them into the same folder.

**Linux or other NVIDIA GPUs (build from source):**

```sh
git clone -b inkling-fast https://github.com/hridaya423/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

### 2. Download the Model

```sh
hf download hridya423/Inkling-2-Mini-12GB-GGUF --local-dir Inkling-2-Mini-12GB
```

This pulls `Inkling-2-Mini-mixB.gguf` (10.3 GiB) and `mmproj-Inkling-2-Mini-BF16.gguf` (0.6 GiB).

### 3. Run the Server

```sh
llama-server -m Inkling-2-Mini-12GB/Inkling-2-Mini-mixB.gguf \
  --mmproj Inkling-2-Mini-12GB/mmproj-Inkling-2-Mini-BF16.gguf --no-mmproj-offload \
  -ngl 99 -fa on --jinja -c 65536 -b 512 -ub 512 -ctk q8_0 -ctv q8_0 -t 8 -np 1 \
  -ot "blk\.(22|23)\.ffn_.*_exps=CPU"
```

This starts an OpenAI-compatible server on `http://127.0.0.1:8080` with chat, tool calling, vision input, and prompt caching enabled—using **11.6 GB VRAM** total.

### Flag Breakdown

| Flag | Why it's needed |
|---|---|
| `-ot "blk\.(22\|23)\.ffn_.*_exps=CPU"` | Keeps the `IQ4_NL` experts of the last two layers on CPU so everything else stays resident in VRAM. |
| `-c 65536 -ub 512` | Largest context + micro-batch that fits cleanly in 12 GB (`-ub 1024` at 64K overflows VRAM). |
| `-ctk q8_0 -ctv q8_0` | Quantizes the KV cache; an `f16` KV cache won't fit at 64K. |
| `--no-mmproj-offload` | Keeps the 0.6 GiB vision projector off the GPU. |

> [!WARNING]
> On Windows, approaching 12 GB VRAM causes the NVIDIA driver to silently spill allocations into system RAM without throwing an OOM error, tanking both decode and prefill throughput. Always verify VRAM usage with `nvidia-smi` after changing `-c` or `-ub`.

## How It Works

Getting Inkling 2 Mini fast on 12 GB required solving three distinct bottlenecks across quantization, attention memory, and kernel launch overhead:

### 1. Expert Tensor Padding + Loop-Aware Quantization (`mixB`)
Inkling 2 Mini's expert down-projection rows have width `1152`, which isn't a multiple of the `256`-element super-block required by `IQ2`/`IQ3`. In upstream `llama-quantize`, every `IQ2` and `IQ3` expert silently falls back to `IQ4_NL`.

- **`llama-quantize --pad-exps`** zero-pads expert down rows from `1152 → 1280` (with matching activation padding in the compute graph), unlocking `IQ2`/`IQ3` quants.
- **Asymmetric bit allocation (`mixB`)** invests extra precision where it matters most—Inkling 2's **loop core (layers 6–17)** executes **3× per token**:

| Layers | Expert Quant | Device | Notes |
|---|---|---|---|
| `0–5` | `IQ2_S` | GPU | Entry layers |
| `6–17` | `IQ3_XXS` | GPU | Loop core (runs 3× per token) |
| `18–21` | `IQ3_XXS` (down), `IQ2_S` (gate/up) | GPU | Post-loop layers |
| `22–23` | `IQ4_NL` | CPU | Offloaded via `-ot` |
| Non-experts & output | `Q6_K` (`Q8_0` embeddings) | GPU | Preserves routing & logit accuracy |

### 2. Fused Relative-Position Bias in CUDA Flash Attention
Upstream `llama.cpp` materializes a dense per-head relative-position bias mask, consuming **~6.5 GB** of compute buffer at 64K context. Fusing the relative-position bias directly inside the CUDA Flash Attention kernel cuts the 64K compute buffer to **~1 GB**, unlocking `-ub 512` and pushing 22K prefill from **88 → ~1,500 tok/s**.

### 3. Custom Ops, CUDA Fusions & Graph Caching
On a 12 GB card, Inkling 2 Mini decode is **kernel-launch-bound**, not memory-bandwidth-bound (all-`IQ2_XXS` and all-`IQ2_S` decode at the exact same speed). The `inkling-fast` fork cuts launches per token from **~5,000 → ~3,500**:

- Added `GGML_OP_DEPTH_MIX` and `GGML_OP_TANH_GATE_MIX` (CPU + CUDA) for block attention residuals and loop controllers.
- Fused `ssm_conv` with residual add + scale, and fused top-k MoE weight normalization with epsilon.
- Cached static graph structures (normalized block stacks, controller inputs, zero sinks, scaled bias table) once per graph.

<details>
<summary><strong>Build <code>mixB</code> yourself from BF16</strong></summary>

1. Download the source BF16 GGUF, imatrix, and vision projector from [ggml-org/Inkling-2-Mini-GGUF](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF):

```sh
hf download ggml-org/Inkling-2-Mini-GGUF \
  Inkling-2-Mini-BF16.gguf imatrix.gguf mmproj-Inkling-2-Mini-BF16.gguf \
  --local-dir src
```

2. Quantize with `inkling-fast`'s `llama-quantize` (keep `--tensor-type` patterns ordered most-specific first):

```sh
LOOPS='6|7|8|9|10|11|12|13|14|15|16|17'
./build/bin/llama-quantize --pad-exps --imatrix src/imatrix.gguf \
  --token-embedding-type q8_0 --output-tensor-type q6_k \
  --tensor-type "blk\.(22|23)\.ffn_.*_exps=iq4_nl" \
  --tensor-type "blk\.($LOOPS)\.ffn_.*_exps=iq3_xxs" \
  --tensor-type "blk\.(18|19|20|21)\.ffn_down_exps=iq3_xxs" \
  --tensor-type "ffn_.*_exps=iq2_s" \
  src/Inkling-2-Mini-BF16.gguf Inkling-2-Mini-mixB.gguf Q6_K
```

</details>

<details>
<summary><strong>Engineering notes &amp; what didn't help</strong></summary>

- **CPU offload cost:** Each expert layer left on CPU costs ~5 tok/s. Higher-bit mixes with more CPU layers reach 91.8% top-1, but drop to 45–48 tok/s at 22K context.
- **Upstream op-offload bug:** Offloading loop-core experts (layers 6–17) to CPU triggers NaNs unless `--no-op-offload` is passed.
- **Dead ends on 12 GB:** DSpark speculative decoding (`k=2` ties baseline, other `k` are slower), `f16` KV cache (+0.6 GB at 32K), and `--moe-cache-mib` (12–15 tok/s).
- **Full write-up:** See [docs/inkling2-fast.md](https://github.com/hridaya423/llama.cpp/blob/inkling-fast/docs/inkling2-fast.md) for the complete technical notes and `test-backend-ops` coverage.

</details>
