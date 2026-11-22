---
license: apache-2.0
base_model: thinkingmachines/Inkling-2-Mini
tags:
- gguf
- llama.cpp
- moe
- 12gb
---

# Inkling 2 Mini, mixB GGUF for 12 GB GPUs

This is a mixed 2-3 bit quant of [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini). It fits on a 12 GB GPU with 64K context. On an RTX 4070 Ti, it decodes at 51-64 tok/s and processes prompts at about 1500 tok/s.

**This file needs the fork [hridaya423/llama.cpp](https://github.com/hridaya423/llama.cpp), branch `inkling-fast`.** Upstream llama.cpp cannot load it, because the expert down tensors are zero-padded from 1152 to 1280 columns. Setup guide: [hridaya423/inkling-2-mini-12gb](https://github.com/hridaya423/inkling-2-mini-12gb).

## Files

| File | Size | Use |
|---|---|---|
| `Inkling-2-Mini-mixB.gguf` | 10.3 GiB | The model |
| `mmproj-Inkling-2-Mini-BF16.gguf` | 0.6 GiB | Vision projector, a copy from [ggml-org/Inkling-2-Mini-GGUF](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF) |

## Quality

The reference is Q8_0, on 50 chunks of 512 tokens of wikitext-2.

| Model | KLD | Same top token | PPL ratio |
|---|---|---|---|
| mixB | 0.048 | 90.9% | 1.034 |
| ggml-org IQ4_NL (16.4 GiB, does not fit) | 0.009 | 95.7% | 1.002 |

## Expert types

- Layers 6-17 (the loop core, run three times per token): IQ3_XXS.
- Layers 18-21: IQ3_XXS down, IQ2_S gate and up.
- Layers 22-23: IQ4_NL. Run these on the CPU with `-ot`.
- All other layers: IQ2_S.

The non-expert tensors are Q6_K, the output is Q6_K and the token embedding is Q8_0. The quant uses the ggml-org imatrix.

## Run

```sh
llama-server -m Inkling-2-Mini-mixB.gguf \
  --mmproj mmproj-Inkling-2-Mini-BF16.gguf --no-mmproj-offload \
  -ngl 99 -fa on --jinja -c 65536 -b 512 -ub 512 -ctk q8_0 -ctv q8_0 -t 8 -np 1 \
  -ot "blk\.(22|23)\.ffn_.*_exps=CPU"
```

This uses 11.6 GB of VRAM, including the Windows desktop.
