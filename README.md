# Per-Layer Embeddings on a microcontroller memory hierarchy

Validating one risky assumption before building anything on an ESP32:

> **Can a very thin transformer core exploit per-layer conditioning fed from a
> large, slow, sparsely-read memory tier — or does PLE only work at billions of
> parameters?**

## Why this question

An ESP32-S3 has three memory tiers that happen to be the same *shape* as the
GPU-VRAM / CPU-RAM split that Gemma's Per-Layer Embeddings were designed for,
three orders of magnitude down:

| Tier | Size | Speed | Suits |
|---|---|---|---|
| Internal SRAM | ~512KB | on-die | dense weights, touched every token |
| PSRAM | 8–16MB | ~40–80 MB/s octal SPI | medium working set |
| Flash (memory-mapped XIP) | 16–32MB | random-read friendly | **sparse lookups** |

Every ESP32 language model today streams its whole weight set per token, which
is exactly why they run at 3–5 tok/s. PLE offers a different bargain: keep a
tiny core permanently in SRAM, and per token read only the handful of table rows
that token needs from flash.

If it works, a ~$6 chip could hold ~50M stored parameters with ~1M active ones.
If it doesn't, the whole plan dies here — which is why this is the first thing
built.

**No published evidence exists either way at this scale.** Neither the Gemma 3n
nor the Gemma 4 technical report ablates PLE, and there is no standalone PLE
paper. The only public commentary argues the case for PLE is strongest for
*small-stack, large-memory* models, which is encouraging but not evidence.

## The experiment

Five arms, all trained on identical data with an **identical core parameter
budget** (`ffn_hidden` is auto-solved per arm so each lands within 0.1% of
1.5M core params). Core = dense weights multiplied every token, i.e. what must
sit in SRAM. Table = params that are only ever a per-token row lookup, i.e. what
may sit in flash.

| Arm | Core | Table | What it isolates |
|---|---|---|---|
| `baseline` | 1.50M | 0 | what a tiny LM looks like today |
| `ple` | 1.50M | 1.57M | the real proposal |
| `ple_notable` | 1.50M | 0 | per-layer *plumbing* with no table — is the table doing the work? |
| `fatembed` | 1.50M | 1.57M | same table budget injected only at the bottom — does *per-layer* matter? |
| `bigcore` | 3.00M | 0 | spend the table budget on SRAM instead — the price of a thin core |

`ple_notable` and `fatembed` are the arms that make this a validation rather
than a demo. Without them, a win for `ple` would be unattributable.

### The PLE implementation is faithful to Gemma 4

Reproduced from `transformers/models/gemma4`, minus AltUp (which Gemma 4
dropped anyway). Per-layer input is **not** just a table lookup:

```
ple = RMSNorm( proj(embed) * d_model^-0.5 )          # context-aware half
ple = (ple + table[token] * sqrt(ple_dim)) / sqrt(2) # + token-identity half
```

and it enters each block *after* the FFN residual as a multiplicative gate:

```
h = h + RMSNorm( W_up( gelu(W_gate(h)) * ple ) )
```

The `sqrt(ple_dim)` embed scale is undocumented in Gemma's config and is the
main reproduction gotcha. The gate direction matters: the hidden state is
squeezed to `ple_dim` and gated *by* the PLE vector, so either factor can
suppress the other.

### Fairness controls

- **Matched core params** across arms, solved automatically, not hand-tuned.
- **Identical tokenizer** (4096 BPE) — cross-entropy is meaningless across vocabs.
- **Identical val batches** for every arm (fixed RNG seed on the val sampler).
- **Init parity**: the per-layer branch ends in an RMSNorm that would otherwise
  undo the small residual init and inject unit-scale noise from step 0, so its
  norm gain is zero-initialised and the branch starts as an exact no-op. All
  five arms begin at uniform loss (~8.32 nats).

## Setup

```bash
uv sync
uv run python data/prepare.py      # ~300MB of TinyStories -> 78M train tokens
bash experiments/run_ablation.sh   # 5 arms x 3000 steps (49M tokens each)
uv run python src/analyze.py       # summary table + the deciding comparisons
uv run python src/sample.py --run runs/ple-s0.pt
```

## Status

Validated end to end on an ESP32-S3 N16R8. The deploy model has 28.9M stored
parameters in a 14.91MB group-wise int4 artifact, reproduces its PyTorch golden
logits in the portable C runtime to `1e-5`, and generates coherent TinyStories
text on the chip. The runtime measures 102.9ms per model step (**9.72 tok/s
compute-only**); attached serial runs measure **~9.5 tok/s end to end**.

The current firmware stages the output head as int8 in PSRAM and quantizes
activations to int8 (validated on host val perplexity, delta ~0), keeps the 25M
PLE table memory-mapped in flash, and places scratch plus KV cache in PSRAM. The
head is now PSRAM-bandwidth-bound; further speed comes from reducing bytes-read or
a smaller output head, not from more vectorization. See `firmware/esp32_llm/README.md`
for reproducible build and flash commands.

See `RESULTS.md` for findings.
