# FlashAttention-2 CUDA

A custom CUDA implementation of **FlashAttention-2-style exact attention**, designed for efficient execution on NVIDIA GPUs through tiled computation, split-Q warp parallelism, online softmax, Tensor Core matrix multiplication, and asynchronous memory movement.

The implementation targets low-level GPU experimentation rather than providing a general-purpose inference framework. It exposes the core attention computation directly as a CUDA kernel, making it possible to study how tile sizes, warp-level work partitioning, memory staging, and Tensor Core execution affect attention performance.

---

## Overview

Standard scaled dot-product attention computes:

$$
S = QK^T
$$

$$
P = \mathrm{softmax}(S)
$$

$$
O = PV
$$

For a sequence of length \(N\), the intermediate attention matrix has dimensions \(N \times N\). Materializing this matrix in GPU global memory introduces substantial memory traffic and an \(O(N^2)\) intermediate-memory requirement.

This implementation follows the IO-aware approach introduced by FlashAttention and incorporates the improved parallelization strategy of FlashAttention-2.

Instead of materializing the complete attention matrix, the kernel processes attention using blocks of queries, keys, and values. Query rows are partitioned across CUDA warps, while each warp independently performs the attention computation for its assigned query rows across successive key/value tiles.

The implementation therefore combines:

* **IO-aware tiled attention**
* **Split-Q warp parallelism**
* **Online softmax**
* **Tensor Core MMA operations**
* **Register-resident intermediate computation**
* **Asynchronous global-to-shared memory transfers**
* **Double-buffered K/V staging**
* **FP16 and BF16 support**
* **Causal and non-causal attention**
* **Grouped Query Attention (GQA)**

---

## Key Features

### FlashAttention-2-Style Split-Q Parallelism

A major feature of the kernel is its query-side work partitioning.

For the default configuration:

```text
BLOCK_M  = 64
BLOCK_N  = 64
NUM_WARPS = 4
```

the 64 query rows processed by a CUDA thread block are divided across four warps:

```text
             Query Tile
          64 rows × D_HEAD
                 │
       ┌─────────┼─────────┐
       │         │         │
     Warp 0    Warp 1    Warp 2    Warp 3
     rows      rows      rows      rows
      0–15     16–31     32–47     48–63
```

Each warp owns a subset of query rows and processes the complete sequence of K/V tiles for those rows.

This reduces the need for multiple warps to communicate intermediate attention results and follows the work-partitioning principles introduced by FlashAttention-2.

---

### IO-Aware Tiled Attention

The full \(N \times N\) attention matrix is never materialized in global memory.

Instead, computation proceeds through smaller tiles:

```text
Q tile × K tileᵀ
       ↓
   Online Softmax
       ↓
Probability tile × V tile
       ↓
Output accumulation
```

This keeps intermediate attention data on-chip and reduces unnecessary global-memory traffic.

---

### Online Softmax

The kernel computes softmax incrementally as successive K/V tiles are processed.

For each query row, the running maximum and normalization factor are updated using:

$$m_{new} = \max(m_{old}, m_{tile})$$

$$d_{new} = e^{m_{old}-m_{new}} d_{old} + e^{m_{tile}-m_{new}}$$

The accumulated output is rescaled using the same correction factor.

This produces the exact attention result without requiring the complete score or probability matrix to be stored.

---

### Tensor Core MMA

The computationally intensive matrix multiplications are implemented using NVIDIA Tensor Core instructions.

The kernel directly invokes:

```text
mma.sync.aligned.m16n8k16
```

for both:

```text
Q × Kᵀ
```

and:

```text
P × V
```

The current implementation supports FP16 and BF16 inputs with FP32 accumulation.

For BF16:

```text
BF16 × BF16 → FP32
```

For FP16:

```text
FP16 × FP16 → FP32
```

---

### Warp-Level Matrix Loading

The kernel uses NVIDIA's `ldmatrix` instructions to efficiently move matrix fragments from shared memory into registers for Tensor Core operations.

The implementation uses:

```text
ldmatrix.x4
ldmatrix.x2
ldmatrix.x2.trans
```

to load Q, K, and V fragments according to the operand layout required by the MMA instructions.

---

### Asynchronous Memory Movement

K/V tiles are transferred from global memory to shared memory using CUDA asynchronous copy instructions:

```text
cp.async
```

The kernel uses:

```text
cp.async.commit_group
cp.async.wait_group
```

to control asynchronous copy groups.

This allows memory movement to be organized independently from the Tensor Core computation and provides a mechanism for overlapping data staging with attention computation.

---

### Double-Buffered K/V Staging

The K and V shared-memory regions use two stages:

```text
Stage 0
Stage 1
```

Successive K/V tiles alternate between the two stages.

Conceptually:

```text
Iteration i:

Shared Memory
┌───────────────┐
│ K/V tile i    │ ← computation
├───────────────┤
│ K/V tile i+1  │ ← staging
└───────────────┘

Iteration i+1:

┌───────────────┐
│ K/V tile i+1  │ ← computation
├───────────────┤
│ K/V tile i+2  │ ← staging
└───────────────┘
```

This reduces idle periods caused by repeatedly loading K/V tiles from global memory.

---

## Kernel Execution

For every query block, the kernel follows the following execution structure:

```text
                    Query Block
                         │
                         ▼
                Load Q → Shared Memory
                         │
                         ▼
                  Load Q → Registers
                         │
                         ▼
                ┌──────────────────┐
                │  K/V Tile Loop   │
                └──────────────────┘
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
         Q × Kᵀ                  Load next K/V
             │                    asynchronously
             ▼
       Apply scaling
             │
             ▼
       Causal masking
             │
             ▼
       Online softmax
             │
             ▼
          P × V
             │
             ▼
       Output accumulation
             │
             └──────────────┐
                            ▼
                    Next K/V tile
                            │
                            ▼
                    Final normalization
                            │
                            ▼
                         Output
```

The probability values produced by softmax remain in registers and are immediately consumed by the \(P \times V\) computation rather than being written to global memory.

---

## Algorithmic Details

### 1. Query Tiling

A CUDA thread block operates on:

```text
BLOCK_M = 64
```

query rows.

For the default four-warp configuration:

```text
64 / 4 = 16 query rows per warp
```

Each warp therefore maintains its own query fragments, softmax statistics, and output accumulators.

---

### 2. Key/Value Tiling

Keys and values are processed using:

```text
BLOCK_N = 64
```

tokens per tile.

For every K/V tile, the kernel:

1. Loads K asynchronously into shared memory.
2. Computes \(QK^T\).
3. Applies scaling and causal masking.
4. Performs online softmax.
5. Loads V.
6. Computes \(PV\).
7. Updates the running output.
8. Stages the next K tile.

---

### 3. QKᵀ Computation

The query and key fragments are multiplied using Tensor Core MMA instructions.

The kernel uses a matrix-multiply shape of:

```text
m16n8k16
```

with FP32 accumulation.

The query fragments are kept in registers after their initial load, allowing them to be reused across multiple K/V tiles.

---

### 4. Online Softmax

The kernel maintains per-row:

```text
running maximum
running normalization factor
running output accumulator
```

When a new K tile produces a larger maximum, the previous accumulated output is rescaled before the new probability-weighted V contribution is added.

This allows the final output to be mathematically equivalent to conventional softmax attention while avoiding storage of the complete attention matrix.

---

### 5. Register-Resident Probabilities

After the score matrix is computed, the kernel converts the scores to normalized exponential values directly in registers.

The same register fragments are then packed and used as the input to the \(P \times V\) Tensor Core computation.

Thus the intermediate probability matrix does not need to be written to global memory.

---

### 6. Causal Attention

For causal attention, scores corresponding to future key positions are masked:

$$
S_{ij}=-\infty
\qquad\text{for }j>i
$$

The kernel also detects completely non-interacting causal tiles and avoids unnecessary masking work where possible.

---

### 7. Grouped Query Attention

The kernel supports different numbers of query and key/value heads.

For a query head \(h_q\), the corresponding KV head is selected using:

```text
h_kv = h_q / (num_q_heads / num_kv_heads)
```

This supports architectures using Grouped Query Attention, such as Llama-class models.

---

## GPU Memory Hierarchy

The implementation deliberately maps different parts of the computation to different levels of the GPU memory hierarchy.

```text
Global Memory
     │
     │ cp.async
     ▼
Shared Memory
     │
     │ ldmatrix
     ▼
Registers
     │
     │ mma.sync
     ▼
Tensor Cores
```

### Global Memory

Stores:

* Q
* K
* V
* Output
* Optional LSE values

### Shared Memory

Used primarily for:

* K tiles
* V tiles
* Initial Q staging

### Registers

Used for:

* Q fragments
* Score fragments
* Probability fragments
* Online softmax statistics
* Output accumulators

### Tensor Cores

Perform:

```text
Q × Kᵀ
P × V
```

using hardware matrix-multiplication instructions.

---

## Complexity

For sequence length \(N\) and head dimension \(D\):

### Computation

$$
O(N^2D)
$$

The mathematical attention computation remains quadratic in sequence length.

### Intermediate Memory

The complete \(N \times N\) attention matrix is never materialized.

The intermediate memory requirement is therefore approximately:

$$
O(ND)
$$

rather than:

$$
O(N^2)
$$

for the attention matrix.

FlashAttention therefore does **not** change the asymptotic arithmetic complexity of attention. Its advantage comes from reducing memory traffic and improving how the computation maps onto the GPU memory hierarchy and execution units.

---

## Supported Configuration

The current implementation supports:

| Feature                     | Support |
| --------------------------- | ------- |
| FP16                        | ✓       |
| BF16                        | ✓       |
| FP32 accumulation           | ✓       |
| Causal attention            | ✓       |
| Non-causal attention        | ✓       |
| Grouped Query Attention     | ✓       |
| Tensor Core MMA             | ✓       |
| Online softmax              | ✓       |
| `cp.async`                  | ✓       |
| Double-buffered K/V staging | ✓       |
| Head dimension 64           | ✓       |
| Head dimension 128          | ✓       |
| NVIDIA SM 89                | ✓       |

Default kernel configuration:

```text
BLOCK_M   = 64
BLOCK_N   = 64
NUM_WARPS = 4
SMEM_PAD  = 8
```

The kernel is currently optimized around NVIDIA Ada Lovelace-class hardware and can be retargeted to other compatible architectures with appropriate instruction and performance validation.

---

## Repository Structure

```text
flashattentionV2-cuda/
├── flashattention.cu
├── flashattention.h
└── README.md
```

### `flashattention.cu`

Contains the CUDA implementation, including:

* FlashAttention-2-style forward kernel
* Query/Key/Value tiling
* Split-Q warp partitioning
* Online softmax
* Tensor Core MMA
* `ldmatrix`
* `cp.async`
* Double-buffered shared memory
* Causal masking
* GQA handling
* Kernel dispatch

### `flashattention.h`

Contains the public kernel interface, configuration structures, and supporting definitions.

---

## Prerequisites

Recommended environment:

* NVIDIA GPU with Tensor Core support
* CUDA Toolkit 11.8 or newer
* NVCC
* C++17-compatible host compiler
* Linux environment recommended
* 
## Build & Run

```bash
nvcc -O3 -std=c++20 -arch=sm_89 flashattention.cu -o flashattention
nvcc -O3 -std=c++20 -arch=sm_89 stress_test_flashattention.cu flashattention.cu -o stress_test_flashattention
./flashattention
./stress_test_flashattention
```
For a different GPU architecture, change the `-arch` value accordingly.

Examples:

```text
sm_75  → Turing
sm_80  → Ampere
sm_86  → Ampere
sm_89  → Ada Lovelace
sm_90  → Hopper
```

Instruction availability and performance characteristics may differ between architectures.

---

## Execution Model

At the grid level, each CUDA block processes one query tile for one batch/head combination:

```text
Grid X = ceil(Q_sequence_length / BLOCK_M)

Grid Y = batch_size × num_query_heads
```

Conceptually:

```text
                 Query sequence
        ┌───────┬───────┬───────┬───────┐
        │ Q 0   │ Q 64  │ Q 128 │ Q 192 │ ...
        └───────┴───────┴───────┴───────┘
           │       │       │
         CTA 0   CTA 1   CTA 2
```

Each CTA operates independently on its assigned query block and attention head.

Within a CTA, the query rows are distributed across warps using the split-Q execution strategy.

---

## Performance Investigation

This repository is intended not only as an implementation of the attention algorithm but also as a platform for studying GPU kernel behavior.

Relevant parameters for experimentation include:

* `BLOCK_M`
* `BLOCK_N`
* `D_HEAD`
* `NUM_WARPS`
* shared-memory padding
* Tensor Core instruction layout
* K/V staging strategy
* asynchronous copy behavior
* causal versus non-causal execution
* FP16 versus BF16

Performance can be evaluated using NVIDIA profiling tools such as:

```text
Nsight Systems
Nsight Compute
```

Useful measurements include:

* kernel latency
* achieved TFLOP/s
* Tensor Core utilization
* global-memory throughput
* shared-memory throughput
* register usage
* occupancy
* warp stalls
* synchronization overhead
* memory-transfer overlap

---
## Benchmarking

The kernel is benchmarked on the following system:

| Component | Specification |
|---|---|
| CPU | 13th Gen Intel Core i7-13650HX (20 cores) |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU / Max-Q |
| GPU VRAM | 8 GB (8188 MiB) |
| GPU Architecture | NVIDIA Ada Lovelace |
| Compute Capability | 8.9 (sm_89) |
| GPU Power Limit | 55 W |
| System RAM | 16 GB |
| OS | Ubuntu 24.04.4 LTS x86_64 |
| CUDA Toolkit | 13.3 |
| NVIDIA Driver | 595.84 |
| Compiler | NVCC 13.3.73 |
| Desktop | GNOME 46 |
| Shell | Bash 5.2.21 |

### Benchmark configuration

The attention kernel is evaluated using:

- FP16 and BF16 data types
- Batch size: 1
- Head dimension: 64
- Query heads: 32
- KV heads: 8
- Grouped Query Attention (GQA)
- Causal and non-causal execution
- Configurable query/key-value tile sizes
- Configurable warp count
- Tensor Core MMA instructions
- Asynchronous global-to-shared memory transfers

The benchmark measures execution latency and effective throughput across different sequence lengths and kernel configurations.

Performance measurements are intended to characterize the behavior of this specific kernel implementation on a consumer Ada Lovelace GPU rather than to reproduce the official FlashAttention benchmark methodology.

### Benchmark Results

| Sequence Length | Avg (us) | Median (us) | TFLOP/s | Effective BW (GB/s) |
|---:|---:|---:|---:|---:|
| 32 | 8.206 | 8.192 | 1.022 | 40.429 |
| 64 | 9.707 | 9.952 | 3.457 | 68.361 |
| 128 | 17.695 | 17.408 | 7.585 | 74.999 |
| 256 | 39.348 | 38.928 | 13.644 | 67.456 |
| 512 | 103.268 | 103.424 | 20.795 | 51.405 |
| 1024 | 318.227 | 317.632 | 26.993 | 33.362 |
| 2048 | 1108.879 | 1108.992 | 30.986 | 19.149 |

## Design Goals

The implementation is designed around four main goals:

### 1. Reduce Global-Memory Traffic

Avoid materializing the \(N \times N\) attention matrix.

### 2. Increase GPU Parallelism

Partition query work across CUDA warps so that independent query rows can execute concurrently.

### 3. Maximize Matrix-Multiply Throughput

Use Tensor Core MMA instructions for the dominant \(QK^T\) and \(PV\) operations.

### 4. Exploit the GPU Memory Hierarchy

Use asynchronous transfers, shared-memory tiling, register reuse, and double buffering to keep the Tensor Cores supplied with data.

---

## References

This implementation is motivated by the ideas presented in the following work:

**Tri Dao.**  
*FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.*  
2023.

```bibtex
@misc{dao2023flashattention2fasterattentionbetter,
      title={FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning}, 
      author={Tri Dao},
      year={2023},
      eprint={2307.08691},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2307.08691}, 
}
```
---
## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Author

**Kevin Abraham Wilson**  
*Developer*  
**Gmail:** [kevinabrahamwilson8@gmail.com](mailto:kevinabrahamwilson8@gmail.com)
