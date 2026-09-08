# FlashAttention CUDA

A custom, efficient CUDA implementation of the **FlashAttention** algorithm. This repository features standalone CUDA kernels designed to optimize the standard attention mechanism by leveraging tiling, online softmax, and high-speed GPU shared memory to minimize global memory traffic.

## 🚀 Overview

Standard Attention ($O(N^2)$ memory) quickly bottlenecks on long sequences due to reading and writing the intermediate attention matrix ($N \times N$) to global memory (HBM). 

This project implements **FlashAttention** as a native CUDA kernel. By computing attention in localized blocks (tiles) and utilizing on-chip shared memory, the implementation completely avoids materializing the massive $N \times N$ matrix in global memory, resulting in significant memory savings and execution speedups.

### Key Features
* **Fused CUDA Kernel:** Combined forward pass computations within custom CUDA grids.
* **Shared Memory Tiling:** Blocks of Query ($Q$), Key ($K$), and Value ($V$) metrics are loaded into fast shared memory tiles.
* **Online Softmax:** Tracks running row-maxima and normalization statistics dynamically for numerical stability without a global sync.
* **Minimal Global Memory Traffic:** Keeps memory reads/writes bounded by the sequence length rather than its square.

---

## 📂 Repository Structure

* `flashattention.cu`: Core CUDA source code housing the kernel configurations, shared memory tiling configurations, and thread-level block computation logic.
* `flashattention.h`: Header file declaring definitions, structures, interfaces, and helper routines.
* `README.md`: Project documentation and setup guide.

---

## 🛠️ Prerequisites

To build and run this repository, ensure your system has the following components installed:

* **NVIDIA GPU** (Ampere architecture or newer recommended for optimal shared memory performance)
* **CUDA Toolkit** (v11.8 or higher recommended)
* **NVCC Compiler** (typically bundled with the CUDA Toolkit)
* A C++17 compliant host compiler (e.g., `g++` or `clang`)

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com
cd flashattention-cuda
```

### 2. Compile the CUDA Module
You can compile the implementation using `nvcc`. Below is a basic example to compile into a static library or object file:

```bash
nvcc -O3 -std=c++17 -arch=sm_89 -c flashattention.cu -o flashattention.o
```
*(Note: Change `-arch=sm_89` to match your specific GPU architecture code, e.g., `sm_75` for Turing, `sm_89` for Ada Lovelace, or `sm_90` for Hopper).*

---

## 💡 Algorithmic Details

The kernel divides the attention calculation into manageable chunks:
1. **Blocks & Grids:** Iterates over sequence blocks using outer loops for keys/values and inner loops for queries.
2. **Online Reduction:** Instead of computing a standard softmax across the entire sequence matrix dynamically, it computes:
   $$m_{\text{new}} = \max(m_{\text{old}}, \tilde{m})$$
   $$d_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} \cdot d_{\text{old}} + e^{\tilde{m} - m_{\text{new}}}$$
3. **Rescaling:** Scaled values are accumulated into the shared memory blocks before writing out the final matrix slice directly to HBM.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

* Tri Dao et al. for the original paper: [*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*](https://arxiv.org/abs/2205.14135).

