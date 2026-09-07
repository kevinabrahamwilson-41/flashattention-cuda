#pragma once
// ============================================================================
// Flash Attention — Standalone Header
//
// Hand-written PTX flash attention kernel for consumer NVIDIA GPUs.
// Uses mma.sync.aligned.m16n8k16 with in-register softmax.
//
// Usage:
//   FlashAttentionParams params = {};
//   params.Q = d_Q;  params.K = d_K;  params.V = d_V;  params.O = d_O;
//   params.batch_size = B;  params.num_heads = H;
//   params.q_seq_len = Q_len;
//   params.kv_seq_len = KV_len;
//   params.scale = 1.0f / sqrtf(64.0f);
//   params.causal = true;
//   params.stream = 0;
//   transformer::launch_flash_attention(params);
//
// Constraints:
//   - d_head must be 64 or 128 (tuned paths; must be a multiple of 16)
//   - Q, K, V, O are [batch_size * num_heads, seq_len, d_head] in FP16
//   - L (optional) is [batch_size * num_heads, seq_len] in FP32
//   - Minimum compute capability: sm_80 (Ampere)
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
// ============================================================================
// Error checking macro
// ============================================================================
#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
#endif

namespace transformer {

// Input/output element type. FP16==0 so a zero-initialized params struct ({})
// defaults to FP16 (back-compat). BF16 for Llama-3/Mistral/Qwen/GLM (bf16
// weights).
enum class DType { FP16 = 0, BF16 = 1 };

// ============================================================================
// Launch Parameters
// ============================================================================
struct FlashAttentionParams {
  const void *Q;
  const void *K;
  const void *V;
  void *O;
  float *L;      // [B*H, S]    log-sum-exp (FP32, optional — can be nullptr)
  int batch_size;
  int num_heads;    // query heads (H_q)
  int num_kv_heads; // KV heads for GQA/MQA; 0 (or == num_heads) means MHA.
                    // K/V are [B, num_kv_heads, S, D]; num_heads % num_kv_heads
                    // == 0.
  int q_seq_len;     // Q sequence length
  int kv_seq_len;    // KV cache active sequence length (number of valid tokens)
  int kv_stride;     // physical stride (in tokens) between KV heads in memory (e.g., max_seq_len)
  int d_head;  // 64 or 128
  float scale; // Typically 1.0f / sqrtf(d_head)
  bool causal; // true = causal mask (upper triangle masked)
  DType dtype; // FP16 (default) or BF16. Q/K/V/O carry that type;
               // the pointers are typed half* as address carriers.
  cudaStream_t stream;
};
// Implemented in kernels/flash_attention.cu
void launch_flash_attention(const FlashAttentionParams &params);
}