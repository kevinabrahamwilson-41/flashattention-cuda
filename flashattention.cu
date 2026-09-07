#include "flashattention.h"
#include <cfloat>
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <cstdint>
#include <vector>
namespace transformer {
static constexpr int WARP_SIZE_FA = 32;
template <class T> __device__ __forceinline__ T to_elem(float x);
template <> __device__ __forceinline__ half to_elem<half>(float x) {
  return __float2half(x);
}
template <>
__device__ __forceinline__ __nv_bfloat16 to_elem<__nv_bfloat16>(float x) {
  return __float2bfloat16(x);
}
template <class T>
__device__ __forceinline__ void
ptx_mma_m16n8k16(float &d0, float &d1, float &d2, float &d3, uint32_t a0,
                 uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                 uint32_t b1, float c0, float c1, float c2, float c3) {
  if constexpr (::std::is_same<T, __nv_bfloat16>::value) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0),
          "f"(c1), "f"(c2), "f"(c3));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0),
          "f"(c1), "f"(c2), "f"(c3));
  }
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t &r0, uint32_t &r1,
                                            uint32_t &r2, uint32_t &r3,
                                            const void *smem_ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t &r0, uint32_t &r1,
                                                  const void *smem_ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
               : "=r"(r0), "=r"(r1)
               : "r"(addr));
}

// Plain (non-transposed) x2 variant — required for the K (B-operand) load.
__device__ __forceinline__ void ldmatrix_x2(uint32_t &r0, uint32_t &r1,
                                            const void *smem_ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
               : "=r"(r0), "=r"(r1)
               : "r"(addr));
}

__device__ __forceinline__ void cp_async_16(void *smem_dst,
                                            const void *gmem_src, bool pred) {
  uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  int src_size = pred ? 16 : 0;
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_int),
      "l"(gmem_src), "r"(src_size));
}

__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n" ::);
}
template <int N> __device__ __forceinline__ void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// ============================================================================
// Kernel
// ============================================================================
template <class T> __device__ __forceinline__ uint32_t pack2(float a, float b);
template <> __device__ __forceinline__ uint32_t pack2<half>(float a, float b) {
  __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<uint32_t *>(&h);
}
template <>
__device__ __forceinline__ uint32_t pack2<__nv_bfloat16>(float a, float b) {
  __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t *>(&h);
}

template <int BLOCK_M, int BLOCK_N, int D_HEAD, int NUM_WARPS, bool CAUSAL,
          class T>
__global__ void flash_attention_fat_kernel(
    const T *__restrict__ Q, const T *__restrict__ K, const T *__restrict__ V,
    T *__restrict__ O, float *__restrict__ LSE, const int q_seq_len,
    const int kv_seq_len, const int kv_stride, const int num_q_heads, const int num_kv_heads, const float scale) {
  static_assert(BLOCK_M == 16 * NUM_WARPS, "one m16 tile per warp");
  static_assert(BLOCK_N % 16 == 0 && D_HEAD % 16 == 0, "tile granularity");
  static_assert(BLOCK_M <= BLOCK_N, "Q stages through smem_k");

  const int bh_idx = blockIdx.y;
  const int q_start = blockIdx.x * BLOCK_M;
  const int tid = threadIdx.x + threadIdx.y * WARP_SIZE_FA;
  const int q_offset = kv_seq_len - q_seq_len;
  const int warp_id = threadIdx.y;
  const int lane_id = threadIdx.x;
  constexpr int THREADS = WARP_SIZE_FA * NUM_WARPS;
  if (q_start >= q_seq_len)
    return;

  constexpr int SMEM_PAD = 8;
  constexpr int KV_STRIDE = D_HEAD + SMEM_PAD;
  constexpr int QK_N8 = BLOCK_N / 8;     // score n8-tiles per warp (full BN)
  constexpr int TILES_K = D_HEAD / 16;   // k16-tiles for Q*K^T
  constexpr int PV_N8 = D_HEAD / 8;      // output n8-tiles per warp (full D)
  constexpr int TILES_BN = BLOCK_N / 16; // k16-tiles for P*V

  extern __shared__ char smem_raw[];
  T *smem_k = reinterpret_cast<T *>(smem_raw);
  T *smem_v = smem_k + BLOCK_N * KV_STRIDE;

  const int b = bh_idx / num_q_heads;
  const int h_q = bh_idx - b * num_q_heads;
  const int h_kv = h_q / (num_q_heads / num_kv_heads);
  const size_t q_off = static_cast<size_t>(bh_idx) * q_seq_len * D_HEAD;
  // kv_stride is the physical stride (in tokens) between KV heads in memory.
  const size_t kv_off =
      static_cast<size_t>(b * num_kv_heads + h_kv) * kv_stride * D_HEAD;
  const T *Q_head = Q + q_off;
  const T *K_head = K + kv_off;
  const T *V_head = V + kv_off;
  T *O_head = O + q_off;

  const int mi = warp_id;
  const int local_row0 = (lane_id / 4) % 8;
  const int global_row0 = mi * 16 + local_row0;
  const int global_row1 = global_row0 + 8;
  const float scale2 = scale * 1.4426950408889634f;
  uint32_t q_frag[TILES_K][4];
  {
    constexpr int VEC_COLS = D_HEAD / 8;
    for (int idx = tid; idx < BLOCK_M * VEC_COLS; idx += THREADS) {
      int row = idx / VEC_COLS, col = idx % VEC_COLS;
      int g = q_start + row;
      uint4 val = (g < q_seq_len)
                      ? reinterpret_cast<const uint4 *>(Q_head + g * D_HEAD)[col]
                      : make_uint4(0, 0, 0, 0);
      reinterpret_cast<uint4 *>(smem_k + row * KV_STRIDE)[col] = val;
    }
    __syncthreads();
#pragma unroll
    for (int ki = 0; ki < TILES_K; ki++) {
      int row = lane_id % 16;
      int col = (lane_id / 16) * 8;
      ldmatrix_x4(q_frag[ki][0], q_frag[ki][1], q_frag[ki][2], q_frag[ki][3],
                  smem_k + (mi * 16 + row) * KV_STRIDE + ki * 16 + col);
    }
    __syncthreads(); // Q in regs; smem_k free for K
  }

  float o_acc[PV_N8][4] = {{0}};
  float row_max0 = -FLT_MAX, row_max1 = -FLT_MAX;
  float row_sum0 = 0.0f, row_sum1 = 0.0f;

  const int kv_end = kv_seq_len;
  //CAUSAL ? min(q_start + BLOCK_M, seq_len) : seq_len;
  const int num_kv_tiles = (kv_end + BLOCK_N - 1) / BLOCK_N;

  constexpr int VEC_COLS = D_HEAD / 8;
  auto issue_k = [&](int kv_start_i, int kv_count_i) {
#pragma unroll
    for (int idx = tid; idx < BLOCK_N * VEC_COLS; idx += THREADS) {
      int row = idx / VEC_COLS, col = idx % VEC_COLS;
      bool valid = (row < kv_count_i);
      cp_async_16(reinterpret_cast<uint4 *>(smem_k + row * KV_STRIDE) + col,
                  reinterpret_cast<const uint4 *>(
                      K_head + (size_t)(kv_start_i + row) * D_HEAD) +
                      col,
                  valid);
    }
    cp_async_commit_group();
  };
  issue_k(0, min(BLOCK_N, kv_seq_len)); // prologue: K(0)

  for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
    const int kv_start = kv_tile * BLOCK_N;
    const int kv_count = min(BLOCK_N, kv_seq_len - kv_start);

    cp_async_wait_group<0>(); // K(i) landed (only group in flight here)
    __syncthreads();          // K visible to all warps

    // V(i): streams behind QK + softmax.
    {
#pragma unroll
      for (int idx = tid; idx < BLOCK_N * VEC_COLS; idx += THREADS) {
        int row = idx / VEC_COLS, col = idx % VEC_COLS;
        bool valid = (row < kv_count);
        cp_async_16(reinterpret_cast<uint4 *>(smem_v + row * KV_STRIDE) + col,
                    reinterpret_cast<const uint4 *>(
                        V_head + (size_t)(kv_start + row) * D_HEAD) +
                        col,
                    valid);
      }
      cp_async_commit_group();
    }

    // -- Step A: S = Q * K^T, full BN per warp ------------------------------
    float s_acc[QK_N8][4];
#pragma unroll
    for (int ni = 0; ni < QK_N8; ni++) {
      s_acc[ni][0] = s_acc[ni][1] = s_acc[ni][2] = s_acc[ni][3] = 0.0f;
#pragma unroll
      for (int ki = 0; ki < TILES_K; ki++) {
        uint32_t b0, b1;
        int k_row = lane_id % 8;
        int mat = (lane_id / 8) % 2;
        ldmatrix_x2(b0, b1,
                    smem_k + (ni * 8 + k_row) * KV_STRIDE + ki * 16 + mat * 8);
        ptx_mma_m16n8k16<T>(s_acc[ni][0], s_acc[ni][1], s_acc[ni][2],
                            s_acc[ni][3], q_frag[ki][0], q_frag[ki][1],
                            q_frag[ki][2], q_frag[ki][3], b0, b1, s_acc[ni][0],
                            s_acc[ni][1], s_acc[ni][2], s_acc[ni][3]);
      }
      int s_col0 = ni * 8 + (lane_id % 4) * 2;
      int s_col1 = s_col0 + 1;
#pragma unroll
      for (int i = 0; i < 4; i++)
        s_acc[ni][i] *= scale2;
      if (CAUSAL) {
          if (kv_start + s_col0 > q_offset + q_start + global_row0)
              s_acc[ni][0] = -FLT_MAX;
          if (kv_start + s_col1 > q_offset + q_start + global_row0)
              s_acc[ni][1] = -FLT_MAX;
          if (kv_start + s_col0 > q_offset + q_start + global_row1)
              s_acc[ni][2] = -FLT_MAX;
          if (kv_start + s_col1 > q_offset + q_start + global_row1)
              s_acc[ni][3] = -FLT_MAX;
      }
      if (s_col0 >= kv_count) { s_acc[ni][0] = -FLT_MAX; s_acc[ni][2] = -FLT_MAX; }
      if (s_col1 >= kv_count) { s_acc[ni][1] = -FLT_MAX; s_acc[ni][3] = -FLT_MAX; }
    }

    // -- Step B: intra-warp softmax; exp overwrites s_acc (P stays in regs) --
    float pmax0 = -FLT_MAX, pmax1 = -FLT_MAX;
#pragma unroll
    for (int ni = 0; ni < QK_N8; ni++) {
      pmax0 = fmaxf(pmax0, fmaxf(s_acc[ni][0], s_acc[ni][1]));
      pmax1 = fmaxf(pmax1, fmaxf(s_acc[ni][2], s_acc[ni][3]));
    }
#pragma unroll
    for (int d = 1; d < 4; d <<= 1) {
      pmax0 = fmaxf(pmax0, __shfl_xor_sync(0xFFFFFFFFu, pmax0, d));
      pmax1 = fmaxf(pmax1, __shfl_xor_sync(0xFFFFFFFFu, pmax1, d));
    }
    float prev_max0 = row_max0, prev_max1 = row_max1;
    float new_max0 = fmaxf(prev_max0, pmax0);
    float new_max1 = fmaxf(prev_max1, pmax1);

    float psum0 = 0.0f, psum1 = 0.0f;
#pragma unroll
    for (int ni = 0; ni < QK_N8; ni++) {
      float e0 = (s_acc[ni][0] > -FLT_MAX * 0.5f) ? exp2f(s_acc[ni][0] - new_max0) : 0.0f;
      float e1 = (s_acc[ni][1] > -FLT_MAX * 0.5f) ? exp2f(s_acc[ni][1] - new_max0) : 0.0f;
      float e2 = (s_acc[ni][2] > -FLT_MAX * 0.5f) ? exp2f(s_acc[ni][2] - new_max1) : 0.0f;
      float e3 = (s_acc[ni][3] > -FLT_MAX * 0.5f) ? exp2f(s_acc[ni][3] - new_max1) : 0.0f;
      psum0 += e0 + e1;
      psum1 += e2 + e3;
      s_acc[ni][0] = e0; s_acc[ni][1] = e1; s_acc[ni][2] = e2; s_acc[ni][3] = e3;
    }
#pragma unroll
    for (int d = 1; d < 4; d <<= 1) {
      psum0 += __shfl_xor_sync(0xFFFFFFFFu, psum0, d);
      psum1 += __shfl_xor_sync(0xFFFFFFFFu, psum1, d);
    }

    // -- Step C: online correction -------------------------------------------
    float corr0 = (kv_tile == 0) ? 0.0f : exp2f(prev_max0 - new_max0);
    float corr1 = (kv_tile == 0) ? 0.0f : exp2f(prev_max1 - new_max1);
    row_max0 = new_max0;
    row_max1 = new_max1;
    row_sum0 = row_sum0 * corr0 + psum0;
    row_sum1 = row_sum1 * corr1 + psum1;
#pragma unroll
    for (int di = 0; di < PV_N8; di++) {
      o_acc[di][0] *= corr0;
      o_acc[di][1] *= corr0;
      o_acc[di][2] *= corr1;
      o_acc[di][3] *= corr1;
    }

    cp_async_wait_group<0>(); // this thread's V arrived
    __syncthreads();          // all threads' V visible; K(i) dead CTA-wide

    // K(i+1): streams behind PV into the now-dead smem_k.
    if (kv_tile + 1 < num_kv_tiles) {
      int ks = (kv_tile + 1) * BLOCK_N;
      issue_k(ks, min(BLOCK_N, kv_seq_len - ks));
    }
#pragma unroll
    for (int ki = 0; ki < TILES_BN; ki++) {
      uint32_t a0 = pack2<T>(s_acc[2 * ki][0], s_acc[2 * ki][1]);
      uint32_t a1 = pack2<T>(s_acc[2 * ki][2], s_acc[2 * ki][3]);
      uint32_t a2 = pack2<T>(s_acc[2 * ki + 1][0], s_acc[2 * ki + 1][1]);
      uint32_t a3 = pack2<T>(s_acc[2 * ki + 1][2], s_acc[2 * ki + 1][3]);
#pragma unroll
      for (int di = 0; di < PV_N8; di++) {
        uint32_t b0, b1;
        int v_row = lane_id % 8 + ((lane_id / 8) % 2) * 8;
        ldmatrix_x2_trans(b0, b1,
                          smem_v + (ki * 16 + v_row) * KV_STRIDE + di * 8);
        ptx_mma_m16n8k16<T>(o_acc[di][0], o_acc[di][1], o_acc[di][2],
                            o_acc[di][3], a0, a1, a2, a3, b0, b1, o_acc[di][0],
                            o_acc[di][1], o_acc[di][2], o_acc[di][3]);
      }
    }
    __syncthreads(); // smem_v dead before next tile's loads
  }

  // -- Finalize --------------------------------------------------------------
  {
    float inv0 = (row_sum0 > 0.0f) ? (1.0f / row_sum0) : 0.0f;
    float inv1 = (row_sum1 > 0.0f) ? (1.0f / row_sum1) : 0.0f;
#pragma unroll
    for (int di = 0; di < PV_N8; di++) {
      int col0 = di * 8 + (lane_id % 4) * 2;
      int col1 = col0 + 1;
      int gq0 = q_start + global_row0;
      int gq1 = q_start + global_row1;
      if (gq0 < q_seq_len) {
          O_head[gq0 * D_HEAD + col0] =
              to_elem<T>(o_acc[di][0] * inv0);
          O_head[gq0 * D_HEAD + col1] =
              to_elem<T>(o_acc[di][1] * inv0);
      }
      if (gq1 < q_seq_len) {
          O_head[gq1 * D_HEAD + col0] =
              to_elem<T>(o_acc[di][2] * inv1);
          O_head[gq1 * D_HEAD + col1] =
              to_elem<T>(o_acc[di][3] * inv1);
      }
    }
    if (lane_id % 4 == 0 && LSE != nullptr) {
      int gq0 = q_start + global_row0;
      int gq1 = q_start + global_row1;
      // max/sum are in the base-2 domain (see scale2): LSE_e = ln2*(m2+log2 s2)
      constexpr float LN2 = 0.6931471805599453f;
      if (gq0 < q_seq_len)
          LSE[bh_idx * q_seq_len + gq0] =
              LN2 * (row_max0 + log2f(fmaxf(row_sum0, 1e-10f)));
      if (gq1 < q_seq_len)
          LSE[bh_idx * q_seq_len + gq1] =
              LN2 * (row_max1 + log2f(fmaxf(row_sum1, 1e-10f)));
    }
  }
}
#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n",                               \
              __FILE__, __LINE__, cudaGetErrorString(err));                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
namespace {
template <int D_HEAD, class T>
inline void launch_fat_variant(const FlashAttentionParams &params) {
  constexpr int BM = 64, BN = 64, NW = 4;
  constexpr int KV_STRIDE = D_HEAD + 8;

  const int grid_x = (params.q_seq_len + BM - 1) / BM;
  dim3 grid(grid_x, params.batch_size * params.num_heads);
  dim3 block(WARP_SIZE_FA, NW);
  size_t smem_bytes = 2 * (size_t)BN * KV_STRIDE * sizeof(T);

  const int H_q = params.num_heads;
  const int H_kv =
      (params.num_kv_heads > 0) ? params.num_kv_heads : params.num_heads;
  const T *Qp = reinterpret_cast<const T *>(params.Q);
  const T *Kp = reinterpret_cast<const T *>(params.K);
  const T *Vp = reinterpret_cast<const T *>(params.V);
  T *Op = reinterpret_cast<T *>(params.O);
  if (params.causal) {
    flash_attention_fat_kernel<BM, BN, D_HEAD, NW, true, T>
      <<<grid, block, smem_bytes, params.stream>>>(
          Qp,
          Kp,
          Vp,
          Op,
          params.L,
          params.q_seq_len,
          params.kv_seq_len,
          params.kv_stride,
          H_q,
          H_kv,
          params.scale);
  } else {
    flash_attention_fat_kernel<BM, BN, D_HEAD, NW, false, T>
      <<<grid, block, smem_bytes, params.stream>>>(
          Qp,
          Kp,
          Vp,
          Op,
          params.L,
          params.q_seq_len,
          params.kv_seq_len,
          params.kv_stride,
          H_q,
          H_kv,
          params.scale);
  }
  CUDA_CHECK(cudaGetLastError());
}
} // anonymous namespace
void launch_flash_attention(const FlashAttentionParams &params) {
  const bool bf16 = (params.dtype == DType::BF16);

  switch (params.d_head) {
  case 64:
    if (bf16)
      launch_fat_variant<64, __nv_bfloat16>(params);
    else
      launch_fat_variant<64, half>(params);
    break;

  case 128:
    if (bf16)
      launch_fat_variant<128, __nv_bfloat16>(params);
    else
      launch_fat_variant<128, half>(params);
    break;

  default:
    fprintf(stderr,
            "flash_attention: unsupported d_head=%d "
            "(supported: 64, 128)\n",
            params.d_head);
    abort();
  }
}
} // namespace transformer