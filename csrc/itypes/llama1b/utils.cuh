#pragma once

#include "kittens.cuh"

namespace megakittens {
namespace llama1b {

// <NT> RMS Norm - 版本1: 激活值在寄存器中
//      y = x * (rms_scale / sqrt(variance(x) + eps))
//      其中 rms_scale 是预计算的缩放因子 (来自 weight 的统计)
//      使用 warp 间规约求和, 8 个 warp 并行处理 N 个元素
//
// @param activations_vec: 输入激活向量 (N / NUM_CONSUMER_WARPS 个 bf16 元素, 在寄存器)
// @param rms_scale_smem:  预计算的 rms 缩放因子 (在共享内存)
// @param rms_norm_eps:    epsilon, 防止除零
// @param scratch_memory: 临时存储, 用于 warp 间规约 (每个 warp 贡献一个 partial_sum)

// activations in registers
template <typename Config, int N>
__device__ static inline auto rms_norm(kittens::rv_fl<N / Config::NUM_CONSUMER_WARPS> activations_vec,
         const kittens::sv_bf<N / Config::NUM_CONSUMER_WARPS> &rms_scale_smem,
         float rms_norm_eps, float *scratch_memory) {
    // <NT> 每个 warp 处理 ELEMS_PER_WARP 个元素
    constexpr int ELEMS_PER_WARP = N / Config::NUM_CONSUMER_WARPS;
    using rv_t = kittens::rv_fl<ELEMS_PER_WARP>;
    rv_t sq_activations_vec, rms_scale_vec;
    // <NT> Step 1: 计算 x^2 (平方)
    kittens::warp::copy(sq_activations_vec, activations_vec);
    kittens::warp::mul(sq_activations_vec, sq_activations_vec, sq_activations_vec);
    // <NT> Step 2: warp 内求和
    float partial_sum = kittens::warp::sum(sq_activations_vec);
    // <NT> Step 3: 只有 warp leader 写入 scratch (每个 warp 贡献一个 partial_sum)
    if (kittens::warp::elect_leader()) {
        scratch_memory[kittens::warpid()] = partial_sum;
    }
    // <NT> Step 4: 所有 warp 同步, 确保所有 partial_sum 都写入了 scratch
    kittens::group<Config::NUM_CONSUMER_WARPS>::sync(1);

    // <NT> Step 5: warp leader 收集所有 warp 的 partial_sum, 计算完整方差
    float full_sum = 0.f;
    #pragma unroll
    for (int i = 0; i < Config::NUM_CONSUMER_WARPS; i++) {
        full_sum += scratch_memory[i];
    }

    // <NT> Step 6: 计算方差和 rms_scale
    //      variance = sum(x^2) / N
    //      rms_scale = 1 / sqrt(variance + eps)
    float variance = full_sum / static_cast<float>(N);
    float rms_scale = rsqrtf(variance + rms_norm_eps);

    kittens::warp::load(rms_scale_vec, rms_scale_smem);
    kittens::warp::mul(rms_scale_vec, rms_scale_vec, rms_scale);
    kittens::warp::mul(activations_vec, activations_vec, rms_scale_vec);

    return activations_vec;
}

// <NT> RMS Norm - 版本2: 激活值在共享内存中
//      与版本1相同, 但多了从 smem 加载到寄存器的步骤
// activations in smem
template <typename Config, int N>
__device__ static inline auto rms_norm(const kittens::sv_bf<N / Config::NUM_CONSUMER_WARPS> &rms_scale_smem,
         const kittens::sv_bf<N / Config::NUM_CONSUMER_WARPS> &activations_smem,
         float rms_norm_eps, float *scratch_memory) {
    constexpr int ELEMS_PER_WARP = N / Config::NUM_CONSUMER_WARPS;
    using rv_t = kittens::rv_fl<ELEMS_PER_WARP>;
    rv_t activations_vec;

    kittens::warp::load(activations_vec, activations_smem);
    return rms_norm<Config, N>(activations_vec, rms_scale_smem, rms_norm_eps, scratch_memory);
}

#ifdef KITTENS_BLACKWELL
// <NT> matvec - 矩阵×向量乘法 (Blackwell 专用版本)
//      使用 MMA tensor core 指令加速
//      out_smem = weights_smem × activations
//
// @param out_smem: 输出向量 (在共享内存-sv_fl-smem vector float)
// @param weights_smem: 权重矩阵 (在共享内存)
// @param activations: 输入激活向量 (在寄存器)
template <kittens::ducks::st::all st_t>
__device__ static inline void matvec(kittens::sv_fl<st_t::rows> &out_smem,
                                     st_t &weights_smem,
                                     kittens::rv_fl<st_t::cols> &activations) {
    // <NT> rt_t是register tile 类型 (2D)
    // rrv_t是rt_t 的一行，rcv_t 是rt_t 的一行，这样的1D向量，方便做规约和广播
    // rv_t是独立向量，都是寄存器数据。
    using rt_t  = kittens::rt_bf<st_t::rows, st_t::cols>;
    using rrv_t = typename rt_t::row_vec;

    // <NT> 激活值拷贝到寄存器
    rrv_t row_activations;
    kittens::warp::copy(row_activations, activations);

    rt_t broadcast_activations, weights;
    // <NT> 将寄存器的activations按列广播成2D，将smem的2D权重加载到寄存器，并做mma。
    // 所以这里全程不涉及tma操作。
    kittens::warp::broadcast_col(broadcast_activations, row_activations);
    kittens::warp::load(weights, weights_smem);

    kittens::rt_fl<16, 16> out_activations;
    kittens::warp::zero(out_activations);
    kittens::warp::mma_ABt(out_activations, weights, broadcast_activations, out_activations);

    // <NT> 每个 lane 写 2 个元素到 out_smem (每个线程写 2 个元素: lane 0,8 写 row0, row1)
    if (kittens::laneid() % 4 == 0) {
        int row0 = kittens::laneid() / 4;
        int row1 = row0 + 8;
        out_smem[row0] = out_activations.tiles[0][0].data[0].x;
        out_smem[row1] = out_activations.tiles[0][0].data[1].x;
    }
    kittens::warp::sync();
}
#else
template <kittens::ducks::st::all st_t>
__device__ static inline void matvec(kittens::sv_fl<st_t::rows> &out_smem,
                                     st_t &weights_smem,
                                     kittens::rv_fl<st_t::cols> &activations) {
    using rt_t  = kittens::rt_fl<st_t::rows, st_t::cols>;
    using rrv_t = typename rt_t::row_vec;
    using rcv_t = typename rt_t::col_vec;
    using rv_t  = kittens::rv_fl<st_t::rows>;

    rrv_t row_activations;
    kittens::warp::copy(row_activations, activations);

    rt_t broadcast_activations, weights;
    kittens::warp::broadcast_col(broadcast_activations, row_activations);
    kittens::warp::load(weights, weights_smem);
    kittens::warp::mul(broadcast_activations, broadcast_activations, weights);

    rcv_t sum_col_vec;
    kittens::warp::row_sum(sum_col_vec, broadcast_activations);

    rv_t sum_vec;
    kittens::warp::copy(sum_vec, sum_col_vec);

    if (kittens::laneid() < st_t::rows) {
        out_smem[kittens::laneid()] = sum_vec[0][0];
    }
    kittens::warp::sync();
}
#endif

// <NT> matvec_reduce - 多 warp 输出的规约累加
//      把 NUM_CONSUMER_WARPS 个 warp 的结果累加到 sum_vec
//      用于 Attention 等需要跨 warp 合并结果的场景
//
// @param scratch: scratch 内存基址 (每个 warp 的输出在 SCRATCH_BYTES_PER_WARP 偏移处)
// @param sum_vec: 输出累加向量 (in/out 参数, 初始值为 0)
template <typename Config, int SCRATCH_BYTES_PER_WARP>
__device__ static inline void matvec_reduce(uint8_t *scratch, kittens::rv_fl<16> &sum_vec) {
    using sv_t = kittens::sv_fl<16>;
    kittens::rv_fl<16> part_vec;
    kittens::warp::zero(sum_vec);

    #pragma unroll
    for (int i = 0; i < Config::NUM_CONSUMER_WARPS; i++) {
        sv_t &part = *reinterpret_cast<sv_t *>(scratch + i * SCRATCH_BYTES_PER_WARP);
        kittens::warp::load(part_vec, part);
        kittens::warp::add(sum_vec, sum_vec, part_vec);
    }
}

} // namespace llama1b
} // namespace megakittens
