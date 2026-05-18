#pragma once
// <NT> rms_qkv_rope_append.cuh - RMS Normalization + QKV MatVec + RoPE 融合算子
//
// 功能: 对 hidden_states 做 RMS Norm, 然后计算 QKV 矩阵乘, 最后应用 RoPE 位置编码
// 融合方式: RMS + MatVec 沿用 matvec_pipeline 的融合逻辑，
//          rope 代码由pipeline_specifics 插入到 rms_matvec_pipeline中，补充loader和store操作。
//          rope的计算在 store 中完成: matvec规约 + RoPE + 写入。
//          整体融合以matvec为中心，rms_norm用重复计算来和matvec的block输入对应，rope用重复加载来和matvec的block输出对应。
//          达到整体无跨block操作的目的。
//          
// 数据分块:
//   QKV: [Q | K | V] = [2048 | 512 | 512] = 3072 (按行分块)
//   每个 tile 处理 BLOCK_SIZE=16 行的输出
//   总共 192 个 tiles: 128 Q tiles + 32 K tiles + 32 V tiles
//
// RoPE: 对 Q 和 K 应用旋转位置编码 (只对 Q 和 K 做, V 不做)

#include "kittens.cuh"
#include "schema.cuh"
#include "utils.cuh"
#include "itypes/llama1b/matvec_pipeline.cuh"

namespace megakittens {

template <typename Config, typename Globals, int N, int HEAD_DIM, int NUM_KV_HEADS,
          int SRC_ACT, int SRC_NORM, int SRC_QKV_W, int SRC_ROPE_COS, int SRC_ROPE_SIN,
          int SRC_K_CACHE, int SRC_V_CACHE, int SCALAR_POS_ID, int SCALAR_RMS_EPS,
          int DST_Q, int DST_K_CACHE = -1, int DST_V_CACHE = -1>
struct RmsQkvRopeAppend {
    // KV outputs alias their inputs (inplace); kernel writes via SRC_*
    static_assert(DST_K_CACHE == -1 || DST_K_CACHE == SRC_K_CACHE);
    static_assert(DST_V_CACHE == -1 || DST_V_CACHE == SRC_V_CACHE);
    static constexpr int BLOCK_SIZE = 16;
        // <NT> QKV 块索引边界
    //      Q: 0 .. 127  (2048 / 16 = 128 blocks)
    //      K: 128 .. 159  ((2048 + 512) / 16 = 160, but K starts at N/BLOCK_SIZE)
    //      V: 160 .. 191  ((2048 + 512 + 512) / 16 = 192)
    static constexpr int K_BLK_START = N / BLOCK_SIZE;
    static constexpr int V_BLK_START = (N + NUM_KV_HEADS * HEAD_DIM) / BLOCK_SIZE;
    // <NT> 每个 head 的 block 数, 和每个 group 的 block 数 (for GQA)
    static constexpr int BPH = HEAD_DIM / BLOCK_SIZE;
    static constexpr int GQA_RATIO = (N / HEAD_DIM) / NUM_KV_HEADS;
    static constexpr int BPG = BPH * GQA_RATIO;

    using rope_t = kittens::sv_fl<HEAD_DIM>;

    // <NT> subregion_offset: 将 block_idx 映射到 subregion 索引
    //      用于 barrier 分配 - Q/K/V 各有不同的 subregion
    //
    // Q 布局: 128 blocks, 分成 NUM_KV_HEADS=8 个 group, 每组 16 blocks
    //        block_idx 0..15 → group 0 (对应 kv_head 0)
    //        block_idx 16..31 → group 1 (对应 kv_head 1)
    //        ...
    //
    // K 布局: 32 blocks, 直接对应 8 个 kv_head, 每 head 4 blocks
    //        block_idx 128..131 → kv_head 0
    //        block_idx 132..135 → kv_head 1
    //        ...
    //
    // V 布局: 同 K
    //        block_idx 160..163 → kv_head 0
    //        block_idx 164..167 → kv_head 1
    //
    // subregion 返回值:
    //   0..7: Q groups (Q group 0..7)
    //   8..15: K heads (kv_head 0..7)
    //   16..23: V heads (kv_head 0..7)
    //
    // QKV_SUB_BARRIERS = 24 = 3 × 8
    // Q groups 0..NUM_KV_HEADS-1 | K heads NUM_KV_HEADS..2N-1 | V heads 2N..3N-1
    __device__ __host__ static inline int subregion_offset(int block_idx) {
        if (block_idx < K_BLK_START) return block_idx / BPG;
        if (block_idx < V_BLK_START) return NUM_KV_HEADS + (block_idx - K_BLK_START) / BPH;
        return 2 * NUM_KV_HEADS + (block_idx - V_BLK_START) / BPH;
    }

    struct parsed_instruction {
        int layer_idx, start_block_idx, end_block_idx, iters, start_sub;
        __device__ inline parsed_instruction(const instruction_t &instruction) {
            layer_idx       = instruction.indices[0];
            start_block_idx = instruction.indices[1];
            end_block_idx   = instruction.indices[2];
            iters           = end_block_idx - start_block_idx; // 该 SM 要处理的 block 数
            start_sub       = subregion_offset(start_block_idx);
        }
        __device__ inline parsed_instruction(state_t<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    struct pipeline_specifics {
        __device__ static inline void gmem_wait(const Globals &g, state_t<Config> &s) {
            all_input_barrier_wait<Config>(g, s.instruction());
        }

        __device__ static inline void load_iter(state_t<Config> &s, const Globals &g, parsed_instruction &inst,
                  int iter, int col_idx,
                  kittens::st_bf<16, 512> &weight_chunk,
                  kittens::semaphore &sem) {
            int block_idx = inst.start_block_idx + iter;
            kittens::tma::load_async<kittens::dim::ROW, kittens::cache_policy::EVICT_FIRST>(
                weight_chunk, g.template gls<SRC_QKV_W>(),
                {inst.layer_idx, block_idx, col_idx}, sem);
        }
        // <NT> store: 规约 + RoPE + 写入
        //      流程:
        //        1. matvec_reduce: 收集该 SM 所有 warp 的输出
        //        2. 加载 RoPE cos/sin
        //        3. 对 Q/K 应用 RoPE (V 不做)
        //        4. TMA store 写回
        //        5. barrier_arrive: 最后一个 block 触发
        //
        // Q: 32 heads × 4 blocks/head = 128 blocks  (每个 head 64 元素 / 16 per block)
        // K: 8 heads × 4 blocks/head = 32 blocks
        // V: 8 heads × 4 blocks/head = 32 blocks
        // 总: 192 blocks
        // 
        // rope是逐元素独立计算的kernel，matvec输出哪个tile，就加载对应的cos/sin数据进行计算即可。
        // rope[pos_id(当前token的位置，跟head无关), head_dim]，rope在这里会被各个block重复加载。
        // Q 矩阵的 block 划分 (block_idx = head_id * 4 + chunk):
        // ┌──────────────────────────────────────────────────────────────────┐
        // │  block 0 (head_id=0, chunk=0): Q[head_id, 0:16]   + rope[pos_id, 0:16]   │
        // │  block 1 (head_id=0, chunk=1): Q[head_id, 16:32]  + rope[pos_id, 16:32]  │
        // │  block 2 (head_id=0, chunk=2): Q[head_id, 32:48]  + rope[pos_id, 32:48]  │
        // │  block 3 (head_id=0, chunk=3): Q[head_id, 48:64]  + rope[pos_id, 48:64]  │
        // └──────────────────────────────────────────────────────────────────┘
        //   ↓ 只有 4 个 blocks 都完成，Q head 0 的 64 元素才算完整
        __device__ static inline void store(state_t<Config> &s, const Globals &g, parsed_instruction &inst,
              int output_idx, int output_stage) {
            int block_idx = inst.start_block_idx + output_idx;
            uint8_t *output_scratch = pipeline::get_output_start(s, output_stage);
            kittens::sv_bf<16> &qkv_proj_smem_bf =
                *reinterpret_cast<kittens::sv_bf<16> *>(output_scratch);
            kittens::rv_fl<16> qkv_proj;
            llama1b::matvec_reduce<Config, pipeline::SCRATCH_BYTES_PER_WARP>(
                output_scratch, qkv_proj);

            kittens::rv_fl<16> rope_cos_rv, rope_sin_rv;
            kittens::wait(rope_arrived(s), 0);
            auto head_chunk = block_idx % (HEAD_DIM / BLOCK_SIZE);
            kittens::sv_fl<16> &rope_cos_sv = *reinterpret_cast<kittens::sv_fl<16> *>(
                get_rope_cos_ptr(s) + head_chunk * BLOCK_SIZE * sizeof(float));
            kittens::sv_fl<16> &rope_sin_sv = *reinterpret_cast<kittens::sv_fl<16> *>(
                get_rope_sin_ptr(s) + head_chunk * BLOCK_SIZE * sizeof(float));
            kittens::warp::load(rope_cos_rv, rope_cos_sv);
            kittens::warp::load(rope_sin_rv, rope_sin_sv);

            // <NT> v不做rope
            if (block_idx < V_BLK_START) {
                int mod = (kittens::laneid() & 0b1) ? -1 : 1;
                kittens::warp::sync();
                float pair_val =
                    __shfl_sync(0xFFFFFFFF, qkv_proj[0][0], kittens::laneid() + mod);
                if (kittens::laneid() < 16) {
                    qkv_proj[0][0] =
                        float(qkv_proj[0][0]) * rope_cos_rv[0][0] +
                        float(-1 * mod) * float(pair_val) * rope_sin_rv[0][0];
                }
            }

            kittens::warp::sync();
            kittens::warp::store(qkv_proj_smem_bf, qkv_proj);
            kittens::warp::sync();

            if (kittens::warp::elect_leader()) {
                if (block_idx < K_BLK_START) { // Q
                    kittens::tma::store_async<kittens::cache_policy::EVICT_LAST>(
                        g.template gls<DST_Q>(), qkv_proj_smem_bf, {0, block_idx});
                } else if (block_idx < V_BLK_START) { // K
                    int base_index = (block_idx - K_BLK_START) * BLOCK_SIZE;
                    int head_idx = base_index / HEAD_DIM;
                    int dim_idx = (base_index % HEAD_DIM) / BLOCK_SIZE;
                    kittens::tma::store_async<kittens::cache_policy::EVICT_LAST>(
                        g.template gls<SRC_K_CACHE>(), qkv_proj_smem_bf,
                        {inst.layer_idx, static_cast<int>(g.template gls<SCALAR_POS_ID>().raw_ptr[0]), head_idx, dim_idx});
                } else { // V
                    int base_index = (block_idx - V_BLK_START) * BLOCK_SIZE;
                    int head_idx = base_index / HEAD_DIM;
                    int dim_idx = (base_index % HEAD_DIM) / BLOCK_SIZE;
                    kittens::tma::store_async<kittens::cache_policy::EVICT_LAST>(
                        g.template gls<SRC_V_CACHE>(), qkv_proj_smem_bf,
                        {inst.layer_idx, static_cast<int>(g.template gls<SCALAR_POS_ID>().raw_ptr[0]), head_idx, dim_idx});
                }
                kittens::tma::store_async_wait();
                // one arrive per dst_barriers entry: fire on the last block of each sub-region
                int curr_sub = subregion_offset(block_idx);
                bool last_of_sub = (output_idx + 1 == inst.iters) ||
                                   (subregion_offset(block_idx + 1) != curr_sub);
                if (last_of_sub) {
                    int k = curr_sub - inst.start_sub;
                    barrier_arrive<Config>(
                        &g.barriers.raw_ptr[s.instruction().dst_barriers[k]], 1);
                }
            }
            kittens::warp::sync();
        }
    };

    // <NT> rms_matvec_pipeline: rms+matvec融合逻辑复用
    using pipeline = llama1b::rms_matvec_pipeline<
        Config, Globals, N, parsed_instruction, pipeline_specifics, SRC_ACT, SRC_NORM, SCALAR_RMS_EPS>;

    // <NT> RoPE cos/sin 在 activation page 中的偏移
    //      布局: [RMS 输出 scratch] [RoPE cos] [RoPE sin]
    static constexpr int ROPE_COS_OFFSET = ((pipeline::OUTPUT_SCRATCH_OFFSET + // 1024-align
        pipeline::OUTPUT_PIPELINE_STAGES * pipeline::SCRATCH_BYTES_PER_STAGE) + 1023) & ~1023;
    static constexpr int ROPE_SIN_OFFSET = ROPE_COS_OFFSET + HEAD_DIM * sizeof(float);

    __device__ static inline uint8_t *get_rope_cos_ptr(state_t<Config> &s) {
        return static_cast<uint8_t *>(s.pages[pipeline::get_activation_page(s)].ptr(ROPE_COS_OFFSET));
    }
    __device__ static inline uint8_t *get_rope_sin_ptr(state_t<Config> &s) {
        return static_cast<uint8_t *>(s.pages[pipeline::get_activation_page(s)].ptr(ROPE_SIN_OFFSET));
    }

    __device__ static inline kittens::semaphore &rope_arrived(state_t<Config> &s) { return s.semaphores()[pipeline::SEM_COUNT]; }

    struct controller {
        __device__ __forceinline__ static int lid_release_order(const Globals &g, state_t<Config> &s, int query) {
            return pipeline::lid_release_order(g, s, query);
        }
        __device__ __forceinline__ static int init_semaphores(const Globals &g, state_t<Config> &s) {
            pipeline::init_semaphores(g, s);
            if (kittens::warp::elect_leader())
                kittens::init_semaphore(rope_arrived(s), 1);
            return pipeline::SEM_COUNT + 1;
        }
    };

    // <NT> Loader: 加载 RoPE cos/sin + QKV 权重
    struct loader {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            if (kittens::warp::elect_leader()) {
                s.page_wait(pipeline::get_activation_page(s));
                rope_t &rope_cos_smem = *reinterpret_cast<rope_t *>(get_rope_cos_ptr(s));
                rope_t &rope_sin_smem = *reinterpret_cast<rope_t *>(get_rope_sin_ptr(s));
                int pos_id = static_cast<int>(g.template gls<SCALAR_POS_ID>().raw_ptr[0]);
                auto &sem = rope_arrived(s);
                kittens::tma::expect_bytes(sem, 2 * HEAD_DIM * sizeof(float));
                kittens::tma::load_async<kittens::cache_policy::EVICT_LAST>(
                    rope_cos_smem, g.template gls<SRC_ROPE_COS>(), {0, 0, pos_id, 0}, sem);
                kittens::tma::load_async<kittens::cache_policy::EVICT_LAST>(
                    rope_sin_smem, g.template gls<SRC_ROPE_SIN>(), {0, 0, pos_id, 0}, sem);
            }
            parsed_instruction inst{s.instruction()};
            pipeline::loader_loop(s, g, inst.layer_idx);
        }
    };

    struct launcher {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            pipeline::launcher_loop(s, g);
        }
    };

    struct consumer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            pipeline::consumer_loop(s, g);
        }
    };

    struct storer {
        __device__ __forceinline__ static void run(const Globals &g, state_t<Config> &s) {
            pipeline::storer_loop(s, g);
            if (kittens::warp::elect_leader())
                all_reuse_barrier_arrive<Config>(g, s.instruction());
        }
    };
};

} // namespace megakittens
