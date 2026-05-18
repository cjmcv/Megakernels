#pragma once

#include "kittens.cuh"
#include "schema.cuh"
#include "utils.cuh"
#include "itypes/llama1b/utils.cuh"

namespace megakittens {
namespace llama1b {

// <NT> matvec_pipeline: 矩阵×向量 (MatVec) 的流水线实现
//      T: Config, Globals, N (hidden dim), parsed_instruction, pipeline_specifics
//
// 流水线设计:
//   - 输入流水线: INPUT_PIPELINE_STAGES = 3 级
//   - 输出流水线: OUTPUT_PIPELINE_STAGES = 3 级
//   - 每级 2 个页面 (STAGE_PAGES = 2), 共 6 个页面用于权重
//
// 页面布局 (NUM_PAGES=7):
//   page 0: activation page (包含输出 scratch)
//   page 1-2: stage 0 weights
//   page 3-4: stage 1 weights
//   page 5-6: stage 2 weights
template <typename Config, typename Globals, int N,
          typename parsed_instruction, typename pipeline_specifics>
struct matvec_pipeline {
    static constexpr int INPUT_PIPELINE_STAGES = 3;
    static constexpr int OUTPUT_PIPELINE_STAGES = 3;
    static constexpr int STAGE_PAGES = 2;
    static constexpr int ACTIVATION_PAGE = 0;
    static constexpr int WEIGHTS_START_PAGE = 1;

    static constexpr int MATVEC_BLOCK_SIZE = 16;
    static constexpr int TILES_PER_PAGE = Config::NUM_CONSUMER_WARPS / STAGE_PAGES; // 4
    static constexpr int TILES_PER_STAGE = STAGE_PAGES * TILES_PER_PAGE;             // 8

    static constexpr int REDUCTION_DIM_PER_WARP = N / Config::NUM_CONSUMER_WARPS;
    static constexpr int WARPS_PER_PAGE = Config::NUM_CONSUMER_WARPS / STAGE_PAGES;

    static constexpr int SEM_COUNT = 1 + (INPUT_PIPELINE_STAGES + OUTPUT_PIPELINE_STAGES) * 2;

    static constexpr int SCRATCH_BYTES_PER_WARP = MATVEC_BLOCK_SIZE * sizeof(float);
    static constexpr int SCRATCH_BYTES_PER_STAGE = SCRATCH_BYTES_PER_WARP * Config::NUM_CONSUMER_WARPS;

    // <NT> Activation 页面布局:
    //      [0, N*sizeof(bf16)): 激活值
    //      [N*sizeof(bf16), ...): 输出 scratch (用于收集多 warp 结果)
    // offsets on activation page
    static constexpr int OUTPUT_SCRATCH_OFFSET = N * sizeof(kittens::bf16);

    __device__ static inline kittens::semaphore &activations_arrived(state_t<Config> &s)            { return s.semaphores()[0]; }
    __device__ static inline kittens::semaphore &weights_arrived(state_t<Config> &s, int stage)    { return s.semaphores()[1 + stage]; }
    __device__ static inline kittens::semaphore &weights_finished(state_t<Config> &s, int stage)   { return s.semaphores()[1 + INPUT_PIPELINE_STAGES + stage]; }
    __device__ static inline kittens::semaphore &outputs_arrived(state_t<Config> &s, int stage)    { return s.semaphores()[1 + 2 * INPUT_PIPELINE_STAGES + stage]; }
    __device__ static inline kittens::semaphore &outputs_finished(state_t<Config> &s, int stage)   { return s.semaphores()[1 + 2 * INPUT_PIPELINE_STAGES + OUTPUT_PIPELINE_STAGES + stage]; }

    __device__ static inline int get_activation_page(state_t<Config> &s)                    { return s.lid_to_pid(ACTIVATION_PAGE); }
    __device__ static inline int get_weight_page(state_t<Config> &s, int stage, int page)  { return s.lid_to_pid(WEIGHTS_START_PAGE + stage * STAGE_PAGES + page); }

    __device__ static inline kittens::sv_bf<N> &get_activations(state_t<Config> &s) {
        return s.pages[get_activation_page(s)].template as<kittens::sv_bf<N>>();
    }
    // <NT> 获取输出 scratch 起始位置 (每个 stage 有独立的 scratch 区域)
    __device__ static inline uint8_t *get_output_start(state_t<Config> &s, int stage) {
        return static_cast<uint8_t *>(s.pages[get_activation_page(s)].ptr(OUTPUT_SCRATCH_OFFSET + stage * SCRATCH_BYTES_PER_STAGE));
    }

    // <NT> lid_release_order: 控制页面释放顺序, 避免页面竞争
    //      activation page 最后释放 (因为包含 scratch)
    //      其他页面根据迭代次数动态决定释放顺序
    __device__ static inline int lid_release_order(const Globals &g, state_t<Config> &s, int query) {
        // activation page last (contains scratch)
        if (query == Config::NUM_PAGES - 1) return ACTIVATION_PAGE;

        parsed_instruction inst{s.instruction()};
        int remainder = inst.iters % INPUT_PIPELINE_STAGES;

        if (inst.iters == 1) {
            // unused pages first
            constexpr int order[] = {3, 4, 5, 6, 1, 2};
            return order[query];
        } else if (inst.iters == 2) {
            constexpr int order[] = {5, 6, 1, 2, 3, 4};
            return order[query];
        } else if (remainder == 1) {
            // 3 and 4 finish first.
            constexpr int order[] = {3, 4, 5, 6, 1, 2};
            return order[query];
        } else if (remainder == 2) {
            constexpr int order[] = {5, 6, 1, 2, 3, 4};
            return order[query];
        } else {
            constexpr int order[] = {1, 2, 3, 4, 5, 6};
            return order[query];
        }
    }

    __device__ static inline int init_semaphores(const Globals &g, state_t<Config> &s) {
        if (kittens::laneid() == 0)
            kittens::init_semaphore(activations_arrived(s), 1);
        if (kittens::laneid() < INPUT_PIPELINE_STAGES)
            kittens::init_semaphore(weights_arrived(s, kittens::laneid()), 1);
        if (kittens::laneid() < INPUT_PIPELINE_STAGES)
            kittens::init_semaphore(weights_finished(s, kittens::laneid()), Config::NUM_CONSUMER_WARPS);
        if (kittens::laneid() < OUTPUT_PIPELINE_STAGES)
            kittens::init_semaphore(outputs_arrived(s, kittens::laneid()), Config::NUM_CONSUMER_WARPS);
        if (kittens::laneid() < OUTPUT_PIPELINE_STAGES)
            kittens::init_semaphore(outputs_finished(s, kittens::laneid()), 1);
        return SEM_COUNT;
    }

    // <NT> Launcher: 等待 tensor 就绪即可 (无实际计算)
    __device__ static inline void launcher_loop(state_t<Config> &s, const Globals &g) {
        s.tensor_wait();
        if (kittens::warp::elect_leader()) s.tensor_finish();
    }

    // <NT> Loader: 加载权重矩阵
    //      1. 等待 activation page 就绪
    //      2. 循环加载权重, 利用流水线隐藏延迟
    __device__ static inline void loader_loop(state_t<Config> &s, const Globals &g) {
        parsed_instruction inst{s.instruction()};

        int needed_pages = 1 + min(inst.iters, INPUT_PIPELINE_STAGES) * STAGE_PAGES;

        if (kittens::laneid() == 0) {
            // <NT> 先等激活页就绪
            s.page_wait(get_activation_page(s));
            // <NT> 当前加载到哪一级 (0,1,2 循环)
            int input_stage = 0;
            for (int iter = 0; iter < inst.iters; iter++) {
                // <NT> 等待上一批权重计算完成 (流水线_flush)
                kittens::wait(weights_finished(s, input_stage),
                    (iter % (2 * INPUT_PIPELINE_STAGES)) < INPUT_PIPELINE_STAGES);

                auto &sem = weights_arrived(s, input_stage);
                kittens::tma::expect_bytes(sem, sizeof(kittens::bf16) * N * MATVEC_BLOCK_SIZE);

                // 2 pages per stage, 2 × st_bf<16,512> per page = 4 TMA loads
                #pragma unroll
                for (int i = 0; i < STAGE_PAGES * 2; i++) {
                    int pid = get_weight_page(s, input_stage, i / 2);
                    if (iter < INPUT_PIPELINE_STAGES && i % 2 == 0)
                        s.page_wait(pid);
                    auto &tile = s.pages[pid].template as<
                        kittens::st_bf<MATVEC_BLOCK_SIZE, 512>>(
                        (i % 2) * sizeof(kittens::st_bf<MATVEC_BLOCK_SIZE, 512>));
                    pipeline_specifics::load_iter(s, g, inst, iter, i, tile, sem);
                }
                // <NT> advance 到下一级 (0→1→2→0 循环)
                input_stage = (input_stage + 1) % INPUT_PIPELINE_STAGES;
            }
        }

        // <NT> 处理多余页面的释放 (防止资源泄漏)
        if (kittens::laneid() >= needed_pages && kittens::laneid() < Config::NUM_PAGES) {
            int pid = s.lid_to_pid(kittens::laneid());
            s.page_wait(pid);
            s.page_finish(pid);
        }
    }

    // <NT> Consumer: 执行 MatVec 计算
    //      每个 warp 处理一个 tile 的权重, 与 activations 向量相乘
    //      输出写入 scratch, 后续由 storer 规约合并
    //
    // @param output_scratch_off: 输出 scratch 在 activation page 中的偏移
    // @param activations_vec: 输入激活向量 (已经过 RMS Norm 的版本)
    template <int output_scratch_off = OUTPUT_SCRATCH_OFFSET, typename rv_t>
    __device__ static inline void consumer_loop(state_t<Config> &s, const Globals &g, rv_t &activations_vec) {
        parsed_instruction inst{s.instruction()};
        int page_index = kittens::warpid() / WARPS_PER_PAGE;
        int activation_page = get_activation_page(s);
        int input_stage = 0, output_stage = 0;
        for (int i = 0; i < inst.iters; i++) {
            int weight_page = get_weight_page(s, input_stage, page_index);
            // <NT> 等待权重加载完成 && 输出 slot 可用
            kittens::wait(weights_arrived(s, input_stage),
                (i % (2 * INPUT_PIPELINE_STAGES)) >= INPUT_PIPELINE_STAGES);
            kittens::wait(outputs_finished(s, output_stage),
                (i % (2 * OUTPUT_PIPELINE_STAGES)) < OUTPUT_PIPELINE_STAGES);
            // <NT> 获取权重 tile 和输出 scratch
            using weight_t = kittens::st_bf<MATVEC_BLOCK_SIZE, REDUCTION_DIM_PER_WARP>;
            using out_t = kittens::sv_fl<MATVEC_BLOCK_SIZE>;
            weight_t &weights = reinterpret_cast<weight_t *>(
                s.pages[weight_page].ptr())[kittens::warpid() % WARPS_PER_PAGE];
            auto *out_base = reinterpret_cast<uint8_t *>(s.pages[activation_page].ptr(
                output_scratch_off + output_stage * SCRATCH_BYTES_PER_STAGE));
            out_t &out_smem = *reinterpret_cast<out_t *>(
                out_base + kittens::warpid() * SCRATCH_BYTES_PER_WARP);
            
            // <NT> 执行 matvec: weights × activations_vec
            llama1b::matvec(out_smem, weights, activations_vec);
            // <NT> 标记权重计算完成 && 输出已生成
            kittens::warp::arrive(outputs_arrived(s, output_stage));
            kittens::warp::arrive(weights_finished(s, input_stage));

            // <NT> 流水线 Flush: 当迭代接近结束时, 释放已用完的页面
            if (i + INPUT_PIPELINE_STAGES >= inst.iters) {
                kittens::group<Config::NUM_CONSUMER_WARPS>::sync(1);
                if (kittens::warpid() == 0 && kittens::warp::elect_leader()) {
                    int released_stage = i % INPUT_PIPELINE_STAGES;
                    for (int p = 0; p < STAGE_PAGES; p++)
                        s.page_finish(get_weight_page(s, released_stage, p));
                }
            }

            input_stage  = (input_stage  + 1) % INPUT_PIPELINE_STAGES;
            output_stage = (output_stage + 1) % OUTPUT_PIPELINE_STAGES;
        }
    }

    // <NT> Storer: 规约合并多 warp 输出, 写回全局内存
    template <int iter_scale = 1>
    __device__ static inline void storer_loop(state_t<Config> &s, const Globals &g) {
        parsed_instruction inst{s.instruction()};
        int output_stage = 0;
        for (int i = 0; i < inst.iters; i++) {
            // <NT> 等待该 stage 的输出可用
            kittens::wait(outputs_arrived(s, output_stage),
                (i % (2 * OUTPUT_PIPELINE_STAGES)) >= OUTPUT_PIPELINE_STAGES);

            // <NT> 调用具体的存储实现 (由 pipeline_specifics 提供)            
            pipeline_specifics::store(s, g, inst, i, output_stage);
            // <NT> 标记输出已完成
            if ((i + 1) % iter_scale == 0) {
                for (int j = 0; j < iter_scale; j++) {
                    int stage = (output_stage - j + OUTPUT_PIPELINE_STAGES) % OUTPUT_PIPELINE_STAGES;
                    kittens::warp::arrive(outputs_finished(s, stage));
                }
            }

            output_stage = (output_stage + 1) % OUTPUT_PIPELINE_STAGES;
        }

        // storer releases activation page
        kittens::warp::sync();
        if (kittens::warp::elect_leader()) {
            kittens::tma::store_async_wait();
            s.page_finish(get_activation_page(s));
        }
    }
};

// <NT> rms_matvec_pipeline: 将 RMS Norm 与 MatVec 融合的流水线
//      融合方式:
//        1. loader 先加载 rms_scale (预计算的缩放因子)
//        2. consumer 先执行 RMS Norm, 然后再做 MatVec
//        3. 两者共享 activation page, 无需额外同步
//
// 融合前:  RMS_Norm(activations) → MatVec(weights, normalized_activations)
// 融合后:  一个流水线, activation page 同时存储:
//          - 原始激活值
//          - RMS 缩放因子
//          - 中间结果 (scratch)
template <typename Config, typename Globals, int N,
          typename parsed_instruction, typename pipeline_specifics,
          int SRC_ACT, int SRC_NORM, int SCALAR_RMS_EPS>
struct rms_matvec_pipeline
    : public matvec_pipeline<Config, Globals, N, parsed_instruction, pipeline_specifics> {

    using pipeline = matvec_pipeline<Config, Globals, N, parsed_instruction, pipeline_specifics>;

    // <NT> Activation page 布局:
    //      [0, N*sizeof(bf16)):              原始激活值 (bf16)
    //      [N*sizeof(bf16), 2N*sizeof(bf16)): RMS 缩放因子 (bf16)
    //      [2N*sizeof(bf16), ...):            RMS scratch (float, 每 warp 1 个)
    //      [aligned, ...):                   输出 scratch
    static constexpr int ACTIVATIONS_SIZE      = N * sizeof(kittens::bf16);
    static constexpr int RMS_SCALE_OFFSET      = ACTIVATIONS_SIZE;
    static constexpr int RMS_SCALE_SIZE        = N * sizeof(kittens::bf16);
    static constexpr int RMS_SCRATCH_OFFSET    = RMS_SCALE_OFFSET + RMS_SCALE_SIZE;
    static constexpr int RMS_SCRATCH_SIZE      = Config::NUM_CONSUMER_WARPS * sizeof(float);
    static constexpr int OUTPUT_SCRATCH_OFFSET = ((RMS_SCRATCH_OFFSET + RMS_SCRATCH_SIZE) + 1023) & ~1023; // 1024-align

    static constexpr int SEM_COUNT = pipeline::SEM_COUNT + 1;

    // <NT> RMS 缩放因子信号量
    __device__ static inline kittens::semaphore &rms_scale_arrived(state_t<Config> &s) { return s.semaphores()[pipeline::SEM_COUNT]; }
    __device__ static inline kittens::sv_bf<N> &get_rms_scale(state_t<Config> &s) {
        return s.pages[pipeline::get_activation_page(s)].template as<kittens::sv_bf<N>>(RMS_SCALE_OFFSET);
    }
    __device__ static inline uint8_t *get_output_start(state_t<Config> &s, int stage) {
        return static_cast<uint8_t *>(s.pages[pipeline::get_activation_page(s)].ptr(
            OUTPUT_SCRATCH_OFFSET + stage * pipeline::SCRATCH_BYTES_PER_STAGE));
    }

    __device__ static inline int lid_release_order(const Globals &g, state_t<Config> &s, int query) {
        return pipeline::lid_release_order(g, s, query);
    }

    __device__ static inline int init_semaphores(const Globals &g, state_t<Config> &s) {
        pipeline::init_semaphores(g, s);
        if (kittens::warp::elect_leader())
            kittens::init_semaphore(rms_scale_arrived(s), 1);
        return SEM_COUNT;
    }

    // <NT> Loader: 先加载 RMS 缩放因子, 再加载权重
    __device__ static inline void loader_loop(state_t<Config> &s, const Globals &g, int norm_layer_idx = 0) {
        if (kittens::warp::elect_leader()) {
            s.page_wait(pipeline::get_activation_page(s));
            auto &rms_scale = get_rms_scale(s);
            auto &rms_sem = rms_scale_arrived(s);
            kittens::tma::expect_bytes(rms_sem, sizeof(rms_scale));
            // <NT> 异步加载 RMS 缩放因子
            kittens::tma::load_async<kittens::cache_policy::EVICT_LAST>(
                rms_scale, g.template gls<SRC_NORM>(), {0, 0, norm_layer_idx, 0}, rms_sem);
        }
        // <NT> 继续加载权重 (调用基类matvec的 loader)
        pipeline::loader_loop(s, g);
    }

    // <NT> Consumer: RMS Norm → MatVec 融合计算
    //      关键融合点:
    //        1. RMS Norm 的输出直接作为 MatVec 的输入 (activations_vec)
    //        2. 无需额外的存储/加载, 数据直接在寄存器间传递
    //        3. 两者的 scratch 可以共享 (都在 activation page 中)
    //        4. RMS Norm似乎每个block都重复计算了？
    //           -- RMS Norm原本只需要占用一个block，其结果作为matvec的输入，有依赖关系，计算时其他block会闲置。
    //              计算完了后，需要将结果同步给其他block，让matvec的其他block也拿到输入，才能继续计算。
    //              缺点：需要跨block同步，且激活需要写回到gmem进行中转。
    //           -- 这里的方案，将rms norm在每个block都进行计算，虽然是重复计算，但利用的是闲置资源。
    //              好处：形成一对一的block级别依赖，不再需要跨block同步，且激活值可以直接由共享内存传递给matvec。
    __device__ static inline void consumer_loop(state_kt<Config> &s, const Globals &g) {
        constexpr int ELEMS_PER_WARP = N / Config::NUM_CONSUMER_WARPS;
        using sv_slice_t = kittens::sv_bf<ELEMS_PER_WARP>;
        using rv_t = kittens::rv_fl<ELEMS_PER_WARP>;
        // <NT> 等待 GMEM 操作完成
        if (kittens::warpid() == 0 && kittens::warp::elect_leader())
            pipeline_specifics::gmem_wait(g, s);
        kittens::group<Config::NUM_CONSUMER_WARPS>::sync(4);

        // <NT> Step 1: 从全局内存加载激活值到寄存器
        rv_t activations_vec;
        const kittens::bf16 *act_src = reinterpret_cast<const kittens::bf16 *>(
            g.template gls<SRC_ACT>().raw_ptr) + kittens::warpid() * ELEMS_PER_WARP;
        #pragma unroll
        for (int w = 0; w < rv_t::outer_dim; w++)
            activations_vec.data[w][0] = __bfloat162float(
                act_src[w * kittens::WARP_THREADS + kittens::laneid()]);

        // <NT> Step 2: 等待 RMS 缩放因子加载完成
        kittens::wait(rms_scale_arrived(s), 0);
        // <NT> Step 3: RMS Norm - 使用 utils.cuh 中的实现
        //      rms_norm 返回归一化后的 activations_vec
        auto &rms_scale_smem = reinterpret_cast<sv_slice_t *>(&get_rms_scale(s))[kittens::warpid()];
        float *rms_scratch = static_cast<float *>(s.pages[pipeline::get_activation_page(s)].ptr(RMS_SCRATCH_OFFSET));
        // <NT> 融合点: rms_norm 的输出直接作为下面 matvec 的输入
        activations_vec = llama1b::rms_norm<Config, N>(
            activations_vec, rms_scale_smem, g.template gls<SCALAR_RMS_EPS>().raw_ptr[0], rms_scratch);
        kittens::warp::sync();
        // <NT> Step 4: MatVec 计算 - 使用归一化后的 activations_vec
        //      注意: 这里传入的 activations_vec 已经是 RMS Norm 后的版本
        pipeline::template consumer_loop<OUTPUT_SCRATCH_OFFSET>(s, g, activations_vec);
    }
};

} // namespace llama1b
} // namespace megakittens
