#pragma once

#include "kittens.cuh"

#include "schema.cuh"
#include "utils.cuh"
#include "controller.cuh"
#include "workers.cuh"

namespace megakittens {

// <NT> 主 kernel 入口, 由 4 个 warpgroup 组成:
//      - Warpgroup 0 (1 warp): controller_loop
//      - Warpgroup 1 (1 warp): loader_loop
//      - Warpgroup 2 (1 warp): launcher_loop
//      - Warpgroup 3 (1 warp): storer_loop
//      - Warpgroup 4-11 (8 warps): consumer_loop
template <typename Config, typename Globals>
__global__ __launch_bounds__(Config::NUM_THREADS, Config::MIN_BLOCKS_PER_SM)
void kernel(const __grid_constant__ Globals g) {
    // Allocate shared memory
    __shared__ alignas(128) instruction_state_t<Config> instruction_states[Config::INSTRUCTION_PIPE_STAGES];
    // <NT> CLC (Compute Language Center) 句柄和信号量, 用于动态调度 (CLC handle and semaphores for dynamic scheduling)
    __shared__ kittens::clc::handle clc_handle[Config::INSTRUCTION_PIPE_STAGES];
    __shared__ kittens::semaphore clc_arrived[Config::INSTRUCTION_PIPE_STAGES];
    // <NT> 指令到达/完成信号量, worker 间同步用
    //      instruction_arrived: controller 发送, worker 等待
    //      instruction_finished: worker 发送, controller 等待
    __shared__ kittens::semaphore instruction_arrived[Config::INSTRUCTION_PIPE_STAGES];
    __shared__ kittens::semaphore instruction_finished[Config::INSTRUCTION_PIPE_STAGES];
    // <NT> 页面完成信号量
    __shared__ kittens::semaphore page_finished[Config::NUM_PAGES];
    // <NT> 张量完成信号量
    __shared__ kittens::semaphore tensor_finished;
    // <NT> 外部共享内存, 用于存储动态分配的页面
    extern __shared__ int __shm[];
    // <NT> 将 __shm 强转为 page_t 数组, 1024 字节对齐 
    page_t<Config> (&pages)[Config::NUM_PAGES] = *reinterpret_cast<page_t<Config>(*)[Config::NUM_PAGES]>(
            reinterpret_cast<void *>(((uint64_t)&__shm[0] + 1023) & ~(uint64_t)1023));

    // Allocate tensor memory
    kittens::tensor_allocator<1, Config::CLUSTER_SIZE> tensor_alloc;

    // Instantiate MegaKittens state
    state_t<Config> s{0, 0, clc_handle, clc_arrived,
                      instruction_states, instruction_arrived, instruction_finished,
                      pages, page_finished, tensor_finished, tensor_alloc};

    // <NT> 初始化公共信号量
    //      - instruction_arrived[stage]: 由 controller 发送, worker 等待
    //      - instruction_finished[stage]: 由 worker 发送, controller 等待
    //      - clc_arrived[stage]: 用于动态调度
    //      - page_finished[page]: 页面完成信号量
    //      - tensor_finished: 张量完成信号量
    // Initialize common semaphores
    if (threadIdx.x < Config::INSTRUCTION_PIPE_STAGES) {
        init_semaphore(instruction_arrived[threadIdx.x], Config::CLUSTER_SIZE);
    } else if (threadIdx.x < Config::INSTRUCTION_PIPE_STAGES*2) {
        init_semaphore(instruction_finished[threadIdx.x - Config::INSTRUCTION_PIPE_STAGES], Config::NUM_WARPS - 1);
    } else if (threadIdx.x < Config::INSTRUCTION_PIPE_STAGES*3) {
        init_semaphore(clc_arrived[threadIdx.x - Config::INSTRUCTION_PIPE_STAGES*2], 1);
    } else if (threadIdx.x < Config::INSTRUCTION_PIPE_STAGES*3 + Config::NUM_PAGES) {
        init_semaphore(page_finished[threadIdx.x - Config::INSTRUCTION_PIPE_STAGES*3], 1);
        arrive(page_finished[threadIdx.x - Config::INSTRUCTION_PIPE_STAGES*3], 1);
    } else if (threadIdx.x < Config::INSTRUCTION_PIPE_STAGES*3 + Config::NUM_PAGES + 1) {
        init_semaphore(tensor_finished, 1);
        arrive(tensor_finished, 1);
    }
    // <NT> Cluster 同步或 block 同步，这个操作是否冗余？
    if constexpr (Config::CLUSTER_SIZE > 1) kittens::everyone::tma::cluster::sync();
    else __syncthreads();
    // <NT> grid级别同步，等待所有 CTA 就绪，一起启动
    kittens::pdl::wait();

    // <NT> 以warpid区分，使controller/loader/launcher/storer同时运行。
    // 指令内，如rms_qkv_rope_append，通过内部手动精准调控，完成深度融合（smem衔接/权重预取）。
    // 指令间，通过当前指令的launcher和后置指令的loader并行，实现权重预取的轻度融合。
    //        (基于指令级双缓冲-INSTRUCTION_PIPE_STAGES = 2，两份指令状态/到达信号量/完成信号量，
    //         Loader 加载 Inst N+1 的权重时，Consumer 正在执行 Inst N)
    //
    // Initiate the main loops
    if (kittens::warpid() < Config::NUM_CONSUMER_WARPS) {
        kittens::warpgroup::increase_registers<Config::CONSUMER_REGISTERS>();
        consumer_loop<Config, Globals>(g, s);
    } else {
        kittens::warpgroup::decrease_registers<Config::NON_CONSUMER_REGISTERS>();
        switch (kittens::warpgroup::warpid()) {
            case 0:
                controller_loop<Config, Globals>(g, s);
                break;
            case 1:
                loader_loop<Config, Globals>(g, s);
                break;
            case 2:
                launcher_loop<Config, Globals>(g, s);
                break;
            case 3:
                storer_loop<Config, Globals>(g, s);
                break;
            default:
                asm volatile("{trap;\n}");
        }
    }

    // Sync all threads in the cluster before exiting
    if constexpr (Config::CLUSTER_SIZE > 1) kittens::everyone::tma::cluster::sync();
    else __syncthreads();
}

} // namespace megakittens
