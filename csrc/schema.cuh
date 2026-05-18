#pragma once

#include "kittens.cuh"

namespace megakittens {

// <NT> Worker 类型枚举, 与 dispatch_instruction 的模板参数对应
//      page_manager:    管理共享内存页面的分配和释放顺序
//      semaphore_manager: 初始化动态信号量
//      loader:          加载数据到共享内存
//      launcher:        启动计算 (GEMM、Attention 等)
//      consumer:        执行实际计算 (warp 最多的 worker)
//      storer:          将结果写回全局内存
enum WorkerType {
    page_manager = 0,
    semaphore_manager = 1,
    loader = 2,
    launcher = 3,
    consumer = 4,
    storer = 5
};

// <NT> MegaKittens IType (instruction type 具体算子实现) 的 concept约束
//      每个指令类型必须定义 controller, loader, launcher, consumer, storer 五种结构
template <typename IType>
concept MegaKittensIType = requires {
    typename IType::controller;
    typename IType::loader;
    typename IType::launcher;
    typename IType::consumer;
    typename IType::storer;
};

// <NT> 指令结构体 (256 字节), 包含执行一条指令所需的所有元数据
struct instruction_t {
    static constexpr int MAX_SRC_TENSORS = 16;
    static constexpr int MAX_DST_TENSORS = 8;
    static constexpr int MAX_INDICES = 16;
    static constexpr int MAX_SRC_BARRIERS = 16;
    static constexpr int MAX_DST_BARRIERS = 8;

    // <NT> 指令操作码, 指向具体 handler
    int icode;                                  //  4B
    uint8_t src_tensors[MAX_SRC_TENSORS];       // 16B
    uint8_t dst_tensors[MAX_DST_TENSORS];       //  8B
    int indices[MAX_INDICES];                   // 64B
    uint32_t src_barriers[MAX_SRC_BARRIERS];    // 64B
    int src_barrier_targets[MAX_SRC_BARRIERS];  // 64B
    uint8_t num_src_input_barriers;             //  1B
    uint8_t num_src_reuse_barriers;             //  1B
    uint8_t num_dst_input_barriers;             //  1B
    uint8_t num_dst_reuse_barriers;             //  1B
    uint32_t dst_barriers[MAX_DST_BARRIERS];    // 32B
};
// <NT> 确保指令大小符合 TMA 对齐要求
static_assert(sizeof(instruction_t) == 256);

// <NT> 默认配置, 定义流水线、内存、线程等参数
struct default_config {
    static constexpr bool GLOBAL_WORK_QUEUE = false;

    // <NT> 表示Controller 同时管理 2 条指令的状态，与LOAD_PIPE_DEPTH不是一回事，LOAD_PIPE_DEPTH在算子内部自己定义使用
    // 指令状态双缓冲，意味着可以 stage 0：等待指令 1 完成; stage 1：同时准备指令 2
    static constexpr int INSTRUCTION_PIPE_STAGES = 2;
    // <NT> CLUSTER_SIZE 被写死为2，是针对B100/B200，作者认为最优的特化参数（数据中心 GPU，2 个 CTA 组成 cluster 是常见配置）
    static constexpr int CLUSTER_SIZE = 2;
    static constexpr int MIN_BLOCKS_PER_SM = 1;
    static_assert(INSTRUCTION_PIPE_STAGES == 2 && MIN_BLOCKS_PER_SM == 1);
    static_assert(CLUSTER_SIZE == 1 || CLUSTER_SIZE == 2);

    // <NT> 4个warp对应Controller/Loader/Launcher/Storer，加上8个用于计算的Consumer，共12个warp组成一个block
    // 为什么是8个warp：Blackwell SM 有 4 个 warp scheduler，8 warps 刚好 2 warp/scheduler。但可能不一定是最优化，可能也是作者试验后的B100/B200特化数据。
    static constexpr int NUM_CONSUMER_WARPS = 8;
    static constexpr int NUM_WARPS = 4 + NUM_CONSUMER_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;
    // <NT> blackwell不再支持显式修改寄存器数量，这里的寄存器调整可能是空的无效操作（ThunderKittens 库内部可能做了兼容处理）
    static constexpr int CONSUMER_REGISTERS = 224;
    static constexpr int NON_CONSUMER_REGISTERS = 56;

    static constexpr int DYNAMIC_SEMAPHORES = 32;
    static_assert(DYNAMIC_SEMAPHORES <= 32); // for warp parallel processing

    // <NT>  PAGE_SIZE 是唯一需要调的 knob，决定了：
    //       -- STATIC_SHARED_MEMORY: 静态内存大小
    //       -- DYNAMIC_SHARED_MEMORY: 动态内存大小
    //       -- NUM_PAGES: 分页数量
    //       -- SCRATCH_BYTES: 每阶段的 scratch 大小, Scratch 内存是 指令执行过程中的临时存储空间，生命周期在单指令内。
    //                         如 规约操作的临时存储 / 算子输出缓冲 / Cos/Sin 预计算。
    //                         按 INSTRUCTION_PIPE_STAGES 分配，因为指令用了双缓冲，所以每个缓冲都需要有自己的一份。
    static constexpr int PAGE_SIZE = 32768; // this is the only knob and everything else is derived
    static constexpr int STATIC_SHARED_MEMORY_BASE = 512 + INSTRUCTION_PIPE_STAGES*(sizeof(instruction_t) + 128 + DYNAMIC_SEMAPHORES*8);
    static constexpr int DYNAMIC_SHARED_MEMORY_ALIGN = 1024; // alignment overhead
    static constexpr int NUM_PAGES = (kittens::MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY_BASE - DYNAMIC_SHARED_MEMORY_ALIGN) / PAGE_SIZE;
    static_assert(NUM_PAGES >= 1 && NUM_PAGES <= 32); // for warp parallel processing and instruction_state_t padding
    static constexpr int DYNAMIC_SHARED_MEMORY = NUM_PAGES * PAGE_SIZE + DYNAMIC_SHARED_MEMORY_ALIGN;
    static constexpr int SCRATCH_BYTES = (kittens::MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY_BASE - DYNAMIC_SHARED_MEMORY) / INSTRUCTION_PIPE_STAGES;
    static constexpr int STATIC_SHARED_MEMORY = STATIC_SHARED_MEMORY_BASE + INSTRUCTION_PIPE_STAGES*SCRATCH_BYTES;

    // <NT> 自旋等待时的 sleep 时长 (ns)
    static constexpr int SPIN_LOOP_SLEEP_NS = 20;
};

// <NT> 指令状态结构, 128 字节对齐
//      每个流水线阶段有一个实例
template <typename Config>
struct __align__(128) instruction_state_t {
    // <NT> 指令数据 (256B, 但只占结构体一部分)
    instruction_t instruction;
    // <NT> 物理页面顺序 (lane 到 pid 的映射）
    int pid_order[Config::NUM_PAGES];
    // <NT> 对齐填充, 使 pid_order + padding = 128B
    int _padding[((Config::NUM_PAGES + 31) & ~31) - Config::NUM_PAGES]; // make pid_order + _padding 128 bytes
    kittens::semaphore semaphores[Config::DYNAMIC_SEMAPHORES];
    // <NT> scratch 内存, 用于指令间临时存储
    int scratch[Config::SCRATCH_BYTES/sizeof(int)];
};

// <NT> 共享内存页面结构，每个页面大小为 PAGE_SIZE (32KB), 用于存储 tile 数据
template <typename Config>
struct page_t {
    int data[Config::PAGE_SIZE/sizeof(int)];
    // <NT> 获取页面指针, 支持字节偏移
    __device__ __forceinline__ void *ptr(int byte_offset = 0) {
        return reinterpret_cast<void *>(reinterpret_cast<uint64_t>(&data[0])+byte_offset);
    }
    __device__ __forceinline__ const void *ptr(int byte_offset = 0) const {
        return reinterpret_cast<const void *>(reinterpret_cast<uint64_t>(&data[0])+byte_offset);
    }
    // <NT> 将页面数据强转为类型 T (安全检查: T 大小不超过页面大小)
    template <typename T>
    __device__ __forceinline__ T &as(int byte_offset = 0) {
        static_assert(sizeof(T) <= Config::PAGE_SIZE, "T exceeds page size"); // only guarantees safety for byte_offset=0
        return *reinterpret_cast<T*>(reinterpret_cast<uint64_t>(&data[0]) + byte_offset);
    }
    template <typename T>
    __device__ __forceinline__ const T &as(int byte_offset = 0) const {
        static_assert(sizeof(T) <= Config::PAGE_SIZE, "T exceeds page size"); // only guarantees safety for byte_offset=0
        return *reinterpret_cast<const T*>(reinterpret_cast<uint64_t>(&data[0]) + byte_offset);
    }
};

// <NT> MegaKittens 运行时状态, 包含所有共享数据和信号量
template <typename Config>
struct state_t {
    uint32_t iter;
    uint32_t stage;

    kittens::clc::handle (&clc_handle)[Config::INSTRUCTION_PIPE_STAGES];
    kittens::semaphore (&clc_arrived)[Config::INSTRUCTION_PIPE_STAGES];

    // <NT> 指令状态和同步信号量
    //      instruction_arrived：Controller 发送, Worker 等待
    //      instruction_finished：Worker 发送, Controller 等待
    instruction_state_t<Config> (&instruction_states)[Config::INSTRUCTION_PIPE_STAGES];
    kittens::semaphore (&instruction_arrived)[Config::INSTRUCTION_PIPE_STAGES];
    kittens::semaphore (&instruction_finished)[Config::INSTRUCTION_PIPE_STAGES];

    // <NT> 页面和页面完成信号量
    page_t<Config> (&pages)[Config::NUM_PAGES];
    kittens::semaphore (&page_finished)[Config::NUM_PAGES];

    // <NT> 张量完成信号量
    kittens::semaphore &tensor_finished;
    kittens::tensor_allocator<1, Config::CLUSTER_SIZE> &tensor_alloc;

    __device__ __forceinline__ const instruction_t &instruction() const {
        return instruction_states[stage].instruction;
    }
    __device__ __forceinline__ const int (&pid_order() const)[Config::NUM_PAGES] {
        return instruction_states[stage].pid_order;
    }
    __device__ __forceinline__ kittens::semaphore (&semaphores())[Config::DYNAMIC_SEMAPHORES] {
        return instruction_states[stage].semaphores;
    }
    __device__ __forceinline__ const kittens::semaphore (&semaphores() const)[Config::DYNAMIC_SEMAPHORES] {
        return instruction_states[stage].semaphores;
    }
    __device__ __forceinline__ void *scratch() const {
        return reinterpret_cast<void *>(&instruction_states[stage].scratch[0]);
    }
    __device__ __forceinline__ int lid_to_pid(int lid) {
        return pid_order()[lid];
    }
    __device__ __forceinline__ void page_wait(int pid) {
        kittens::wait(page_finished[pid], iter&0b1);
    }
    __device__ __forceinline__ void page_finish(int pid) {
        kittens::arrive(page_finished[pid]);
    }
    __device__ __forceinline__ void tensor_wait() {
        kittens::wait(tensor_finished, iter&0b1);
    }
    __device__ __forceinline__ void tensor_finish() {
        kittens::arrive(tensor_finished);
    }
};

} // namespace megakittens
