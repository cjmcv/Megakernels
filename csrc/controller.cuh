#pragma once

#include "kittens.cuh"

namespace megakittens {

// <NT> Controller 是指令流水线的调度器，负责按正确的顺序为每个流水线阶段准备指令、页面和信号量，并通知其他 worker 开始工作。
// 1. 获取指令 (Fetch) - 动态调度: 从 CLC 工作队列获取
//                     - 静态调度: 从预定义张量读取
// 2. 建立页面顺序 (Page Order) - 首轮: pid_order[lane] = lane_id
//                             - 后续: 根据上一指令的 page_manager 确定释放顺序
// 3. 初始化信号量 (Semaphores) - 调用各指令类型的 semaphore_manager
//                             - 确定需要多少信号量用于同步
// 4. 广播就绪信号 (Broadcast Ready) - 通知 loader/launcher/storer: 指令+页面+信号量已就绪
template <typename Config, typename Globals>
__device__ __forceinline__ void controller_loop(const Globals &g, megakittens::state_t<Config> &s) {
    // <NT> cta_rank: 当前CTA在cluster中的排名，如 cluster 设为8，有:
    //                # cluster 0: block 0-7,  cta_rank = 0,1,2,3,4,5,6,7
    //                # cluster 1: block 8-15, cta_rank = 0,1,2,3,4,5,6,7 
    //                用于 instruction_index = schedule.x + cta_rank; // 用 1D 网格
    //                schedule.x 是分配给当前 cluster 的起始指令索引，使不同rank拿到不同的指令。
    //                即指令的分配是按cluster分配的，所以需要使用rank来代替blockIdx.x
    //      lane_id: 当前线程在线程束的lane ID (0-31)
    //      num_semaphores: 每个流水线阶段需要的信号量数量
    //      last_stage: 上一个执行的流水线阶段
    const int cta_rank = ::kittens::cluster_ctarank();
    const int lane_id = ::kittens::laneid();
    int num_semaphores[Config::INSTRUCTION_PIPE_STAGES];
    int last_stage = -1;

    for (s.iter = 0, s.stage = 0; true; ++s.iter) {
        // Step 0. If this is not the first time the slot is being used, wait for the
        //         previous instruction to complete and invalidate its semaphores
        // <NT> 跨指令预取的关键
        //   iter	stage	if ≥ 2	             等待什么	                       实际预取情况
        //    0	     0	     false	                无	                       取 Inst 0，Loader 加载
        //    1	     1	     false	                无	                       取 Inst 1，Loader 加载 Inst 1 与 Inst 0 Consumer 并行
        //    2	     0	     true	inst_finished[0]（Loader Inst 0 已完成）	取 Inst 2，Loader 加载 Inst 2 与 Inst 1 Consumer 并行
        //    3	     1	     true	inst_finished[1]（Loader Inst 1 已完成）	取 Inst 3，Loader 加载 Inst 3 ...
        //    4 	 0	     true	inst_finished[0]（已完成）	                取 Inst 4，Loader 加载 Inst 4 ...
        //  Loader 不受 barrier 约束：只要收到 instruction_arrived，就开始加载下一指令的权重
        //  Consumer 受 barrier 约束：必须等 src_barriers arrive 才能开始计算
        //  stage 0 和 stage 1 独立：两个 stage 的 instruction_arrived 是不同的 semaphore
        // 所以：预取持续存在，Loader stage=1 和 Loader stage=0 交替工作。只要前一个 Loader 完成了，下一个 Loader 立即开始，不需要等 Consumer/Storer。
        //       即 Inst N 的 Loader 和 Inst N+1 的 Loader 总是能 overlap。
        if (s.iter >= Config::INSTRUCTION_PIPE_STAGES) {
            const int phasebit = ((s.iter - Config::INSTRUCTION_PIPE_STAGES) / Config::INSTRUCTION_PIPE_STAGES) & 0b1;
            // <NT> 等待上一条指令完成
            kittens::wait(s.instruction_finished[s.stage], phasebit);
            // <NT> 使该阶段的信号量失效，lane_id有32个，会超过数量，只取前面几个线程来执行即可。
            if (lane_id < num_semaphores[s.stage])
                invalidate_semaphore(s.instruction_states[s.stage].semaphores[lane_id]);
            kittens::warp::sync(); // invalidate_semaphore relies on the instruction
        }

        // Step 1. Fetch next instruction (mode-dependent)
        int *inst_src;
        if constexpr (Config::GLOBAL_WORK_QUEUE) {
            // <NT> 动态调度模式: 从全局工作队列获取指令索引
            int instruction_index;
            if (s.iter == 0) {
                // <NT> 首次迭代, 使用blockIdx.x作为指令索引，其实等价于cta_rank，
                //      但不能使用schedule.x + cta_rank，因为 s.clc_handle[s.stage] 在首轮时还没有初始化
                instruction_index = blockIdx.x;
            } else {
                // <NT> 非首次迭代, 从CLC(Compute Language Center)获取调度信息
                const int phasebit = ((s.iter - 1) / Config::INSTRUCTION_PIPE_STAGES) & 0b1;
                if (kittens::warp::elect_leader()) {
                    // <NT> 只有warp leader执行调度逻辑
                    if (cta_rank == 0) kittens::clc::schedule(s.clc_handle[s.stage], s.clc_arrived[s.stage]);
                    kittens::tma::expect_bytes(s.clc_arrived[s.stage], sizeof(s.clc_handle[s.stage]));
                }
                // <NT> 1. 等待调度完成
                //      2. 查询调度结果
                //      3. 如果调度失败, 发送停止信号
                //         如果调度成功，使用1D网格, 即使用cta_rank就够了
                kittens::wait(s.clc_arrived[s.stage], phasebit);
                auto schedule = kittens::clc::query(s.clc_handle[s.stage]);
                if (!schedule.success) instruction_index = 0x7FFFFFFF; // signal to stop
                else                   instruction_index = schedule.x + cta_rank; // we only use 1D grid
            }
            if (instruction_index >= g.instructions.rows()) {
                // <NT> 指令索引超出范围, 停止执行
                if (kittens::warp::elect_leader()) {
                    s.instruction_states[s.stage].instruction.icode = -1; // signal to stop.
                    kittens::tensor_commit<Config::CLUSTER_SIZE>(s.instruction_arrived[s.stage]); // hack: use tcgen05.commit for mbarrier broadcast
                }
                break;
            }
            inst_src = &g.instructions[{instruction_index, 0}];
        } else {
            // <NT> 静态调度模式: 指令已预定义在instructions张量中
            if ((int)s.iter >= g.instructions.depth()) {
                // <NT> 迭代次数超出预定义指令数, 停止执行, icode=-1表示停止
                if (kittens::warp::elect_leader()) {
                    s.instruction_states[s.stage].instruction.icode = -1; // signal to stop.
                    kittens::tensor_commit<Config::CLUSTER_SIZE>(s.instruction_arrived[s.stage]); // hack: use tcgen05.commit for mbarrier broadcast
                }
                break;
            }
            // <NT> 使用iter和blockIdx.x索引
            inst_src = &g.instructions[{(int)s.iter, (int)blockIdx.x, 0}];
        }
        // <NT> 确保指令大小为64个int (256字节) = 2个warp的加载量
        static_assert(sizeof(instruction_t)/sizeof(int) == 64); // 2 warp-wide loads
        int *inst_dst = reinterpret_cast<int*>(&s.instruction_states[s.stage].instruction);
        inst_dst[lane_id + 0]  = inst_src[lane_id + 0];
        inst_dst[lane_id + 32] = inst_src[lane_id + 32];
        kittens::warp::sync();

        // Step 2. Establish physical page order
        if (s.iter == 0) {
            // <NT> 首次迭代: 直接使用lane id作为页面顺序。
            //      NUM_PAGES = (MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY_BASE - DYNAMIC_SHARED_MEMORY_ALIGN) / PAGE_SIZE;
            //      * PAGE_SIZE = 32768 (32KB)
            //      * STATIC_SHARED_MEMORY_BASE = 静态基底（指令状态、信号量等）
            //      * DYNAMIC_SHARED_MEMORY_ALIGN = 1024 (1KB 对齐开销)
            //      * MAX_SHARED_MEMORY = GPU 架构的共享内存上限
            //      NUM_PAGES在blackwell里为7，而lane_id有32个，只取前面7个参与管理smem页面。
            if (lane_id < Config::NUM_PAGES)
                s.instruction_states[s.stage].pid_order[lane_id] = lane_id;
        } else {
            // <NT> 非首次迭代: 根据上一条指令的页面释放顺序确定当前指令的页面顺序
            const int last_icode = s.instruction_states[last_stage].instruction.icode;
            if (lane_id < Config::NUM_PAGES) {
                const uint32_t current_stage = s.stage;
                s.stage = last_stage; // so lid_release_order(...) can use s.instruction()
                const int lid = dispatch_instruction<WorkerType::page_manager, int, Config, Globals>(last_icode, g, s, lane_id);
                s.stage = current_stage;
                s.instruction_states[s.stage].pid_order[lane_id] = s.instruction_states[last_stage].pid_order[lid];
            }
        }

        // Step 3. Initialize dynamic semaphores
        const int icode = s.instruction_states[s.stage].instruction.icode;
        num_semaphores[s.stage] = dispatch_instruction<WorkerType::semaphore_manager, int, Config, Globals>(icode, g, s);
        asm volatile("{fence.proxy.async.shared::cta;\n}" ::: "memory"); // TODO: is this really needed?

        // Step 4. Signal other workers that the instruction/pages/semaphores are ready
        kittens::warp::sync();
        if (kittens::warp::elect_leader())
            kittens::tensor_commit<Config::CLUSTER_SIZE>(s.instruction_arrived[s.stage]); // hack: use tcgen05.commit for mbarrier broadcast

        // Update bookkeeping variables
        // <NT> 环形 advance 到下一个阶段
        last_stage = s.stage;
        s.stage = kittens::ring_advance<Config::INSTRUCTION_PIPE_STAGES>(s.stage);
    }
}

} // namespace megakittens
