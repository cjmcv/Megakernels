#pragma once

#include "kittens.cuh"

namespace megakittens {

// <NT> 宏生成 loader/launcher/consumer/storer 的主循环
//      这四种 worker 的执行流程相同, 只是 dispatch_instruction 的 WorkerType 不同
#define MAKE_WORKER(name)                                                                        \
template <typename Config, typename Globals>                                                     \
__device__ __forceinline__ void name##_loop(const Globals &g, megakittens::state_t<Config> &s) { \
    for (s.iter = 0, s.stage = 0; true; ++s.iter) {                                              \
        // <NT> 计算 phasebit 用于 double buffering 同步
        const int phasebit = (s.iter / Config::INSTRUCTION_PIPE_STAGES) & 0b1;                   \
        // <NT> 等待 controller 发送指令就绪信号
        kittens::wait(s.instruction_arrived[s.stage], phasebit);                                 \
        // <NT> 获取当前指令的操作码
        const int icode = s.instruction_states[s.stage].instruction.icode;                       \
        // <NT> icode == -1 表示停止信号, 退出循环 (stop signal)
        if (icode == -1) break;                                                                  \
        // <NT> 分发到具体的 worker handle
        dispatch_instruction<WorkerType::name, void, Config, Globals>(icode, g, s);              \
        // <NT> 同步所有线程
        kittens::warp::sync();                                                                   \
        // <NT> 只有 warp leader 发送完成信号, 通知 controller 本指令已完成
        if (kittens::warp::elect_leader())                                                       \
            kittens::arrive(s.instruction_finished[s.stage]);                                    \
        // <NT> 环形 advance 到下一个流水线阶段
        s.stage = kittens::ring_advance<Config::INSTRUCTION_PIPE_STAGES>(s.stage);               \
    }                                                                                            \
}

// <NT> 四种 worker 各自独立运行, 但都遵循相同的流水线协议:
//      1. 等待指令到达 (instruction_arrived)
//      2. 执行具体操作 (loader/launcher/consumer/storer)
//      3. 发送完成信号 (instruction_finished)
//      4. 推进流水线阶段 (stage)
MAKE_WORKER(loader)
MAKE_WORKER(launcher)
MAKE_WORKER(consumer)
MAKE_WORKER(storer)

} // namespace megakittens
