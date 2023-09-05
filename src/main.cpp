#include "emulator.h"
#include "control_flow.h"
#include "codegen.h"
#include "jit.h"

#include <llvm/Support/TargetSelect.h>

#include <spdlog/spdlog.h>
#include <fmt/chrono.h>

#include <cstdint>
#include <chrono>
#include <thread>

using namespace emu::literals;
using namespace std::literals;

llvm::ExitOnError exit_on_error;

namespace global {
emu::Emulator emulator;
}

int main()
{
    spdlog::set_level(spdlog::level::debug);

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    using namespace global;
    emulator.bus.load_file("/home/aesteve/git/6502/test-data/6502_functional_test.bin");
    emulator.cpu.PC = 0x400_w;

    llvm::orc::ThreadSafeContext context{std::make_unique<llvm::LLVMContext>()};

    static constexpr auto JIT_FN = 0x37ab_w;
    auto const flow = emu::build_control_flow(emulator.bus, JIT_FN);
    auto module = emu::codegen(context, flow);
    auto jit = exit_on_error(emu::make_jit());
    exit_on_error(emu::materialize(*jit, context, std::move(module)));

    static constexpr auto step_length = 0ms;
    static constexpr uint64_t step_cycles = 20000;

    uint64_t instruction_count = 0;
    auto const start = std::chrono::steady_clock::now();
    auto next_realtime = start + step_length;
    uint64_t next_cycles = step_cycles;

    while (true) {
        auto const prev_pc = emulator.cpu.PC;
        if (emulator.cpu.PC != JIT_FN) {
            emulator.run();
        } else {
            auto entry = exit_on_error(jit->lookup(fmt::format("fn_{:4x}", JIT_FN)));

            // Cast the entry point address to a function pointer.
            auto* jit_fn = (uint64_t(*)(emu::CPU&, emu::Bus&))entry.getValue();
            auto const cycles = jit_fn(emulator.cpu, emulator.bus);
            emulator.clock_counter += cycles;
        }
        ++instruction_count;
        if (prev_pc == emulator.cpu.PC) {
            spdlog::info("Stuck: {:#04x}", prev_pc);
            break;
        }

        if (emulator.clock_counter >= next_cycles) {
            next_cycles += step_cycles;
            auto const now = std::chrono::steady_clock::now();
            if (now < next_realtime) {
                std::this_thread::sleep_until(next_realtime);
                next_realtime += step_length;
            } else {
                next_realtime = now + step_length;
            }
        }
    }
    auto const end = std::chrono::steady_clock::now();

    spdlog::info("Total instructions: {}", instruction_count);
    spdlog::info("Total cycles: {}", emulator.clock_counter);
    spdlog::info("Real time: {}",
                 std::chrono::duration_cast<std::chrono::milliseconds>(end - start));
}
