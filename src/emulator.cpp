#include "emulator.h"
#include "instruction.h"
#include "control_flow.h"
#include "codegen.h"

#include <llvm/Support/TargetSelect.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>

#include <spdlog/spdlog.h>
#include <fmt/chrono.h>

#include <chrono>
#include <array>

namespace emu {
namespace {
llvm::ExitOnError exit_on_error;
} // namespace

struct JitFn {
    llvm::orc::ThreadSafeContext llvm_context = std::make_unique<llvm::LLVMContext>();
    std::unique_ptr<llvm::orc::LLJIT> jit = exit_on_error(emu::make_jit());
    std::map<word_t, control_block> flow;
    std::unique_ptr<llvm::Module> module;
};

Emulator::Emulator(int optimization_level)
    : _jit_functions(size_t{0x10000})
    , _jit_functions_cache(size_t{0x10000})
    , _thread_pool{std::make_unique<llvm::StdThreadPool>()}
    , _optimization_level{optimization_level}
{}

Emulator::~Emulator() {}

uint64_t Emulator::run()
{
    uint64_t counter = 0;
    while (true) {
        if (_jit_functions_cache[static_cast<size_t>(_cpu.PC)] != nullptr) {
            _intr_clock_counter += counter;
            return call_function(_cpu.PC);
        }
        inst::Instruction const inst{_cpu.PC, _bus};
        counter += inst.run(_cpu, _bus);
        if (inst.is_call()) {
            auto const next_pc = inst.pc + word_t{inst.length};
            call_function(_cpu.PC);
            if (_cpu.PC != next_pc) {
                _intr_clock_counter += counter;
                return counter;
            }
        } else if (inst.is_indirect_jump()) {
            _intr_clock_counter += counter;
            return call_function(_cpu.PC);
        } else if (inst.is_return()) {
            _intr_clock_counter += counter;
            return counter;
        }
    }
}

template<typename Fn>
void Emulator::async(Fn&& fn)
{
    _thread_pool->async(std::forward<Fn>(fn));
}

void Emulator::jit_function(word_t addr)
{
    auto const index = static_cast<size_t>(addr);
    JitFn* jit_fn_ptr;

    {
        std::lock_guard lock{_map_mutex};
        auto& ptr = _jit_functions[index];
        if (ptr != nullptr) {
            return;
        }
        ptr = std::make_unique<JitFn>();
        jit_fn_ptr = ptr.get();
    }

    spdlog::info("Starting jit {:#04x}", addr);
    auto const start = std::chrono::steady_clock::now();

    JitFn& jit_fn = *jit_fn_ptr;

    std::unordered_set<word_t> function_calls;
    jit_fn.flow = emu::build_control_flow(_bus, addr, &function_calls);

    auto const finish_control_flow = std::chrono::steady_clock::now();

    for (auto call_addr : function_calls) {
        {
            std::lock_guard lock{_map_mutex};
            if (_jit_functions[static_cast<size_t>(call_addr)] != nullptr) {
                continue;
            }
        }
        async([=, this] { jit_function(call_addr); });
    }

    jit_fn.module = emu::codegen(jit_fn.llvm_context, jit_fn.flow);
    auto const finish_codegen = std::chrono::steady_clock::now();

    emu::optimize(*jit_fn.module,
                  std::array{llvm::OptimizationLevel::O0,
                             llvm::OptimizationLevel::O1,
                             llvm::OptimizationLevel::O2,
                             llvm::OptimizationLevel::O3}
                      .at(_optimization_level));

    auto const finish_optimize = std::chrono::steady_clock::now();

    exit_on_error(emu::materialize(*jit_fn.jit, jit_fn.llvm_context, std::move(jit_fn.module)));

    auto const fn_addr = exit_on_error(jit_fn.jit->lookup(fmt::format("fn_{:04x}", addr)));
    auto const finish_materialize = std::chrono::steady_clock::now();
    auto* const fn = reinterpret_cast<jit_fn_t>(fn_addr.getValue());

    _jit_functions_cache[index] = fn;
    ++_jit_counter;

    auto const finish = std::chrono::steady_clock::now();
    spdlog::info(
        "Finished jit {:#04x} (total: {}, flow: {}, codegen: {}, optimize: {}, materialize: {})",
        addr,
        duration_cast<std::chrono::milliseconds>(finish - start),
        duration_cast<std::chrono::milliseconds>(finish_control_flow - start),
        duration_cast<std::chrono::milliseconds>(finish_codegen - finish_control_flow),
        duration_cast<std::chrono::milliseconds>(finish_optimize - finish_codegen),
        duration_cast<std::chrono::milliseconds>(finish_materialize - finish_optimize));
}

JitFn* Emulator::get_jit_fn(word_t addr)
{
    std::lock_guard lock{_map_mutex};
    return _jit_functions[static_cast<size_t>(addr)].get();
}

uint64_t Emulator::call_function(word_t addr)
{
    if (_optimization_level < 0) {
        return run();
    }

    auto const index = static_cast<size_t>(addr);
    jit_fn_t const cache_fn = _jit_functions_cache[index];
    if (cache_fn != nullptr) {
        auto const ret = cache_fn(&_cpu, &_bus, _bus.memory_space.data(), this);
        _jit_clock_counter += ret;
        return ret;
    }

    auto* jit_fn = get_jit_fn(addr);

    if (jit_fn == nullptr) {
        async([this, addr] { jit_function(addr); });
    }

    // Run interpreter while the function is being compiled
    return run();
}

void Emulator::initialize(std::string_view image_file, word_t load_address, word_t entry_point)
{
    _bus.load_file(image_file, load_address);
    _cpu.PC = entry_point;

    if (_optimization_level < 0) {
        return;
    }

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    jit_function(entry_point);
    _thread_pool->wait();
}

} // namespace emu
