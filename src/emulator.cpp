#include "emulator.h"
#include "instruction.h"
#include "control_flow.h"
#include "codegen.h"

#include <llvm/Support/TargetSelect.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>

#include <spdlog/spdlog.h>

#include <atomic>

namespace emu {
namespace {
llvm::ExitOnError exit_on_error;
} // namespace

Emulator::Emulator()
    : _thread_pool{std::make_unique<llvm::ThreadPool>()}
{}

Emulator::~Emulator() {}

uint64_t Emulator::run()
{
    uint64_t counter = 0;
    while (true) {
        inst::Instruction const inst{_cpu.PC, _bus};
        // SPDLOG_DEBUG("{}", inst.disassemble());
        counter += inst.run(_cpu, _bus);
        if (inst.is_call()) {
            auto const next_pc = inst.pc + word_t{inst.length};
            call_function(_cpu.PC);
            if (_cpu.PC != next_pc) {
                _intr_clock_counter += counter;
                return counter;
            }
        } else if (inst.is_return()) {
            _intr_clock_counter += counter;
            return counter;
        }
    }
}

struct JitFn {
    llvm::orc::ThreadSafeContext llvm_context = std::make_unique<llvm::LLVMContext>();
    std::unique_ptr<llvm::orc::LLJIT> jit = exit_on_error(emu::make_jit());
    std::map<word_t, control_block> flow;
    std::unique_ptr<llvm::Module> module;
    std::atomic<jit_fn_t> fn = nullptr;
};

void Emulator::jit_function(word_t addr)
{
    std::pair<decltype(_jit_functions)::iterator, bool> it;
    {
        std::lock_guard lock{_map_mutex};
        it = _jit_functions.try_emplace(addr, nullptr);
    }

    if (!it.second) {
        return;
    }

    it.first->second = std::make_unique<JitFn>();
    JitFn& jit_fn = *it.first->second;

    std::unordered_set<word_t> function_calls;

    for (auto call_addr : function_calls) {
        _thread_pool->async([this, call_addr] { jit_function(call_addr); });
    }

    jit_fn.flow = emu::build_control_flow(_bus, addr, &function_calls);
    jit_fn.module = emu::codegen(jit_fn.llvm_context, jit_fn.flow);
    exit_on_error(emu::materialize(*jit_fn.jit, jit_fn.llvm_context, std::move(jit_fn.module)));
    auto const fn_addr = exit_on_error(jit_fn.jit->lookup(fmt::format("fn_{:04x}", addr)));
    auto* const fn = reinterpret_cast<jit_fn_t>(fn_addr.getValue());
    jit_fn.fn = fn;
}

JitFn* Emulator::get_jit_fn(word_t addr)
{
    std::lock_guard lock{_map_mutex};
    auto it = _jit_functions.find(addr);
    if (it == _jit_functions.end()) {
        return nullptr;
    }
    return it->second.get();
}

uint64_t Emulator::call_function(word_t addr)
{
    auto const cache_fn = _jit_functions_cache.find(addr);
    if (cache_fn != _jit_functions_cache.end()) {
        auto const ret = cache_fn->second(_cpu, _bus, _bus.memory_space.data(), *this);
        _jit_clock_counter += ret;
        return ret;
    }

    auto* jit_fn = get_jit_fn(addr);

    if (jit_fn == nullptr) {
        _thread_pool->async([this, addr] { jit_function(addr); });
        // Run interpreter while the function is being compiled
        return run();
    } else {
        jit_fn_t fn = jit_fn->fn;

        if (!fn) {
            return run();
        } else {
            _jit_functions_cache.emplace(addr, fn);
            auto const ret = fn(_cpu, _bus, _bus.memory_space.data(), *this);
            _jit_clock_counter += ret;
            return ret;
        }
    }
}

void Emulator::initialize(std::string_view image_file, word_t load_address, word_t entry_point)
{
    _bus.load_file(image_file, load_address);
    _cpu.PC = entry_point;

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    jit_function(entry_point);
    _thread_pool->wait();
}

} // namespace emu
