#include "emulator.h"
#include "instruction.h"
#include "control_flow.h"
#include "codegen.h"
#include "jit.h"

#include <llvm/Support/TargetSelect.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>

#include <spdlog/spdlog.h>

namespace emu {
namespace {
llvm::ExitOnError exit_on_error;
} // namespace

Emulator::Emulator() {}
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
            counter += call_function(_cpu.PC);
            if (_cpu.PC != next_pc) {
                return counter;
            }
        } else if (inst.is_return()) {
            return counter;
        }
    }
    inst::Instruction const inst{_cpu.PC, _bus};
    SPDLOG_DEBUG("{}", inst.disassemble());
    _clock_counter += inst.run(_cpu, _bus);
    return 0;
}

struct JitFn {
    llvm::orc::ThreadSafeContext llvm_context = std::make_unique<llvm::LLVMContext>();
    std::unique_ptr<llvm::orc::LLJIT> jit = exit_on_error(emu::make_jit());
    std::map<word_t, control_block> flow;
    std::unique_ptr<llvm::Module> module;
    std::mutex mutex;
    jit_fn_t fn = nullptr;
};

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
    SPDLOG_DEBUG("Calling fn {:04x}", addr);
    uint64_t ret = 0;
    auto* jit_fn = get_jit_fn(addr);

    if (jit_fn == nullptr) {
        ret = run();
    } else {
        jit_fn_t fn = nullptr;
        {
            std::lock_guard lock{jit_fn->mutex};
            fn = jit_fn->fn;
        }
        if (!fn) {
            ret = run();
        } else {
            ret = fn(_cpu, _bus, *this);
        }
    }

    return ret;
}

void Emulator::initialize(std::string_view image_file, word_t load_address, word_t entry_point)
{
    _bus.load_file(image_file, load_address);
    _cpu.PC = entry_point;

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    auto& jit_fn = *_jit_functions.emplace(entry_point, std::make_unique<JitFn>()).first->second;

    std::unordered_set<word_t> function_calls;

    jit_fn.flow = emu::build_control_flow(_bus, entry_point, &function_calls);
    jit_fn.module = emu::codegen(jit_fn.llvm_context, jit_fn.flow);
    exit_on_error(emu::materialize(*jit_fn.jit, jit_fn.llvm_context, std::move(jit_fn.module)));
    auto const fn = exit_on_error(jit_fn.jit->lookup(fmt::format("fn_{:04x}", entry_point)));
    jit_fn.fn = reinterpret_cast<jit_fn_t>(fn.getValue());

    size_t prev_size = 0;
    while (prev_size != function_calls.size()) {
        prev_size = function_calls.size();
        for (auto fn_addr : function_calls) {
            auto& jit = _jit_functions[fn_addr];
            if (jit != nullptr) {
                continue;
            }
            jit = std::make_unique<JitFn>();

            jit->flow = emu::build_control_flow(_bus, fn_addr, &function_calls);
            jit->module = emu::codegen(jit->llvm_context, jit->flow);
            exit_on_error(emu::materialize(*jit->jit, jit->llvm_context, std::move(jit->module)));
            auto const fn = exit_on_error(jit->jit->lookup(fmt::format("fn_{:04x}", fn_addr)));
            jit->fn = reinterpret_cast<jit_fn_t>(fn.getValue());
        }
    }

    // auto jit = exit_on_error(emu::make_jit());
    // exit_on_error(emu::materialize(*jit, context, std::move(module)));
}

} // namespace emu
