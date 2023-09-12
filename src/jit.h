#pragma once

#include "cpu.h"

#include <llvm/IR/Module.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>

#include <memory>
#include <cstdint>

namespace emu {

class Emulator;

using jit_fn_t = uint64_t (*)(CPU& cpu, Bus& bus, Emulator& em);

llvm::Expected<std::unique_ptr<llvm::orc::LLJIT>> make_jit();
llvm::Error materialize(llvm::orc::LLJIT& jit,
                        llvm::orc::ThreadSafeContext tsc,
                        std::unique_ptr<llvm::Module> module);
} // namespace emu
