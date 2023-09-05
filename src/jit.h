#pragma once

#include <llvm/IR/Module.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>

#include <memory>

namespace emu {
llvm::Expected<std::unique_ptr<llvm::orc::LLJIT>> make_jit();
llvm::Error materialize(llvm::orc::LLJIT& jit,
                        llvm::orc::ThreadSafeContext tsc,
                        std::unique_ptr<llvm::Module> module);
} // namespace emu
