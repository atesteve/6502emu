#pragma once

#include "control_flow.h"

#include <llvm/IR/Module.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/Passes/OptimizationLevel.h>

#include <memory>

namespace emu {

std::unique_ptr<llvm::Module> codegen(llvm::orc::ThreadSafeContext context,
                                      std::map<word_t, control_block> const& flow,
                                      llvm::orc::ThreadSafeContext base_context,
                                      llvm::Module& base_module);

std::unique_ptr<llvm::Module> build_base(llvm::orc::ThreadSafeContext tsc);

void optimize(llvm::Module& module, llvm::OptimizationLevel level);

} // namespace emu
