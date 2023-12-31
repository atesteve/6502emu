#pragma once

#include "control_flow.h"

#include <llvm/IR/Module.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/Passes/OptimizationLevel.h>

#include <memory>

namespace emu {

std::unique_ptr<llvm::Module> codegen(llvm::orc::ThreadSafeContext context,
                                      std::map<word_t, control_block> const& flow);

void optimize(llvm::Module& module, llvm::OptimizationLevel level);

} // namespace emu
