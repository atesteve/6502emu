#pragma once

#include "control_flow.h"

#include <llvm/IR/Module.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>

#include <memory>

namespace emu {

std::unique_ptr<llvm::Module> codegen(llvm::orc::ThreadSafeContext context,
                                      std::map<word_t, control_block> const& flow);

}
