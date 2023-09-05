#include "jit.h"
#include "cpu.h"

#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/JITSymbol.h>

#include <spdlog/spdlog.h>

namespace emu {
namespace {

extern "C" uint8_t read_bus(Bus* bus, uint16_t address)
{
    // Forward
    return static_cast<uint8_t>(bus->read(word_t{address}));
}

extern "C" void write_bus(Bus* bus, uint16_t address, uint8_t value)
{
    // Forward
    bus->write(word_t{address}, byte_t{value});
}

} // namespace

llvm::Expected<std::unique_ptr<llvm::orc::LLJIT>> make_jit()
{
    // Try to detect the host arch and construct an LLJIT instance.
    auto jit = llvm::orc::LLJITBuilder().create();

    // If we could not construct an instance, return an error.
    if (!jit) {
        return jit;
    }

    auto& jit_ptr = *jit;
    auto& jd = jit_ptr->getMainJITDylib();
    auto err = jd.define(llvm::orc::absoluteSymbols(llvm::orc::SymbolMap{
            {jit_ptr->mangleAndIntern("read_bus"),
             llvm::JITEvaluatedSymbol::fromPointer(
                     read_bus, llvm::JITSymbolFlags::Callable | llvm::JITSymbolFlags::Exported)},
            {jit_ptr->mangleAndIntern("write_bus"),
             llvm::JITEvaluatedSymbol::fromPointer(
                     write_bus, llvm::JITSymbolFlags::Callable | llvm::JITSymbolFlags::Exported)},
    }));

    return jit;
}

llvm::Error materialize(llvm::orc::LLJIT& jit,
                        llvm::orc::ThreadSafeContext tsc,
                        std::unique_ptr<llvm::Module> module)
{
    // Add the module.
    if (auto Err = jit.addIRModule(llvm::orc::ThreadSafeModule{std::move(module), tsc})) {
        llvm::logAllUnhandledErrors(std::move(Err), llvm::errs());
        return Err;
    }

    return llvm::Error::success();
}
} // namespace emu
