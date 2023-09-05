#include "emulator.h"
#include "instruction.h"

#include <spdlog/spdlog.h>

namespace emu {

void Emulator::run()
{
    inst::Instruction const inst{cpu.PC, bus};
    SPDLOG_DEBUG("{}", inst.disassemble());
    clock_counter += inst.run(cpu, bus);
}

} // namespace emu
