#include "emulator.h"
#include "control_flow.h"
#include "codegen.h"
#include "jit.h"

#include <llvm/Support/TargetSelect.h>

#include <spdlog/spdlog.h>
#include <fmt/chrono.h>
#include <boost/program_options.hpp>

#include <cstdint>
#include <chrono>
#include <thread>
#include <iostream>

using namespace emu::literals;
using namespace std::literals;

llvm::ExitOnError exit_on_error;

namespace global {
emu::Emulator emulator;
}

namespace po = boost::program_options;

int main(int argc, char** argv)
{
    // clang-format off
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("bin-file,b",     po::value<std::string>()->required(),        "binary file image")
        ("entry-point,e",  po::value<uint16_t>()->default_value(0x400), "entry point address")
        ("load-address,a", po::value<uint16_t>()->default_value(0),     "binary file load address")
    ;
    // clang-format on

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
        std::cerr << desc << '\n';
        return 1;
    }

    try {
        po::notify(vm);
    } catch (po::error const& e) {
        std::cerr << e.what() << '\n';
        std::cerr << desc << '\n';
        return 1;
    }

    std::string const bin_file = vm["bin-file"].as<std::string>();
    auto const entry_point = static_cast<emu::word_t>(vm["entry-point"].as<uint16_t>());
    auto const load_address = static_cast<emu::word_t>(vm["load-address"].as<uint16_t>());

    spdlog::set_level(spdlog::level::debug);

    // llvm::InitializeNativeTarget();
    // llvm::InitializeNativeTargetAsmPrinter();

    using namespace global;
    emulator.bus.load_file(bin_file, load_address);
    emulator.cpu.PC = entry_point;

    // llvm::orc::ThreadSafeContext context{std::make_unique<llvm::LLVMContext>()};

    // static constexpr auto JIT_FN = 0x37ab_w;
    // auto const flow = emu::build_control_flow(emulator.bus, JIT_FN);
    // auto module = emu::codegen(context, flow);
    // auto jit = exit_on_error(emu::make_jit());
    // exit_on_error(emu::materialize(*jit, context, std::move(module)));

    static constexpr auto step_length = 0ms;
    static constexpr uint64_t step_cycles = 20000;

    uint64_t instruction_count = 0;
    auto const start = std::chrono::steady_clock::now();
    auto next_realtime = start + step_length;
    uint64_t next_cycles = step_cycles;

    try {
        while (true) {
            auto const prev_pc = emulator.cpu.PC;
            // if (emulator.cpu.PC != JIT_FN) {
            emulator.run();
            // } else {
            //     auto entry = exit_on_error(jit->lookup(fmt::format("fn_{:4x}", JIT_FN)));

            //     // Cast the entry point address to a function pointer.
            //     auto* jit_fn = (uint64_t(*)(emu::CPU&, emu::Bus&))entry.getValue();
            //     auto const cycles = jit_fn(emulator.cpu, emulator.bus);
            //     emulator.clock_counter += cycles;
            // }
            ++instruction_count;
            if (prev_pc == emulator.cpu.PC) {
                spdlog::info("Stuck: {:#04x}", prev_pc);
                break;
            }

            if (emulator.clock_counter >= next_cycles) {
                next_cycles += step_cycles;
                auto const now = std::chrono::steady_clock::now();
                if (now < next_realtime) {
                    std::this_thread::sleep_until(next_realtime);
                    next_realtime += step_length;
                } else {
                    next_realtime = now + step_length;
                }
            }
        }
    } catch (std::exception const& e) {
        spdlog::info("Exception: {}", e.what());
        spdlog::info("PC: {:#04x}", emulator.cpu.PC);
        for (emu::word_t i = 0_w; i < 0x20_w; ++i) {
            if (i % 0x10_w == 0_w) {
                fmt::print("\n{:04x}: ", i);
            } else if (i % 0x8_w == 0_w) {
                fmt::print(" ");
            }
            fmt::print("{:02x} ", emulator.bus.read(i));
        }
        fmt::print("\n\n");
    }
    auto const end = std::chrono::steady_clock::now();

    spdlog::info("Total instructions: {}", instruction_count);
    spdlog::info("Total cycles: {}", emulator.clock_counter);
    spdlog::info("Real time: {}",
                 std::chrono::duration_cast<std::chrono::milliseconds>(end - start));
}
