#include "emulator.h"
#include "control_flow.h"
#include "codegen.h"
#include "jit.h"

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

    using namespace global;
    emulator.initialize(bin_file, load_address, entry_point);

    uint64_t instruction_count = 0;
    auto const start = std::chrono::steady_clock::now();

    try {
        emulator.call_function(entry_point);
        while (true) {
            emulator.run();
        }
    } catch (std::exception const& e) {
        spdlog::info("Exception: {}", e.what());
        spdlog::info("PC: {:#04x}", emulator.get_cpu().PC);
        for (emu::word_t i = 0_w; i < 0x20_w; ++i) {
            if (i % 0x10_w == 0_w) {
                fmt::print("\n{:04x}: ", i);
            } else if (i % 0x8_w == 0_w) {
                fmt::print(" ");
            }
            fmt::print("{:02x} ", emulator.get_bus().read_memory(i));
        }
        fmt::print("\n\n");
    }
    auto const end = std::chrono::steady_clock::now();

    spdlog::info("Total instructions: {}", instruction_count);
    // spdlog::info("Total cycles: {}", emulator.get_clock_counter());
    spdlog::info("Real time: {}",
                 std::chrono::duration_cast<std::chrono::milliseconds>(end - start));
}
