#include "emulator.h"
#include "jit.h"

#include <spdlog/spdlog.h>
#include <fmt/chrono.h>
#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>

#include <cstdint>
#include <chrono>
#include <iostream>

using namespace emu::literals;
using namespace std::literals;

namespace po = boost::program_options;

namespace {
llvm::ExitOnError exit_on_error;

} // namespace

std::unique_ptr<emu::Emulator> global_emu;

struct Opt {
    int level;

    friend void validate(boost::any& v, const std::vector<std::string>& values, Opt*, int)
    {
        po::validators::check_first_occurrence(v);
        const std::string& s = po::validators::get_single_string(values);

        auto level = boost::lexical_cast<int>(s);
        level = std::min(level, 3);
        level = std::max(level, -1);
        v = Opt{level};
    }

    friend std::ostream& operator<<(std::ostream& os, Opt const& arg) { return os << arg.level; }
};

int main(int argc, char** argv)
{
    // clang-format off
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("bin-file,b",     po::value<std::string>()->required(),        "binary file image")
        ("entry-point,e",  po::value<uint16_t>()->default_value(0x400), "entry point address")
        ("load-address,a", po::value<uint16_t>()->default_value(0),     "binary file load address")
        ("opt,O",          po::value<Opt>()->default_value(Opt{0}),     "JIT optimization level. -1 means no JIT")
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
    auto const optimization_level = vm["opt"].as<Opt>().level;

    spdlog::set_level(spdlog::level::debug);

    global_emu = std::make_unique<emu::Emulator>(optimization_level);
    global_emu->initialize(bin_file, load_address, entry_point);

    uint64_t instruction_count = 0;
    auto const start = std::chrono::steady_clock::now();

    try {
        global_emu->call_function(entry_point);
        while (true) {
            global_emu->run();
        }
    } catch (std::exception const& e) {
        spdlog::info("Exception: {}", e.what());
        spdlog::info("PC: {:#04x}", global_emu->get_cpu().PC);
        for (emu::word_t i = 0_w; i < 0x20_w; ++i) {
            if (i % 0x10_w == 0_w) {
                fmt::print("\n{:04x}: ", i);
            } else if (i % 0x8_w == 0_w) {
                fmt::print(" ");
            }
            fmt::print("{:02x} ", global_emu->get_bus().read_memory(i));
        }
        fmt::print("\n\n");
    }
    auto const end = std::chrono::steady_clock::now();

    spdlog::info("Total instructions: {}", instruction_count);
    // spdlog::info("Total cycles: {}", emulator.get_clock_counter());
    spdlog::info("Real time: {}",
                 std::chrono::duration_cast<std::chrono::milliseconds>(end - start));
}
