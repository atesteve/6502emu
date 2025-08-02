#include "cpu.h"

#include <spdlog/spdlog.h>
#include <fmt/chrono.h>

#include <fstream>
#include <algorithm>
#include <iostream>
#include <chrono>

namespace emu {

using namespace literals;

Bus::Bus()
    : memory_space(std::size_t{0x10000})
{}

void Bus::load_file(std::filesystem::path const& p, word_t address)
{
    std::ifstream in{p};
    std::vector<char> buffer(4096);
    std::size_t space_i = static_cast<size_t>(address);
    while (in && space_i < 0x10000) {
        in.read(buffer.data(), buffer.size());
        auto const nread = in.gcount();
        auto const to_copy = std::min<size_t>(nread, 0x10000 - space_i);
        std::transform(buffer.begin(),
                       buffer.begin() + to_copy,
                       memory_space.begin() + space_i,
                       [](char c) { return byte_t{c}; });
        space_i += nread;
    }
}

byte_t Bus::read(word_t address) const noexcept
{
    if (address == 0xf004_w) {
        static std::chrono::steady_clock::time_point prev{};
        auto const diff = std::chrono::steady_clock::now() - prev;
        spdlog::info("Time: {}", std::chrono::duration_cast<std::chrono::milliseconds>(diff));

        byte_t const ret{std::cin.get()};
        if (ret == 0x65_b) {
            ::exit(0);
        }
        prev = std::chrono::steady_clock::now();
        return ret;
    }
    return memory_space[static_cast<size_t>(address)];
}

void Bus::write(word_t address, byte_t value) noexcept
{
    if (address == 0xf001_w) {
        std::cout << static_cast<char>(value);
        std::cout.flush();
    }
    memory_space[static_cast<size_t>(address)] = value;
}

byte_t Bus::read_memory(word_t address) const noexcept
{
    return memory_space[static_cast<size_t>(address)];
}

void Bus::write_memory(word_t address, byte_t value) noexcept
{
    memory_space[static_cast<size_t>(address)] = value;
}

} // namespace emu