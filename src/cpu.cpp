#include "cpu.h"

#include <fstream>
#include <algorithm>

namespace emu {

Bus::Bus()
    : memory_space(std::size_t{0x10000})
{}

void Bus::load_file(std::filesystem::path const& p)
{
    if (std::filesystem::file_size(p) != 0x10000) {
        throw std::runtime_error{"Incorrect file size"};
    }
    std::ifstream in{p};
    std::vector<char> buffer(4096);
    std::size_t space_i = 0;
    while (in) {
        in.read(buffer.data(), buffer.size());
        auto const nread = in.gcount();
        std::transform(buffer.begin(),
                       buffer.begin() + nread,
                       memory_space.begin() + space_i,
                       [](char c) { return byte_t{c}; });
        space_i += nread;
    }
}

byte_t Bus::read(word_t address) const { return memory_space[static_cast<size_t>(address)]; }

void Bus::write(word_t address, byte_t value)
{
    memory_space[static_cast<size_t>(address)] = value;
}

} // namespace emu