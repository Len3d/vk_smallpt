// png_writer.h -- minimal, dependency-free PNG writer (8-bit RGB, no alpha).
//
// The sample host programs link nothing but the C++ standard library (and
// Vulkan), so instead of adding a third-party image library this module
// implements the small set of pieces PNG needs. See png_writer.cpp.

#pragma once

#include <cstdint>
#include <string>

// Writes an RGB PNG. `rgb` is width * height pixels, 3 bytes each, row-major,
// top row first. Returns false if the file could not be written.
bool write_png_rgb(const std::string &path, const std::uint8_t *rgb, std::uint32_t width,
                   std::uint32_t height);
