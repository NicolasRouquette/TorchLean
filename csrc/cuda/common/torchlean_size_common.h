#pragma once

#include <lean/lean.h>

#include <limits.h>
#include <stdint.h>
#include <stddef.h>

// Shared checked size arithmetic for CUDA kernels and CPU stubs.
//
// Native wrappers receive many dimensions from Lean as `uint32_t`/`size_t` and then multiply them
// before allocating or validating flat buffers. Keep the overflow policy in one C-safe header so
// CUDA and non-CUDA builds reject impossible sizes in the same way.

static inline size_t checked_mul_size(size_t a, size_t b, const char* msg) {
  if (a != 0 && b > SIZE_MAX / a) {
    lean_internal_panic(msg);
  }
  return a * b;
}

static inline size_t checked_mul3_size(size_t a, size_t b, size_t c, const char* msg) {
  return checked_mul_size(checked_mul_size(a, b, msg), c, msg);
}

static inline uint64_t torchlean_float_bytes_for(size_t count) {
  const uint64_t max = UINT64_MAX / (uint64_t)sizeof(float);
  if ((uint64_t)count > max) {
    return UINT64_MAX;
  }
  return (uint64_t)count * (uint64_t)sizeof(float);
}
