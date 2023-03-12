/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#pragma once

#include <absl/base/thread_annotations.h>

// Enable thread safety attributes only with clang.
// The attributes can be safely erased when compiling with other compilers.
#if defined(__clang__) && (!defined(SWIG))
#define ATTRIBUTE__(x) __attribute__((x))
#else
#define ATTRIBUTE__(x)  // no-op
#endif

#define CAPABILITY(x) ATTRIBUTE__(capability(x))
#define SCOPED_CAPABILITY ATTRIBUTE__(scoped_lockable)
#define ACQUIRE(...) ATTRIBUTE__(acquire_capability(__VA_ARGS__))
#define RELEASE(...) ATTRIBUTE__(release_capability(__VA_ARGS__))
#define ACQUIRE_SHARED(...) ATTRIBUTE__(acquire_shared_capability(__VA_ARGS__))
#define RELEASE_SHARED(...) ATTRIBUTE__(release_shared_capability(__VA_ARGS__))
#define REQUIRES(...) ATTRIBUTE__(requires_capability(__VA_ARGS__))
#define REQUIRES_SHARED(...) \
  ATTRIBUTE__(requires_shared_capability(__VA_ARGS__))
#define PACKED_STRUCT ATTRIBUTE__(packed)
