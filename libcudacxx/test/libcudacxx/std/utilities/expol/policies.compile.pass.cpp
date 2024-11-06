//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// class sequenced_policy;
// class parallel_policy;
// class parallel_unsequenced_policy;
// class unsequenced_policy; // since C++20
//
// inline constexpr sequenced_policy seq = implementation-defined;
// inline constexpr parallel_policy par = implementation-defined;
// inline constexpr parallel_unsequenced_policy par_unseq = implementation-defined;
// inline constexpr unsequenced_policy unseq = implementation-defined; // since C++20

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

#include <cuda/std/__execution_>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
using remove_cvref_t = cuda::std::__remove_cvref_t<T>;
static_assert(
  cuda::std::is_same<remove_cvref_t<decltype(cuda::std::execution::seq)>, cuda::std::execution::sequenced_policy>::value,
  "");
static_assert(cuda::std::is_same<remove_cvref_t<decltype(cuda::std::execution::par_host)>,
                                 cuda::std::execution::parallel_policy_host>::value,
              "");
static_assert(cuda::std::is_same<remove_cvref_t<decltype(cuda::std::execution::par_device)>,
                                 cuda::std::execution::parallel_policy_device>::value,
              "");
static_assert(cuda::std::is_same<remove_cvref_t<decltype(cuda::std::execution::par_unseq_host)>,
                                 cuda::std::execution::parallel_unsequenced_policy_host>::value,
              "");
static_assert(cuda::std::is_same<remove_cvref_t<decltype(cuda::std::execution::par_unseq_device)>,
                                 cuda::std::execution::parallel_unsequenced_policy_device>::value,
              "");
static_assert(cuda::std::is_same<remove_cvref_t<decltype(cuda::std::execution::unseq_host)>,
                                 cuda::std::execution::unsequenced_policy_host>::value,
              "");
static_assert(cuda::std::is_same<remove_cvref_t<decltype(cuda::std::execution::unseq_device)>,
                                 cuda::std::execution::unsequenced_policy_device>::value,
              "");

int main(int, char**)
{
  unused(cuda::std::execution::seq);
  unused(cuda::std::execution::par_host);
  unused(cuda::std::execution::par_device);
  unused(cuda::std::execution::par_unseq_host);
  unused(cuda::std::execution::par_unseq_device);
  unused(cuda::std::execution::unseq_host);
  unused(cuda::std::execution::unseq_device);

  return 0;
}
