//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// template<class T> struct is_execution_policy;
// template<class T> constexpr bool is_execution_policy_v = is_execution_policy<T>::value;

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

#include <cuda/std/__execution_>

#include "test_macros.h"

static_assert(cuda::std::is_execution_policy<cuda::std::execution::sequenced_policy>::value, "");
static_assert(cuda::std::is_execution_policy<cuda::std::execution::parallel_policy_host>::value, "");
static_assert(cuda::std::is_execution_policy<cuda::std::execution::parallel_policy_device>::value, "");
static_assert(cuda::std::is_execution_policy<cuda::std::execution::parallel_unsequenced_policy_host>::value, "");
static_assert(cuda::std::is_execution_policy<cuda::std::execution::parallel_unsequenced_policy_device>::value, "");
static_assert(cuda::std::is_execution_policy<cuda::std::execution::unsequenced_policy_host>::value, "");
static_assert(cuda::std::is_execution_policy<cuda::std::execution::unsequenced_policy_device>::value, "");

#if TEST_STD_VER >= 2014 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
static_assert(cuda::std::is_execution_policy_v<cuda::std::execution::sequenced_policy>, "");
static_assert(cuda::std::is_execution_policy_v<cuda::std::execution::unsequenced_policy_host>, "");
static_assert(cuda::std::is_execution_policy_v<cuda::std::execution::unsequenced_policy_device>, "");
static_assert(cuda::std::is_execution_policy_v<cuda::std::execution::parallel_policy_host>, "");
static_assert(cuda::std::is_execution_policy_v<cuda::std::execution::parallel_policy_device>, "");
static_assert(cuda::std::is_execution_policy_v<cuda::std::execution::parallel_unsequenced_policy_host>, "");
static_assert(cuda::std::is_execution_policy_v<cuda::std::execution::parallel_unsequenced_policy_device>, "");
#endif // TEST_STD_VER >= 2014 && !_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES

int main(int, char**)
{
  return 0;
}
