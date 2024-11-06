//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_EXECUTION_POLICY_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_EXECUTION_POLICY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__execution/policy.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cvref.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class>
struct is_execution_policy : false_type
{};

template <>
struct is_execution_policy<_CUDA_VEXEC::sequenced_policy> : true_type
{};

template <>
struct is_execution_policy<_CUDA_VEXEC::parallel_policy_host> : true_type
{};

template <>
struct is_execution_policy<_CUDA_VEXEC::parallel_policy_device> : true_type
{};

template <>
struct is_execution_policy<_CUDA_VEXEC::parallel_unsequenced_policy_host> : true_type
{};

template <>
struct is_execution_policy<_CUDA_VEXEC::parallel_unsequenced_policy_device> : true_type
{};

template <>
struct is_execution_policy<_CUDA_VEXEC::unsequenced_policy_host> : true_type
{};

template <>
struct is_execution_policy<_CUDA_VEXEC::unsequenced_policy_device> : true_type
{};

template <class>
struct __is_parallel_execution_policy_impl : false_type
{};

template <>
struct __is_parallel_execution_policy_impl<_CUDA_VEXEC::parallel_policy_host> : true_type
{};

template <>
struct __is_parallel_execution_policy_impl<_CUDA_VEXEC::parallel_policy_device> : true_type
{};

template <>
struct __is_parallel_execution_policy_impl<_CUDA_VEXEC::parallel_unsequenced_policy_host> : true_type
{};

template <>
struct __is_parallel_execution_policy_impl<_CUDA_VEXEC::parallel_unsequenced_policy_device> : true_type
{};

template <class _Tp>
struct __is_parallel_execution_policy : __is_parallel_execution_policy_impl<__remove_cvref_t<_Tp>>
{};

template <class>
struct __is_unsequenced_execution_policy_impl : false_type
{};

template <>
struct __is_unsequenced_execution_policy_impl<_CUDA_VEXEC::parallel_unsequenced_policy_host> : true_type
{};

template <>
struct __is_unsequenced_execution_policy_impl<_CUDA_VEXEC::parallel_unsequenced_policy_device> : true_type
{};

template <>
struct __is_unsequenced_execution_policy_impl<_CUDA_VEXEC::unsequenced_policy_host> : true_type
{};

template <>
struct __is_unsequenced_execution_policy_impl<_CUDA_VEXEC::unsequenced_policy_device> : true_type
{};

template <class _Tp>
struct __is_unsequenced_execution_policy : __is_unsequenced_execution_policy_impl<__remove_cvref_t<_Tp>>
{};

#if _CCCL_STD_VER >= 2014 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)

template <class>
_CCCL_INLINE_VAR constexpr bool is_execution_policy_v = false;

template <>
_CCCL_INLINE_VAR constexpr bool is_execution_policy_v<_CUDA_VEXEC::sequenced_policy> = true;

template <>
_CCCL_INLINE_VAR constexpr bool is_execution_policy_v<_CUDA_VEXEC::parallel_policy_host> = true;

template <>
_CCCL_INLINE_VAR constexpr bool is_execution_policy_v<_CUDA_VEXEC::parallel_policy_device> = true;

template <>
_CCCL_INLINE_VAR constexpr bool is_execution_policy_v<_CUDA_VEXEC::parallel_unsequenced_policy_host> = true;

template <>
_CCCL_INLINE_VAR constexpr bool is_execution_policy_v<_CUDA_VEXEC::parallel_unsequenced_policy_device> = true;

template <>
_CCCL_INLINE_VAR constexpr bool is_execution_policy_v<_CUDA_VEXEC::unsequenced_policy_host> = true;

template <>
_CCCL_INLINE_VAR constexpr bool is_execution_policy_v<_CUDA_VEXEC::unsequenced_policy_device> = true;

template <class>
_CCCL_INLINE_VAR constexpr bool __is_parallel_execution_policy_impl_v = false;

template <class _Tp>
_CCCL_INLINE_VAR constexpr bool __is_parallel_execution_policy_v =
  __is_parallel_execution_policy_impl_v<__remove_cvref_t<_Tp>>;

template <>
_CCCL_INLINE_VAR constexpr bool __is_parallel_execution_policy_impl_v<_CUDA_VEXEC::parallel_policy_host> = true;

template <>
_CCCL_INLINE_VAR constexpr bool __is_parallel_execution_policy_impl_v<_CUDA_VEXEC::parallel_policy_device> = true;

template <>
_CCCL_INLINE_VAR constexpr bool __is_parallel_execution_policy_impl_v<_CUDA_VEXEC::parallel_unsequenced_policy_host> =
  true;

template <>
_CCCL_INLINE_VAR constexpr bool __is_parallel_execution_policy_impl_v<_CUDA_VEXEC::parallel_unsequenced_policy_device> =
  true;

template <class>
_CCCL_INLINE_VAR constexpr bool __is_unsequenced_execution_policy_impl_v = false;

template <class _Tp>
_CCCL_INLINE_VAR constexpr bool __is_unsequenced_execution_policy_v =
  __is_unsequenced_execution_policy_impl_v<__remove_cvref_t<_Tp>>;

template <>
_CCCL_INLINE_VAR constexpr bool __is_unsequenced_execution_policy_impl_v<_CUDA_VEXEC::parallel_unsequenced_policy_host> =
  true;

template <>
_CCCL_INLINE_VAR constexpr bool
  __is_unsequenced_execution_policy_impl_v<_CUDA_VEXEC::parallel_unsequenced_policy_device> = true;

template <>
_CCCL_INLINE_VAR constexpr bool __is_unsequenced_execution_policy_impl_v<_CUDA_VEXEC::unsequenced_policy_host> = true;

template <>
_CCCL_INLINE_VAR constexpr bool __is_unsequenced_execution_policy_impl_v<_CUDA_VEXEC::unsequenced_policy_device> = true;

#endif // _CCCL_STD_VER >= 2014 && !_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES

template <class _ExecutionPolicy>
struct __remove_parallel_policy
{
  using type = _ExecutionPolicy;
};

template <>
struct __remove_parallel_policy<_CUDA_VEXEC::parallel_policy_host>
{
  using type = _CUDA_VEXEC::sequenced_policy;
};

template <>
struct __remove_parallel_policy<_CUDA_VEXEC::parallel_policy_device>
{
  using type = _CUDA_VEXEC::sequenced_policy;
};

template <>
struct __remove_parallel_policy<_CUDA_VEXEC::parallel_unsequenced_policy_host>
{
  using type = _CUDA_VEXEC::unsequenced_policy_host;
};

template <>
struct __remove_parallel_policy<_CUDA_VEXEC::parallel_unsequenced_policy_device>
{
  using type = _CUDA_VEXEC::unsequenced_policy_device;
};

// Removes the "parallel" part of an execution policy.
// For example, turns par_unseq into unseq, and par into seq.
template <class _ExecutionPolicy>
using __remove_parallel_policy_t = typename __remove_parallel_policy<_ExecutionPolicy>::type;

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_EXECUTION_POLICY_H
