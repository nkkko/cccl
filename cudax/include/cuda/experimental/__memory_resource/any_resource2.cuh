//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__MEMORY_RESOURCE_ANY_RESOURCE2_H
#define _CUDAX__MEMORY_RESOURCE_ANY_RESOURCE2_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// If the memory resource header was included without the experimental flag,
// tell the user to define the experimental flag.
#if defined(_CUDA_MEMORY_RESOURCE) && !defined(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)
#  error "To use the experimental memory resource, define LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE"
#endif

// cuda::mr is unavable on MSVC 2017
#if defined(_CCCL_COMPILER_MSVC_2017)
#  error "The any_resource header is not supported on MSVC 2017"
#endif

#if !defined(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)
#  define LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
#endif

#include <cuda/__memory_resource/get_property.h>
#include <cuda/__memory_resource/properties.h>
#include <cuda/__memory_resource/resource.h>
#include <cuda/std/__concepts/__concept_macros.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/optional>

#include <cuda/experimental/__utility/basic_any.cuh>

namespace cuda::experimental::mr
{
template <class _Property>
using __property_result_t _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__type_call1< //
  _CUDA_VSTD::conditional_t<cuda::property_with_value<_Property>,
                            _CUDA_VSTD::__type_quote1<__property_value_t>,
                            _CUDA_VSTD::__type_always<void>>,
  _Property>;

template <class _Property>
struct __with_property
{
  template <class _Ty>
  _CUDAX_PUBLIC_API static auto __get_property(const _Ty& __obj, _Property __prop) noexcept //
    -> __property_result_t<_Property>
  {
    static_assert(noexcept(get_property(__obj, __prop)));
    if constexpr (!_CUDA_VSTD::is_same_v<__property_result_t<_Property>, void>)
    {
      return get_property(__obj, __prop);
    }
    else
    {
      return void();
    }
  }

  template <class...>
  struct __iproperty : interface<__iproperty>
  {
    _CUDAX_API friend auto get_property([[maybe_unused]] const __iproperty& __obj,
                                        [[maybe_unused]] _Property __prop) noexcept -> __property_result_t<_Property>
    {
      if constexpr (!_CUDA_VSTD::is_same_v<__property_result_t<_Property>, void>)
      {
        return __cudax::virtcall<&__get_property<__iproperty>>(&__obj, __prop);
      }
      else
      {
        return void();
      }
    }

    template <class _Ty>
    using overrides _CCCL_NODEBUG_ALIAS = overrides_for<_Ty, &__get_property<_Ty>>;
  };
};

template <class _Property>
using __iproperty = typename __with_property<_Property>::template __iproperty<>;

template <class... _Properties>
using __iproperty_set = iset<__iproperty<_Properties>...>;

template <class...>
struct __ibasic_resource : interface<__ibasic_resource, extends<icopyable<>, iequality_comparable<>>>
{
  _CUDAX_PUBLIC_API void* allocate(size_t __bytes, size_t __alignment)
  {
    return __cudax::virtcall<&__ibasic_resource::allocate>(this, __bytes, __alignment);
  }

  _CUDAX_PUBLIC_API void deallocate(void* __pv, size_t __bytes, size_t __alignment)
  {
    return __cudax::virtcall<&__ibasic_resource::deallocate>(this, __pv, __bytes, __alignment);
  }

  _CUDAX_PUBLIC_API void* allocate_async(size_t __bytes, size_t __alignment, ::cuda::stream_ref __stream)
  {
    return __cudax::virtcall<&__ibasic_resource::allocate_async>(this, __bytes, __alignment, __stream);
  }

  _CUDAX_PUBLIC_API void deallocate_async(void* __pv, size_t __bytes, size_t __alignment, ::cuda::stream_ref __stream)
  {
    return __cudax::virtcall<&__ibasic_resource::deallocate_async>(this, __pv, __bytes, __alignment, __stream);
  }

  template <class _Ty>
  using overrides _CCCL_NODEBUG_ALIAS =
    overrides_for<_Ty, &_Ty::allocate, &_Ty::deallocate, &_Ty::allocate_async, &_Ty::deallocate_async>;
};

template <class... _Properties>
using __iresource _CCCL_NODEBUG_ALIAS = iset<__ibasic_resource<>, __iproperty_set<_Properties...>>;

template <class _Property>
using __try_property_result_t =
  _CUDA_VSTD::conditional_t<!_CUDA_VSTD::is_same_v<__property_result_t<_Property>, void>, //
                            _CUDA_VSTD::optional<__property_result_t<_Property>>, //
                            bool>;

template <class _Derived>
struct __with_try_get_property
{
  template <class _Property>
  _CUDAX_HOST_API _CCCL_NODISCARD_FRIEND auto
  try_get_property(const _Derived& __self, _Property) noexcept -> __try_property_result_t<_Property>
  {
    auto __prop = __cudax::dynamic_any_cast<const __iproperty<_Property>*>(&__self);
    if constexpr (_CUDA_VSTD::is_same_v<__property_result_t<_Property>, void>)
    {
      return __prop != nullptr;
    }
    else if (__prop)
    {
      return get_property(*__prop, _Property{});
    }
    else
    {
      return _CUDA_VSTD::nullopt;
    }
  }
};

template <class... _Properties>
struct _LIBCUDACXX_DECLSPEC_EMPTY_BASES any_resource2
    : basic_any<__iresource<_Properties...>>
    , __with_try_get_property<any_resource2<_Properties...>>
{
  using any_resource2::basic_any::basic_any;

private:
  using any_resource2::basic_any::interface;
};

template <class... _Properties>
struct _LIBCUDACXX_DECLSPEC_EMPTY_BASES resource_ref2
    : basic_any<__iresource<_Properties...>&>
    , __with_try_get_property<resource_ref2<_Properties...>>
{
  using resource_ref2::basic_any::basic_any;

private:
  using resource_ref2::basic_any::interface;
};

template <class _Resource, class... _Properties, class... _Args>
auto make_any_resource2(_Args&&... __args) -> any_resource2<_Properties...>
{
  static_assert(_CUDA_VMR::resource<_Resource>, "_Resource does not satisfy the cuda::mr::resource concept");
  static_assert(_CUDA_VMR::resource_with<_Resource, _Properties...>,
                "Resource does not satisfy the required properties");
  return any_resource2<_Properties...>{_CUDA_VSTD::in_place_type<_Resource>, _CUDA_VSTD::forward<_Args>(__args)...};
}

} // namespace cuda::experimental::mr

#endif // _CUDAX__MEMORY_RESOURCE_ANY_RESOURCE2_H
