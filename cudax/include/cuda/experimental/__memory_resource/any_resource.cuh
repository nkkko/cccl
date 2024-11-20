//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__MEMORY_RESOURCE_ANY_RESOURCE_H
#define _CUDAX__MEMORY_RESOURCE_ANY_RESOURCE_H

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
#include <cuda/__memory_resource/resource_ref.h>
#include <cuda/std/__concepts/__concept_macros.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/optional>

#include <cuda/experimental/__utility/basic_any.cuh>

namespace cuda::experimental::mr
{
#ifndef DOXYGEN_ACTIVE // Do not document this

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
  _CUDAX_PUBLIC_API static auto __get_property(const _Ty& __obj) //
    -> __property_result_t<_Property>
  {
    if constexpr (!_CUDA_VSTD::is_same_v<__property_result_t<_Property>, void>)
    {
      return get_property(__obj, _Property());
    }
    else
    {
      return void();
    }
  }

  template <class...>
  struct __iproperty : interface<__iproperty>
  {
    _CUDAX_HOST_API friend auto
    get_property([[maybe_unused]] const __iproperty& __obj, _Property) -> __property_result_t<_Property>
    {
      if constexpr (!_CUDA_VSTD::is_same_v<__property_result_t<_Property>, void>)
      {
        return __cudax::virtcall<&__get_property<__iproperty>>(&__obj);
      }
      else
      {
        return void();
      }
    }

    template <class _Ty>
    using overrides _CCCL_NODEBUG_ALIAS = overrides_for<_Ty, _CUDAX_FNPTR_CONSTANT_WAR(&__get_property<_Ty>)>;
  };
};

template <class _Property>
using __iproperty = typename __with_property<_Property>::template __iproperty<>;

template <class... _Properties>
using __iproperty_set = iset<__iproperty<_Properties>...>;

// Wrap the calls of the allocate_async and deallocate_async member functions
// because of NVBUG#4967486
template <class _Resource>
_CUDAX_PUBLIC_API void* __allocate_async(_Resource& mr, size_t __bytes, size_t __alignment, ::cuda::stream_ref __stream)
{
  return mr.allocate_async(__bytes, __alignment, __stream);
}

template <class _Resource>
_CUDAX_PUBLIC_API void
__deallocate_async(_Resource& mr, void* __pv, size_t __bytes, size_t __alignment, ::cuda::stream_ref __stream)
{
  mr.deallocate_async(__pv, __bytes, __alignment, __stream);
}

template <class...>
struct __ibasic_resource : interface<__ibasic_resource>
{
  _CUDAX_PUBLIC_API void* allocate(size_t __bytes, size_t __alignment = alignof(_CUDA_VSTD::max_align_t))
  {
    return __cudax::virtcall<&__ibasic_resource::allocate>(this, __bytes, __alignment);
  }

  _CUDAX_PUBLIC_API void deallocate(void* __pv, size_t __bytes, size_t __alignment = alignof(_CUDA_VSTD::max_align_t))
  {
    return __cudax::virtcall<&__ibasic_resource::deallocate>(this, __pv, __bytes, __alignment);
  }

  template <class _Ty>
  using overrides _CCCL_NODEBUG_ALIAS =
    overrides_for<_Ty, _CUDAX_FNPTR_CONSTANT_WAR(&_Ty::allocate), _CUDAX_FNPTR_CONSTANT_WAR(&_Ty::deallocate)>;
};

template <class...>
struct __ibasic_async_resource : interface<__ibasic_async_resource>
{
  _CUDAX_PUBLIC_API void* allocate_async(size_t __bytes, size_t __alignment, ::cuda::stream_ref __stream)
  {
    return __cudax::virtcall<&__allocate_async<__ibasic_async_resource>>(this, __bytes, __alignment, __stream);
  }

  _CUDAX_PUBLIC_API void* allocate_async(size_t __bytes, ::cuda::stream_ref __stream)
  {
    return __cudax::virtcall<&__allocate_async<__ibasic_async_resource>>(
      this, __bytes, alignof(_CUDA_VSTD::max_align_t), __stream);
  }

  _CUDAX_PUBLIC_API void deallocate_async(void* __pv, size_t __bytes, size_t __alignment, ::cuda::stream_ref __stream)
  {
    return __cudax::virtcall<&__deallocate_async<__ibasic_async_resource>>(this, __pv, __bytes, __alignment, __stream);
  }

  _CUDAX_PUBLIC_API void deallocate_async(void* __pv, size_t __bytes, ::cuda::stream_ref __stream)
  {
    return __cudax::virtcall<&__deallocate_async<__ibasic_async_resource>>(
      this, __pv, __bytes, alignof(_CUDA_VSTD::max_align_t), __stream);
  }

  template <class _Ty>
  using overrides _CCCL_NODEBUG_ALIAS =
    overrides_for<_Ty,
                  _CUDAX_FNPTR_CONSTANT_WAR(&__allocate_async<_Ty>),
                  _CUDAX_FNPTR_CONSTANT_WAR(&__deallocate_async<_Ty>)>;
};

template <class _Resource>
_CUDAX_HOST_API const _CUDA_VMR::_Alloc_vtable* __get_resource_vptr(_Resource& __mr) noexcept
{
  if constexpr (_CUDA_VMR::resource<_Resource>)
  {
    return &_CUDA_VMR::__alloc_vtable<_CUDA_VMR::_AllocType::_Default, _CUDA_VMR::_WrapperType::_Reference, _Resource>;
  }
  else
  {
    // This branch is taken when called from the thunk of an unspecialized
    // interface; e.g., `icat<>` rather than `icat<ialley_cat<>>`. The thunks of
    // unspecialized interfaces are never called, they just need to exist.
    _CCCL_UNREACHABLE();
  }
}

template <class _VPtr, class... _Properties>
_CUDAX_HOST_API auto __make_resource_vtable(_VPtr __vptr, _CUDA_VMR::_Resource_vtable<_Properties...>*) noexcept
  -> _CUDA_VMR::_Resource_vtable<_Properties...>
{
  return {__vptr->__query_interface(__iproperty<_Properties>())->__fn_...};
}

template <class... _Super>
struct _LIBCUDACXX_DECLSPEC_EMPTY_BASES __iresource_ref_conversions
    : interface<__iresource_ref_conversions>
    , _CUDA_VMR::_Resource_ref_base
{
  using __basic_any_type =
    _CUDA_VSTD::decay_t<decltype(__cudax::basic_any_from(declval<__iresource_ref_conversions&>()))>;

  template <class _Property>
  using __iprop = __rebind_interface<__iproperty<_Property>, _Super...>;

  _LIBCUDACXX_TEMPLATE(class... _Properties)
  _LIBCUDACXX_REQUIRES((std::derived_from<__basic_any_type, __iprop<_Properties>> && ...))
  operator _CUDA_VMR::resource_ref<_Properties...>()
  {
    _CUDA_VMR::_Filtered_vtable<_Properties...>* __prop_vtable = nullptr;
    auto& __self                                               = __cudax::basic_any_from(*this);
    return _CUDA_VMR::_Resource_ref_helper::_Construct<_CUDA_VMR::_AllocType::_Default, _Properties...>(
      __basic_any_access::__get_optr(__self),
      __cudax::virtcall<&__get_resource_vptr<__iresource_ref_conversions>>(this),
      __cudax::mr::__make_resource_vtable(__basic_any_access::__get_vptr(__self), __prop_vtable));
  }

  template <class _Resource>
  using overrides = overrides_for<_Resource, _CUDAX_FNPTR_CONSTANT_WAR(&__get_resource_vptr<_Resource>)>;
};

template <class... _Properties>
using __iresource _CCCL_NODEBUG_ALIAS =
  iset<__ibasic_resource<>,
       __iproperty_set<_Properties...>,
       __iresource_ref_conversions<>,
       icopyable<>,
       iequality_comparable<>>;

template <class... _Properties>
using __iasync_resource _CCCL_NODEBUG_ALIAS = iset<__iresource<_Properties...>, __ibasic_async_resource<>>;

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

#endif // DOXYGEN_ACTIVE

//! @rst
//! .. _cudax-memory-resource-any-resource:
//!
//! Type erased wrapper around a `resource`
//! ----------------------------------------
//!
//! ``any_resource`` wraps any given :ref:`resource <libcudacxx-extended-api-memory-resources-resource>` that
//! satisfies the required properties. It owns the contained resource, taking care of construction / destruction.
//! This makes it especially suited for use in e.g. container types that need to ensure that the lifetime of the
//! container exceeds the lifetime of the memory resource used to allocate the storage
//!
//! @endrst
template <class... _Properties>
struct _LIBCUDACXX_DECLSPEC_EMPTY_BASES any_resource
    : basic_any<__iresource<_Properties...>>
    , __with_try_get_property<any_resource<_Properties...>>
{
  using any_resource::basic_any::basic_any;

private:
  using any_resource::basic_any::interface;
};

//! @rst
//! .. _cudax-memory-resource-any-async-resource:
//!
//! Type erased wrapper around an `async_resource`
//! -----------------------------------------------
//!
//! ``any_async_resource`` wraps any given :ref:`async resource <libcudacxx-extended-api-memory-resources-resource>`
//! that satisfies the required properties. It owns the contained resource, taking care of construction / destruction.
//! This makes it especially suited for use in e.g. container types that need to ensure that the lifetime of the
//! container exceeds the lifetime of the memory resource used to allocate the storage
//!
//! @endrst
template <class... _Properties>
struct _LIBCUDACXX_DECLSPEC_EMPTY_BASES any_async_resource
    : basic_any<__iasync_resource<_Properties...>>
    , __with_try_get_property<any_async_resource<_Properties...>>
{
  using any_async_resource::basic_any::basic_any;

private:
  using any_async_resource::basic_any::interface;
};

//! @brief Type erased wrapper around a `resource` that satisfies \tparam _Properties
//! @tparam _Properties The properties that any resource wrapped within the `resource_ref` needs to satisfy
template <class... _Properties>
struct _LIBCUDACXX_DECLSPEC_EMPTY_BASES resource_ref
    : basic_any<__iresource<_Properties...>&>
    , __with_try_get_property<resource_ref<_Properties...>>
{
  using resource_ref::basic_any::basic_any;

private:
  using resource_ref::basic_any::interface;
};

//! @brief Type erased wrapper around a `async_resource` that satisfies \tparam _Properties
//! @tparam _Properties The properties that any async resource wrapped within the `async_resource_ref` needs to satisfy
template <class... _Properties>
struct _LIBCUDACXX_DECLSPEC_EMPTY_BASES async_resource_ref
    : basic_any<__iasync_resource<_Properties...>&>
    , __with_try_get_property<async_resource_ref<_Properties...>>
{
  using async_resource_ref::basic_any::basic_any;

private:
  using async_resource_ref::basic_any::interface;
};

//! @rst
//! .. _cudax-memory-resource-make-any-resource:
//!
//! Factory function for `any_resource` objects
//! -------------------------------------------
//!
//! ``make_any_resource`` constructs an :ref:`any_resource <cudax-memory-resource-any-resource>` object that wraps a
//! newly constructed instance of the given resource type. The resource type must satisfy the ``cuda::mr::resource``
//! concept and provide all of the properties specified in the template parameter pack.
//!
//! @param __args The arguments used to construct the instance of the resource type.
//!
//! @endrst
template <class _Resource, class... _Properties, class... _Args>
auto make_any_resource(_Args&&... __args) -> any_resource<_Properties...>
{
  static_assert(_CUDA_VMR::resource<_Resource>, "_Resource does not satisfy the cuda::mr::resource concept");
  static_assert(_CUDA_VMR::resource_with<_Resource, _Properties...>,
                "The provided _Resource type does not support the requested properties");
  return any_resource<_Properties...>{_CUDA_VSTD::in_place_type<_Resource>, _CUDA_VSTD::forward<_Args>(__args)...};
}

//! @rst
//! .. _cudax-memory-resource-make-any-async-resource:
//!
//! Factory function for `any_async_resource` objects
//! -------------------------------------------------
//!
//! ``make_any_async_resource`` constructs an :ref:`any_async_resource <cudax-memory-resource-any-async-resource>`
//! object that wraps a newly constructed instance of the given resource type. The resource type must satisfy the
//! ``cuda::mr::async_resource`` concept and provide all of the properties specified in the template parameter pack.
//!
//! @param __args The arguments used to construct the instance of the resource type.
//!
//! @endrst
template <class _Resource, class... _Properties, class... _Args>
auto make_any_async_resource(_Args&&... __args) -> any_async_resource<_Properties...>
{
  static_assert(_CUDA_VMR::async_resource<_Resource>,
                "_Resource does not satisfy the cuda::mr::async_resource concept");
  static_assert(_CUDA_VMR::async_resource_with<_Resource, _Properties...>,
                "The provided _Resource type does not support the requested properties");
  return any_async_resource<_Properties...>{_CUDA_VSTD::in_place_type<_Resource>, _CUDA_VSTD::forward<_Args>(__args)...};
}

} // namespace cuda::experimental::mr

#endif // _CUDAX__MEMORY_RESOURCE_ANY_RESOURCE_H
