//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA_ARCH__

#  include <cuda/memory_resource>

#  include <cuda/experimental/memory_resource.cuh>

#  include "test_resource.cuh"
#  include <catch2/catch.hpp>
#  include <testing.cuh>

static_assert(
  cuda::has_property<cudax::mr::any_resource<cuda::mr::host_accessible, get_data>, cuda::mr::host_accessible>);
static_assert(cuda::has_property<cudax::mr::any_resource<cuda::mr::host_accessible, get_data>, get_data>);
static_assert(
  !cuda::has_property<cudax::mr::any_resource<cuda::mr::host_accessible, get_data>, cuda::mr::device_accessible>);

TEMPLATE_TEST_CASE_METHOD(test_fixture, "any_resource", "[container][resource]", big_resource, small_resource)
{
  using TestResource    = TestType;
  constexpr bool is_big = sizeof(TestResource) > cudax::__default_buffer_size;

  SECTION("construct and destruct")
  {
    Counts expected{};
    CHECK(this->counts == expected);
    {
      cudax::mr::any_resource<cuda::mr::host_accessible, get_data> mr{TestResource{42, this}};
      expected.new_count += is_big;
      ++expected.object_count;
      ++expected.move_count;
      CHECK(this->counts == expected);
      CHECK(get_property(mr, get_data{}) == 42);
      get_property(mr, cuda::mr::host_accessible{});
    }
    expected.delete_count += is_big;
    --expected.object_count;
    CHECK(this->counts == expected);
  }

  // Reset the counters:
  this->counts = Counts();

  SECTION("copy and move")
  {
    Counts expected{};
    CHECK(this->counts == expected);
    {
      cudax::mr::any_resource<cuda::mr::host_accessible, get_data> mr{TestResource{42, this}};
      expected.new_count += is_big;
      ++expected.object_count;
      ++expected.move_count;
      CHECK(this->counts == expected);

      auto mr2 = mr;
      expected.new_count += is_big;
      ++expected.copy_count;
      ++expected.object_count;
      CHECK(this->counts == expected);
      CHECK(mr == mr2);
      ++expected.equal_to_count;
      CHECK(this->counts == expected);

      auto mr3 = std::move(mr);
      expected.move_count += !is_big; // for big resources, move is a pointer swap
      CHECK(this->counts == expected);
      CHECK(mr2 == mr3);
      ++expected.equal_to_count;
      CHECK(this->counts == expected);
    }
    expected.delete_count += 2 * is_big;
    expected.object_count -= 2;
    CHECK(this->counts == expected);
  }

  // Reset the counters:
  this->counts = Counts();

  SECTION("allocate and deallocate")
  {
    Counts expected{};
    CHECK(this->counts == expected);
    {
      cudax::mr::any_resource<cuda::mr::host_accessible, get_data> mr{TestResource{42, this}};
      expected.new_count += is_big;
      ++expected.object_count;
      ++expected.move_count;
      CHECK(this->counts == expected);

      void* ptr = mr.allocate(bytes(50), align(8));
      CHECK(ptr == this);
      ++expected.allocate_count;
      CHECK(this->counts == expected);

      mr.deallocate(ptr, bytes(50), align(8));
      ++expected.deallocate_count;
      CHECK(this->counts == expected);
    }
    expected.delete_count += is_big;
    --expected.object_count;
    CHECK(this->counts == expected);
  }

  // Reset the counters:
  this->counts = Counts();

  SECTION("equality comparable")
  {
    Counts expected{};
    CHECK(this->counts == expected);
    {
      cuda::mr::managed_memory_resource managed1{}, managed2{};
      CHECK(managed1 == managed2);
      cudax::mr::any_resource<cuda::mr::device_accessible> mr{managed1};
      CHECK(mr == managed1);
    }
    CHECK(this->counts == expected);
  }

  // Reset the counters:
  this->counts = Counts();

  SECTION("conversion from any_resource to cudax::mr::resource_ref")
  {
    Counts expected{};
    {
      cudax::mr::any_resource<cuda::mr::host_accessible, get_data> mr{TestResource{42, this}};
      expected.new_count += is_big;
      ++expected.object_count;
      ++expected.move_count;
      CHECK(this->counts == expected);

      cudax::mr::resource_ref<cuda::mr::host_accessible, get_data> ref = mr;

      CHECK(this->counts == expected);
      auto* ptr = ref.allocate(bytes(100), align(8));
      CHECK(ptr == this);
      ++expected.allocate_count;
      CHECK(this->counts == expected);
      ref.deallocate(ptr, bytes(0), align(0));
      ++expected.deallocate_count;
      CHECK(this->counts == expected);
    }
    expected.delete_count += is_big;
    --expected.object_count;
    CHECK(this->counts == expected);
  }

  SECTION("conversion from any_resource to cuda::mr::resource_ref")
  {
    Counts expected{};
    {
      cudax::mr::any_resource<cuda::mr::host_accessible, get_data> mr{TestResource{42, this}};
      expected.new_count += is_big;
      ++expected.object_count;
      ++expected.move_count;
      CHECK(this->counts == expected);

      cuda::mr::resource_ref<cuda::mr::host_accessible, get_data> ref = mr;

      CHECK(this->counts == expected);
      auto* ptr = ref.allocate(bytes(100), align(8));
      CHECK(ptr == this);
      ++expected.allocate_count;
      CHECK(this->counts == expected);
      ref.deallocate(ptr, bytes(0), align(0));
      ++expected.deallocate_count;
      CHECK(this->counts == expected);
    }
    expected.delete_count += is_big;
    --expected.object_count;
    CHECK(this->counts == expected);
  }

  // Reset the counters:
  this->counts = Counts();

  SECTION("conversion from resource_ref to any_resource")
  {
    Counts expected{};
    {
      TestResource test{42, this};
      ++expected.object_count;
      cudax::mr::resource_ref<cuda::mr::host_accessible, get_data> ref{test};
      CHECK(this->counts == expected);

      cudax::mr::any_resource<cuda::mr::host_accessible, get_data> mr = ref;
      expected.new_count += is_big;
      ++expected.object_count;
      ++expected.copy_count;
      CHECK(this->counts == expected);

      auto* ptr = ref.allocate(bytes(100), align(8));
      CHECK(ptr == this);
      ++expected.allocate_count;
      CHECK(this->counts == expected);
      ref.deallocate(ptr, bytes(0), align(0));
      ++expected.deallocate_count;
      CHECK(this->counts == expected);
    }
    expected.delete_count += is_big;
    expected.object_count -= 2;
    CHECK(this->counts == expected);
  }

  // Reset the counters:
  this->counts = Counts();

  SECTION("test slicing off of properties")
  {
    Counts expected{};
    CHECK(this->counts == expected);
    {
      cudax::mr::any_resource<cuda::mr::host_accessible, get_data> mr{TestResource{42, this}};
      expected.new_count += is_big;
      ++expected.object_count;
      ++expected.move_count;
      CHECK(this->counts == expected);

      cudax::mr::any_resource<get_data> mr2 = mr;
      expected.new_count += is_big;
      ++expected.object_count;
      ++expected.copy_count;
      CHECK(this->counts == expected);

      CHECK(get_property(mr2, get_data{}) == 42);
      auto data = try_get_property(mr2, get_data{});
      static_assert(cuda::std::is_same_v<decltype(data), cuda::std::optional<int>>);
      CHECK(data.has_value());
      CHECK(data.value() == 42);

      auto host = try_get_property(mr2, cuda::mr::host_accessible{});
      static_assert(cuda::std::is_same_v<decltype(host), bool>);
      CHECK(host);

      auto device = try_get_property(mr2, cuda::mr::device_accessible{});
      static_assert(cuda::std::is_same_v<decltype(device), bool>);
      CHECK(!device);
    }
    expected.delete_count += 2 * is_big;
    expected.object_count -= 2;
    CHECK(this->counts == expected);
  }

  // Reset the counters:
  this->counts = Counts();

  SECTION("make_any_resource")
  {
    Counts expected{};
    CHECK(this->counts == expected);
    {
      cudax::mr::any_resource<cuda::mr::host_accessible, get_data> mr =
        cudax::mr::make_any_resource<TestResource, cuda::mr::host_accessible, get_data>(42, this);
      expected.new_count += is_big;
      ++expected.object_count;
      CHECK(this->counts == expected);
    }
    expected.delete_count += is_big;
    --expected.object_count;
    CHECK(this->counts == expected);
  }
  // Reset the counters:
  this->counts = Counts();
}

#endif // __CUDA_ARCH__
