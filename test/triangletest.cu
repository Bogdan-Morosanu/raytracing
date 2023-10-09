#include <gtest/gtest.h>

#include "container/shared_buffer.cuh"
#include "core/triangle.cuh"

__global__ void hit_triangle(rt::Triangle t,
			     rt::Ray r,
			     rt::DeviceSharedBufferView<rt::Optional<rt::HitResult>> out)
{
  rt::Interval i(-1.0f, 100.0f);
  out.value() = t.hit(r, i);
}

static constexpr float EPS = 1.e-6f;

// Demonstrate some basic assertions.
TEST(TestTriangle, BasicIntersectZAlignedUpTriangle) {
  rt::Triangle t(Eigen::Vector3f(0.0f, 1.0f, -1.0f),
		 Eigen::Vector3f(-1.0f, -1.0f, -1.0f),
		 Eigen::Vector3f(1.0f, -1.0f, -1.0f));

  rt::Ray r(Eigen::Vector3f(0.0f, 0.0f, 0.0f), Eigen::Vector3f(0.0f, 0.0f, -1.0f));

  rt::SharedBuffer<rt::Optional<rt::HitResult>> hit_result;
  hit_triangle<<<1, 1>>>(t, r, rt::make_view(hit_result));
  cudaDeviceSynchronize();

  EXPECT_TRUE(hit_result.value());
  EXPECT_NEAR(hit_result.value().value().t, 1.0f, EPS);
  EXPECT_NEAR(hit_result.value().value().normal.x(), 0.0f, EPS);
  EXPECT_NEAR(hit_result.value().value().normal.y(), 0.0f, EPS);
  EXPECT_NEAR(hit_result.value().value().normal.z(), 1.0f, EPS);
}

TEST(TestTriangle, BasicIntersectZAlignedDownTriangle) {
  rt::Triangle t(Eigen::Vector3f( 0.0f, -1.0f, -1.0f),
		  Eigen::Vector3f( 1.0f,  1.0f, -1.0f),
		  Eigen::Vector3f(-1.0f,  1.0f, -1.0f));

  rt::Ray r(Eigen::Vector3f(0.0f, 0.0f, 0.0f), Eigen::Vector3f(0.0f, 0.0f, -1.0f));

  rt::SharedBuffer<rt::Optional<rt::HitResult>> hit_result;
  hit_triangle<<<1, 1>>>(t, r, rt::make_view(hit_result));
  cudaDeviceSynchronize();

  EXPECT_TRUE(hit_result.value());
  EXPECT_NEAR(hit_result.value().value().t, 1.0f, EPS);
  EXPECT_NEAR(hit_result.value().value().normal.x(), 0.0f, EPS);
  EXPECT_NEAR(hit_result.value().value().normal.y(), 0.0f, EPS);
  EXPECT_NEAR(hit_result.value().value().normal.z(), 1.0f, EPS);
}

TEST(TestTriangle, BasicIntersectAtRayOrigin) {
  rt::Triangle t(Eigen::Vector3f( 0.0f, -1.0f,  0.0f),
		 Eigen::Vector3f( 1.0f,  1.0f, -1.0f),
		 Eigen::Vector3f(-1.0f,  1.0f,  1.0f));

  rt::Ray r(Eigen::Vector3f(0.0f, 0.0f, 0.0f), Eigen::Vector3f(0.0f, 0.0f, -1.0f));

  rt::SharedBuffer<rt::Optional<rt::HitResult>> hit_result;
  hit_triangle<<<1, 1>>>(t, r, rt::make_view(hit_result));
  cudaDeviceSynchronize();

  auto expected_normal = Eigen::Vector3f(1.0f, 0.0f, 1.0f).normalized();
  EXPECT_TRUE(hit_result.value());
  EXPECT_NEAR(hit_result.value().value().t, 0.0f, EPS);
  EXPECT_NEAR(hit_result.value().value().normal.x(), expected_normal.x(), EPS);
  EXPECT_NEAR(hit_result.value().value().normal.y(), expected_normal.y(), EPS);
  EXPECT_NEAR(hit_result.value().value().normal.z(), expected_normal.z(), EPS);
}

TEST(TestTriangle, BasicIntersectNonZAligned) {
  rt::Triangle t(Eigen::Vector3f( 0.0f, -1.0f, -3.0f),
		 Eigen::Vector3f( 1.0f,  1.0f, -4.0f),
		 Eigen::Vector3f(-1.0f,  1.0f, -2.0f));

  rt::Ray r(Eigen::Vector3f(0.0f, 0.5f, 0.0f), Eigen::Vector3f(0.0f, 0.0f, -1.0f));

  rt::SharedBuffer<rt::Optional<rt::HitResult>> hit_result;
  hit_triangle<<<1, 1>>>(t, r, rt::make_view(hit_result));
  cudaDeviceSynchronize();

  auto expected_normal = Eigen::Vector3f(1.0f, 0.0f, 1.0f).normalized();
  EXPECT_TRUE(hit_result.value());
  EXPECT_NEAR(hit_result.value().value().t, 3.0f, EPS);
  EXPECT_NEAR(hit_result.value().value().normal.x(), expected_normal.x(), EPS);
  EXPECT_NEAR(hit_result.value().value().normal.y(), expected_normal.y(), EPS);
  EXPECT_NEAR(hit_result.value().value().normal.z(), expected_normal.z(), EPS);

}
