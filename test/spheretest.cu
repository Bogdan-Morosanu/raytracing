#include <gtest/gtest.h>

#include "container/shared_buffer.cuh"
#include "core/sphere.cuh"

__global__ void hit_sphere(rt::Sphere s,
			     rt::Ray r,
			     rt::DeviceSharedBufferView<rt::Optional<rt::HitResult>> out)
{
  rt::Interval i(-1.0f, 100.0f);
  out.value() = s.hit(r, i);
}

static constexpr float EPS = 1.e-6f;

TEST(TestSphere, BasicIntersectZAlignedSphereNear) {
  rt::Sphere s(Eigen::Vector3f(0.0f, 0.0f, -2.0f), 1.0f);
  rt::Ray r(Eigen::Vector3f(0.0f, 0.0f, 0.0f), Eigen::Vector3f(0.0f, 0.0f, -1.0f));

  rt::SharedBuffer<rt::Optional<rt::HitResult>> hit_result;
  hit_sphere<<<1, 1>>>(s, r, rt::make_view(hit_result));
  cudaDeviceSynchronize();

  EXPECT_TRUE(hit_result.value());
  EXPECT_NEAR(hit_result.value().value().t, 1.0f, EPS);
  EXPECT_NEAR(hit_result.value().value().normal.x(), 0.0f, EPS);
  EXPECT_NEAR(hit_result.value().value().normal.y(), 0.0f, EPS);
  EXPECT_NEAR(hit_result.value().value().normal.z(), 1.0f, EPS);
}

TEST(TestSphere, BasicIntersectZAlignedSphereFar) {
  rt::Sphere s(Eigen::Vector3f(0.0f, 0.0f, -5.0f), 1.0f);
  rt::Ray r(Eigen::Vector3f(0.0f, 0.0f, 0.0f), Eigen::Vector3f(0.0f, 0.0f, -1.0f));

  rt::SharedBuffer<rt::Optional<rt::HitResult>> hit_result;
  hit_sphere<<<1, 1>>>(s, r, rt::make_view(hit_result));
  cudaDeviceSynchronize();

  EXPECT_TRUE(hit_result.value());
  EXPECT_NEAR(hit_result.value().value().t, 4.0f, EPS);
  EXPECT_NEAR(hit_result.value().value().normal.x(), 0.0f, EPS);
  EXPECT_NEAR(hit_result.value().value().normal.y(), 0.0f, EPS);
  EXPECT_NEAR(hit_result.value().value().normal.z(), 1.0f, EPS);
}

TEST(TestSphere, BasicIntersectZAlignedSphereLeft) {
  rt::Sphere s(Eigen::Vector3f(0.0f, 0.0f, -5.0f), 1.0f);
  rt::Ray r(Eigen::Vector3f(-1.0f, 0.0f, 0.0f), Eigen::Vector3f(0.0f, 0.0f, -1.0f));

  rt::SharedBuffer<rt::Optional<rt::HitResult>> hit_result;
  hit_sphere<<<1, 1>>>(s, r, rt::make_view(hit_result));
  cudaDeviceSynchronize();

  EXPECT_TRUE(hit_result.value());
  EXPECT_NEAR(hit_result.value().value().t, 5.0f, EPS);
  EXPECT_NEAR(hit_result.value().value().normal.x(), -1.0f, EPS);
  EXPECT_NEAR(hit_result.value().value().normal.y(),  0.0f, EPS);
  EXPECT_NEAR(hit_result.value().value().normal.z(),  0.0f, EPS);
}


TEST(TestSphere, BasicIntersectZAlignedSphereRight) {
  rt::Sphere s(Eigen::Vector3f(0.0f, 0.0f, -5.0f), 1.0f);
  rt::Ray r(Eigen::Vector3f(1.0f, 0.0f, 0.0f), Eigen::Vector3f(0.0f, 0.0f, -1.0f));

  rt::SharedBuffer<rt::Optional<rt::HitResult>> hit_result;
  hit_sphere<<<1, 1>>>(s, r, rt::make_view(hit_result));
  cudaDeviceSynchronize();

  EXPECT_TRUE(hit_result.value());
  EXPECT_NEAR(hit_result.value().value().t, 5.0f, EPS);
  EXPECT_NEAR(hit_result.value().value().normal.x(),  1.0f, EPS);
  EXPECT_NEAR(hit_result.value().value().normal.y(),  0.0f, EPS);
  EXPECT_NEAR(hit_result.value().value().normal.z(),  0.0f, EPS);
}


TEST(TestSphere, BasicIntersectZAlignedSphereUp) {
  rt::Sphere s(Eigen::Vector3f(0.0f, 0.0f, -5.0f), 1.0f);
  rt::Ray r(Eigen::Vector3f(0.0f, 1.0f, 0.0f), Eigen::Vector3f(0.0f, 0.0f, -1.0f));

  rt::SharedBuffer<rt::Optional<rt::HitResult>> hit_result;
  hit_sphere<<<1, 1>>>(s, r, rt::make_view(hit_result));
  cudaDeviceSynchronize();

  EXPECT_TRUE(hit_result.value());
  EXPECT_NEAR(hit_result.value().value().t, 5.0f, EPS);
  EXPECT_NEAR(hit_result.value().value().normal.x(),  0.0f, EPS);
  EXPECT_NEAR(hit_result.value().value().normal.y(),  1.0f, EPS);
  EXPECT_NEAR(hit_result.value().value().normal.z(),  0.0f, EPS);
}


TEST(TestSphere, BasicIntersectZAlignedSphereDown) {
  rt::Sphere s(Eigen::Vector3f(0.0f, 0.0f, -5.0f), 1.0f);
  rt::Ray r(Eigen::Vector3f(0.0f, -1.0f, 0.0f), Eigen::Vector3f(0.0f, 0.0f, -1.0f));

  rt::SharedBuffer<rt::Optional<rt::HitResult>> hit_result;
  hit_sphere<<<1, 1>>>(s, r, rt::make_view(hit_result));
  cudaDeviceSynchronize();

  EXPECT_TRUE(hit_result.value());
  EXPECT_NEAR(hit_result.value().value().t, 5.0f, EPS);
  EXPECT_NEAR(hit_result.value().value().normal.x(),  0.0f, EPS);
  EXPECT_NEAR(hit_result.value().value().normal.y(), -1.0f, EPS);
  EXPECT_NEAR(hit_result.value().value().normal.z(),  0.0f, EPS);
}

TEST(TestSphere, BasicIntersectNonZAlignedSphere) {
  auto center = Eigen::Vector3f(1.0f, 1.0f, -5.0f);
  rt::Sphere s(center, 1.0f);
  rt::Ray r(Eigen::Vector3f(0.0f, 0.0f, 0.0f), Eigen::Vector3f(1.0f, 1.0f, -4.0f));

  rt::SharedBuffer<rt::Optional<rt::HitResult>> hit_result;
  hit_sphere<<<1, 1>>>(s, r, rt::make_view(hit_result));
  cudaDeviceSynchronize();

  EXPECT_TRUE(hit_result.value());
  EXPECT_NEAR(hit_result.value().value().t, 1.0f, EPS);
  auto t = hit_result.value().value().t;
  auto expected_normal = (r.point_at_param(t) - center).normalized();
  EXPECT_NEAR(hit_result.value().value().normal.x(),  expected_normal.x(), EPS);
  EXPECT_NEAR(hit_result.value().value().normal.y(),  expected_normal.y(), EPS);
  EXPECT_NEAR(hit_result.value().value().normal.z(),  expected_normal.z(), EPS);
}
