#pragma once

#include "optional.cuh"
#include "hit_result.cuh"
#include "ray.cuh"

namespace rt {
  class Sphere {
  public:
    __host__ __device__ Sphere() = default;
    
    __host__ __device__ Sphere(const Eigen::Vector3f &c, float r)
      : center_(c)
      , radius_(r)
    { }

    __device__ bool can_hit(const Ray &ray) const
    {
      auto qparams = compute_quadratic_params(ray);
      return qparams.discriminant > 0.0f;
    }

    __device__ rt::Optional<HitResult> hit(const Ray &ray) const
    {
      auto qparams = compute_quadratic_params(ray);

      if (qparams.discriminant > 0.0f) {
	HitResult res;

	res.time = 0.0f;
	// take the root with a lower t value (ie closer to the ray being shot)
	auto t = -qparams.b - sqrt(qparams.discriminant) / (2.0f * qparams.a);
	res.point = ray.point_at_param(t);
	res.normal = (res.point - center_).normalized();

	return res;
	
      } else {
	return {};
      }
    }
    
  private:

    struct QuadraticParams {
      float a;
      float b;
      float c;
      float discriminant;
    };
    
    __device__ QuadraticParams compute_quadratic_params(const Ray &ray) const
    {
      Eigen::Vector3f oc = ray.origin() - center_;

      float a = ray.direction().dot(ray.direction());
      float b = 2.0f * oc.dot(ray.direction());
      float c = oc.dot(oc) - radius_ * radius_;
      float discriminant = b * b - 4.0f * a * c;

      return QuadraticParams{a, b, c, discriminant};
    }
    
    Eigen::Vector3f center_;
    float radius_;
  };
}
