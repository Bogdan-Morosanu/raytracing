#pragma once

#include "optional.cuh"
#include "hit_result.cuh"
#include "interval.cuh"
#include "ray.cuh"

namespace rt {

  class DirectionalLight;

  class Sphere {
    friend class SphericalLight;
    
  public:
    Sphere() = default;
    
    __host__ __device__ Sphere(const Eigen::Vector3f &c, float r)
      : center_(c)
      , radius_(r)
    { }

    __device__ bool can_hit(const Ray &ray) const
    {
      auto qparams = compute_quadratic_params(ray);
      return qparams.discriminant > 0.0f;
    }

    __device__ rt::Optional<HitResult> hit(const Ray &ray,
					   const Interval &t_interval) const
    {
      auto qparams = compute_quadratic_params(ray);

      if (qparams.discriminant > 0.0f) {
	HitResult res;

	// take the root with a lower t value (ie closer to the ray being shot)
	auto t = (-qparams.b - sqrt(qparams.discriminant)) / (2.0f * qparams.a);

	if (t_interval.contains(t)) {
	  res.point = ray.point_at_param(t);
	  res.normal = (res.point - center_).normalized();
	  res.t = t;
	  return res;

	} else {
	  t =  (-qparams.b + sqrt(qparams.discriminant)) / (2.0f * qparams.a);
	  if (t_interval.contains(t)) {
	    res.point = ray.point_at_param(t);
	    res.normal = (res.point - center_).normalized();
	    res.t = t;
	    return res;
	  }
	}
	
      }

      // no valid param
      return {};
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
      // (xo + t*xd - xc)^2 + (yo + t*yd - yc)^2 + (zo + t*zd - zc)^2  = r*r
      // (xd^2 + yd^2 + zd^2)* t^2 + 2 * ((xo-xc)*xd  + (yo-yc)*yd + (zo-zc)*zd)*t + ((xo-xc)^2 + (yo-yc)^2 + (zo-zc)^2) = r*r
      //
      
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
