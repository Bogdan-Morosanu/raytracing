#pragma once

#include <Eigen/Dense>

#include "optional.cuh"
#include "hit_result.cuh"
#include "interval.cuh"
#include "ray.cuh"

namespace rt {

  namespace internal {
    class LineDiscriminator {
    public:
      __host__ __device__
      LineDiscriminator(Eigen::Vector3f line_start,
			Eigen::Vector3f line_end,
			Eigen::Vector3f point_inside)
	: normal_(compute_normal(line_start, line_end, point_inside))
	, intercept_(compute_intercept(line_start, normal_))
      {
	std::printf("ld: (%f, %f, %f) * p = %f\n",
		    normal_.x(), normal_.y(), normal_.z(),
		    intercept_);
      }

      __host__ __device__ bool is_inside(Eigen::Vector3f point) const
      {
	return (normal_.dot(point) - intercept_) > 0.0f;
      }
      
    private:
      __host__ __device__
      static Eigen::Vector3f compute_normal(Eigen::Vector3f line_start,
					    Eigen::Vector3f line_end,
					    Eigen::Vector3f point_inside)
      {
	Eigen::Vector3f dir = (line_end - line_start).normalized();
	return (point_inside - dir.dot(point_inside) * dir);
      }

      __host__ __device__
      static float compute_intercept(Eigen::Vector3f origin,
				     Eigen::Vector3f normal)
      {
	return origin.dot(normal);
      }
					       
      Eigen::Vector3f normal_;
      float intercept_;
    };    
  }
  
  class Triangle {
  public:
    Triangle(Eigen::Vector3f a, Eigen::Vector3f b, Eigen::Vector3f c)
      : plane_normal_(((a-b).cross(c-a)).normalized())
      , plane_intercept_(a.dot(plane_normal_))
      , disc_ab_to_c_(a, b, c)
      , disc_bc_to_a_(b, c, a)
      , disc_ca_to_b_(c, a, b)
    {
      std::printf("a %f %f %f\n", a.x(), a.y(), a.z());
      std::printf("b %f %f %f\n", b.x(), b.y(), b.z());
      std::printf("c %f %f %f\n", c.x(), c.y(), c.z());
      std::printf("plane_normal %f %f %f\n", plane_normal_.x(), plane_normal_.y(), plane_normal_.z());
      std::printf("plane_intercept %f\n", plane_intercept_);
    }

    __device__ bool can_hit(const Ray &ray) const
    {
      // (r.o + t * r.d).transpose() * normal_ == intercept
      // r.o.transpose() * normal_ + t * r.d.transpose() * normal = intercept
      // t = (intercept - r.o.dot(normal_)) / r.d.dot(normal_)

      return ray.direction().dot(plane_normal_) != 0.0f;
    }

    __device__ Optional<HitResult> hit(const Ray &ray,
				       const Interval &t_interval) const
    {
      // (r.o + t * r.d).transpose() * normal_ == intercept
      // r.o.transpose() * normal_ + t * r.d.transpose() * normal = intercept
      // t = (intercept - r.o.dot(normal_)) / r.d.dot(normal_)
      auto rdn = ray.direction().dot(plane_normal_);
      if (rdn != 0.0f) {
	auto t = (plane_intercept_ - ray.origin().dot(plane_normal_)) / rdn;
	auto point = ray.point_at_param(t);

	auto inside = disc_ab_to_c_.is_inside(point) &&
	  disc_bc_to_a_.is_inside(point) &&
	  disc_ca_to_b_.is_inside(point);

	if (inside) {
	  if (t_interval.contains(t)) {
	    // all triangles are two-sided for now
	    Eigen::Vector3f hit_normal = (rdn < 0) ? plane_normal_ : -plane_normal_;
	    return HitResult{t, point, hit_normal};
	  }
	}
      }

      return {};      
    }
    
  private:

    // __host__ __device__ Eigen::Vector3f plane_coords(Eigen::Vector3f p) const
    // {
      
    // }
    
    Eigen::Vector3f plane_normal_;
    float plane_intercept_;
    internal::LineDiscriminator disc_ab_to_c_;
    internal::LineDiscriminator disc_bc_to_a_;
    internal::LineDiscriminator disc_ca_to_b_;
  };
}
