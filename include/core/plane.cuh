#pragma once

#include <Eigen/Dense>

#include "core/interval.cuh"
#include "core/transform.cuh"
#include "container/optional.cuh"

namespace rt {

  class Plane {
  public:

    __host__ __device__ Plane(float intercept,
			      Eigen::Vector3f normal)
      : intercept_(intercept)
      , normal_(normal)
    { }

    __device__ bool can_hit(const Ray &ray) const
    {
      return false; // TODO implement
    }

    __device__ Optional<HitResult> hit(const Ray &ray,
				       const Interval &t_interval) const
    {
      return {}; // TODO implement
    }

    __host__ __device__ void apply_transform(const Transform &t) const
    {
      // TODO implement
    }

  private:
    float intercept_;
    Eigen::Vector3f normal_;
  };
}
