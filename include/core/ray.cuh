#pragma once

#include <Eigen/Core>

namespace rt {

  class Ray {
  public:    
    __host__ __device__ Ray(const Eigen::Vector3f &o, const Eigen::Vector3f &d)
      : origin_(o)
      , direction_(d)
    { }

    __host__ __device__ const Eigen::Vector3f &origin() const { return origin_; }

    __host__ __device__ const Eigen::Vector3f &direction() const { return direction_; }

    __host__ __device__ Eigen::Vector3f &origin() { return origin_; }

    __host__ __device__ Eigen::Vector3f &direction() { return direction_; }

    __host__ __device__ Eigen::Vector3f point_at_param(float t) const
    {
      return origin_ + t * direction_;
    }
    
  private:
    Eigen::Vector3f origin_;
    Eigen::Vector3f direction_;
  };
}
