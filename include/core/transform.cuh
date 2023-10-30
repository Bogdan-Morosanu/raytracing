#pragma once

#include <Eigen/Dense>

namespace rt {
  class Transform {
  public:
    __host__ __device__ Transform(Eigen::Matrix3f r, Eigen::Vector3f t)
      : r_(r)
      , t_(t)
    { }

    __host__ __device__ Eigen::Vector3f operator()(const Eigen::Vector3f &v) const
    {
      return r_ * v + t_;
    }
    
  private:
    Eigen::Matrix3f r_;
    Eigen::Vector3f t_;
  };
}
