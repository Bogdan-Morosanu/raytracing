#pragma once

#include "ray.cuh"

namespace rt {
  class Viewport {
  public:

    __host__ __device__ Viewport(const Eigen::Vector3f &o,
				 const Eigen::Vector3f &h,
				 const Eigen::Vector3f &v,
				 const Eigen::Vector3f &dl)
      : origin_(o)
      , horizontal_dir_(h)
      , vertical_dir_(v)
      , down_left_(dl)
    { }

    __device__ Ray compute_ray(float h_percentage, float v_percentage) const
    {
      Eigen::Vector3f dir = down_left_ + h_percentage * horizontal_dir_ + v_percentage * vertical_dir_;
      return Ray(origin_, dir);
    }
    
  private:
    Eigen::Vector3f origin_;
    Eigen::Vector3f horizontal_dir_;
    Eigen::Vector3f vertical_dir_;
    Eigen::Vector3f down_left_;
  };
}