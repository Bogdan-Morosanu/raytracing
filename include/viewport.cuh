#pragma once

#include "ray.cuh"

namespace rt {
  class Viewport {
  public:

    __host__ __device__ Viewport(const Eigen::Vector3f &look_from,
				 const Eigen::Vector3f &look_at,
				 const Eigen::Vector3f &up,
				 float vertical_fov,
				 float aspect)
      : origin_(look_from)
      , horizontal_dir_()
      , vertical_dir_()
      , up_left_()
    {
      float theta = vertical_fov * M_PI / 180.0f;
      float half_height = tan(theta/2);
      float half_width = aspect * half_height;
      Eigen::Vector3f w = (look_from - look_at).normalized();
      Eigen::Vector3f u = up.cross(w).normalized();
      Eigen::Vector3f v = w.cross(u);
      up_left_ = origin_ - half_width * u - half_height * v - w;
      horizontal_dir_ = 2.0f * half_width * u;
      vertical_dir_ = 2.0f * half_height * v;
    }

    __device__ Ray compute_ray(float h_percentage, float v_percentage) const
    {
      Eigen::Vector3f dir = up_left_ + h_percentage * horizontal_dir_ + v_percentage * vertical_dir_ - origin_;
      return Ray(origin_, dir); 
    }

    // __host__ __device__ Viewport translate(Eigen::Vector3f offset) const
    // {
    //   return Viewport(origin_ + offset,
    // 		      horizontal_dir_,
    // 		      vertical_dir_,
    // 		      up_left_ + offset);
    // }
  private:
    Eigen::Vector3f origin_;
    Eigen::Vector3f horizontal_dir_;
    Eigen::Vector3f vertical_dir_;
    Eigen::Vector3f up_left_;
  };
}
