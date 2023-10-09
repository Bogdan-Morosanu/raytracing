#pragma once

#include <array>
#include <Eigen/Dense>

#include "container/alloc.cuh"

namespace rt {

  class MultisampleMesh {
  public:

    using Samples = DeviceArray<Eigen::Vector2f, 9>;
    
    __device__ MultisampleMesh(float norm_x, float norm_y)
      : offsets_{Eigen::Vector2f{-0.5f, -0.5f}, Eigen::Vector2f{0.0f, -0.5f}, Eigen::Vector2f{0.5f, -0.5f},
                 Eigen::Vector2f{-0.5f,  0.0f}, Eigen::Vector2f{0.0f,  0.0f}, Eigen::Vector2f{0.5f,  0.0f},
                 Eigen::Vector2f{-0.5f,  0.5f}, Eigen::Vector2f{0.0f,  0.5f}, Eigen::Vector2f{0.5f,  0.5f}}
    {
      for (auto &o : offsets_) {
	o.x() /= norm_x;
	o.y() /= norm_y;
      }
    }
    
    __device__ Samples generate_samples(Eigen::Vector2f pixel) const
    {
      Samples retval;

      for (auto i = 0; i < 9; ++i) {
	retval[i] = pixel + offsets_[i];
      }

      return retval;
    }
  private:
    Eigen::Vector2f offsets_[9];
  };
}
