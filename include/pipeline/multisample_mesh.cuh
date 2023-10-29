#pragma once

#include <array>
#include <Eigen/Dense>

#include "container/alloc.cuh"

namespace rt {

  class MultisampleMesh {
  public:
    static constexpr std::size_t WIDTH = 11;
    static constexpr std::size_t HEIGHT = 11;
    
    using Samples = DeviceArray<Eigen::Vector2f, WIDTH * HEIGHT>;
    
    __host__ MultisampleMesh(float norm_x, float norm_y)
      : offsets_{}
    {
      constexpr auto DIV_H = float(HEIGHT + 2);
      constexpr auto DIV_W = float(WIDTH + 2);
      
      for (std::size_t r = 0u; r < HEIGHT; ++r) {
      	float y = (-1.0f + 2.0f * float(r+1) / float(DIV_H)) / norm_y;

      	for (std::size_t c = 0u; c < WIDTH; ++c) {
      	  float x = (-1.0f + 2.0f * float(c+1) / float(DIV_W)) / norm_x;
      	  offsets_[r * WIDTH + c] = Eigen::Vector2f{x, y};
      	}
      }
    }
    
    __device__ Samples generate_samples(Eigen::Vector2f pixel) const
    {
      Samples retval;

      for (auto i = 0; i < WIDTH * HEIGHT; ++i) {
	retval[i] = pixel + offsets_[i];
      }

      return retval;
    }
  private:
    Eigen::Vector2f offsets_[WIDTH * HEIGHT];
  };
}
