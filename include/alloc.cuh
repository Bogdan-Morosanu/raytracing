#pragma once

#include <cstdint>

#include "check_error.cuh"

namespace rt {
  class ImageBuffer {
  public:
    ImageBuffer(std::size_t w, std::size_t h)
      : width_(w)
      , height_(h)
      , alloc_size_(w * h * 3 * sizeof(float))
      , buffer_(nullptr)
    {
      checkCudaErrors(cudaMallocManaged((void **)&buffer_, alloc_size_));
    }

    float *buffer() { return buffer_; }

    const float *buffer() const  { return buffer_; }

    std::size_t width() const { return width_; }

    std::size_t height() const { return height_; }
    
    ~ImageBuffer()
    {
      checkCudaErrors(cudaFree(buffer_));
    }
    
  private:
    std::size_t width_;
    std::size_t height_;
    std::size_t alloc_size_;
    float *buffer_;
  };
}
