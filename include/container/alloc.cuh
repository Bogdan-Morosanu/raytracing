#pragma once

#include <cstdint>
#include <new>

#include <Eigen/Dense>

#include "cuda_utils/check_error.cuh"

namespace rt {
  class ImageBufferView;

  class ImageBuffer {
  public:

    friend class ImageBufferView;
    
    using ValueType = Eigen::Vector3f;

    __host__ ImageBuffer(std::size_t w, std::size_t h)
      : header_(nullptr)
      , buffer_(nullptr)	
    {
      // TODO checkout best buffer alignment for runtime perf
      // pad pixel data to be aligned 64 bytes
      auto align = 64;
      auto padding = align - 1;
      auto alloc_size = (sizeof(Header) + padding + w * h * sizeof(ValueType));

      checkCudaErrors(cudaMallocManaged((void **)&header_, alloc_size));
      new (header_) Header{w, h, alloc_size};

      auto align_offset = align - reinterpret_cast<std::uintptr_t>(header_) % align;
      buffer_ = reinterpret_cast<ValueType*>(
                  reinterpret_cast<char*>(header_) + sizeof(Header) + align_offset
		);
    }

    __host__ ImageBuffer(const ImageBuffer&) = delete;
    __host__ ImageBuffer& operator = (const ImageBuffer&) = delete;

    __host__ ImageBuffer(ImageBuffer &&other)
      : header_(other.header_)
      , buffer_(other.buffer_)
    {
      other.header_ = nullptr;
      other.buffer_ = nullptr;
    }

    __host__ ImageBuffer &operator = (ImageBuffer &&other)
    {
      // guard against this = std::move(*this) blowing our object up
      if (this != &other) {
	this->~ImageBuffer();
	new (this) ImageBuffer(std::move(other));
      }
      
      return *this;
    }

    __host__ __device__ ValueType *buffer() { return buffer_; }

    __host__ __device__ const ValueType *buffer() const  { return buffer_; }

    __host__ __device__ std::size_t width() const { return header_->width_; }

    __host__ __device__ std::size_t height() const { return header_->height_; }

    __host__ ~ImageBuffer()
    {
      if (header_) {
	checkCudaErrors(cudaFree(header_));
      }
    }
    
  private:

    struct Header {
      std::size_t width_;
      std::size_t height_;
      std::size_t alloc_size_;
    };

    Header *header_;
    ValueType *buffer_;
  };

  class ImageBufferView {
  public:
    __host__ ImageBufferView(ImageBuffer &data)
      : header_(data.header_)
      , buffer_(data.buffer_)
    { }

    __host__ __device__ ImageBuffer::ValueType *buffer() { return buffer_; }

    __host__ __device__ const ImageBuffer::ValueType *buffer() const  { return buffer_; }

    __host__ __device__ std::size_t width() const { return header_->width_; }

    __host__ __device__ std::size_t height() const { return header_->height_; }
    
  private:
    ImageBuffer::Header *header_;
    ImageBuffer::ValueType *buffer_;
  };

  template <typename T, std::size_t N>
  class ArrayView;
  
  template <typename T, std::size_t N>
  class Array {
  public:

    
    template <typename SameT, std::size_t SameN>
    friend class ArrayView;

    __host__ Array(std::initializer_list<T> init_vals)
      : data_(alloc())
    {
      assert(init_vals.size() <= N);

      int i = 0;
      for (auto it = init_vals.begin(); it != init_vals.end(); ++it, ++i) {
	new (&data_[i]) T(std::move(*it));
      }

      for ( ; i < N; ++i) {
	new (&data_[i]) T();
      }
    }
		   
    __host__ Array()
      : data_(alloc())
    {

      for (auto i = 0u; i < N; ++i) {
	new (&data_[i]) T();
      }
    }

    // TODO add initialiser list ctor.
    
    __host__ __device__ T &operator[] (std::size_t i) { return data_[i]; }

    __host__ __device__ const T &operator[] (std::size_t i) const { return data_[i]; }
    
    __host__ __device__ T *begin() { return &data_[0]; }

    __host__ __device__ T *end() { return &data_[N]; }

    __host__ __device__ const T *begin() const { return &data_[0]; }

    __host__ __device__ const T *end() const { return &data_[N]; }
    
    __host__ ~Array()
    {
      if (data_) {
	for (auto i = N; i > 0; --i) {
	  data_[i-1].~T();
	}

	checkCudaErrors(cudaFree(data_));
      }
    }
    
  private:

    __host__ __device__ T *alloc()
    {
      T *ptr;

      auto alloc_size = sizeof(T[N]);
      checkCudaErrors(cudaMallocManaged((void **)&ptr, alloc_size));

      return ptr;
    }
    
    T *data_;
  };

  template <typename T, std::size_t N>
  class ArrayView {
  public:
    __host__ ArrayView(Array<T, N> &arr)
      : data_(arr.data_)
    { }

    __host__ __device__ T &operator[] (std::size_t i) { return data_[i]; }

    __host__ __device__ const T &operator[] (std::size_t i) const { return data_[i]; }

    __host__ __device__ T *begin() { return &data_[0]; }

    __host__ __device__ T *end() { return &data_[N]; }

    __host__ __device__ const T *begin() const { return &data_[0]; }

    __host__ __device__ const T *end() const { return &data_[N]; }
    
  private:
    T *data_;
  };

  template <typename T, std::size_t N>
  ArrayView<T, N> make_view(Array<T, N> &array)
  {
    return ArrayView<T, N>(array);
  }
  
  template <typename T, std::size_t N>
  class DeviceArray {
  public:

    __device__ DeviceArray()
      : data_()
    {}

    __device__ std::size_t size() const { return N; }

    __device__ T &operator [](std::size_t i) { return data_[i]; }

    __device__ const T &operator [](std::size_t i) const { return data_[i]; }

    __device__ T *begin() { return data_; }

    __device__ const T *begin() const { return data_; }

    __device__ T *end() { return data_ + N; }

    __device__ const T *end() const { return data_ + N; }

  private:
    T data_[N];
  };
}
