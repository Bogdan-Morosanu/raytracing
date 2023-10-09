#pragma once

#include "check_error.cuh"

namespace rt {
  template <typename T>
  class DeviceSharedBufferView;
  
  template <typename T>
  class SharedBuffer {

    template <typename SameT>
    friend class DeviceSharedBufferView;

  public:

    template <typename IsConstructible =
	      typename std::enable_if<std::is_default_constructible<T>::value>::type>
    __host__
    SharedBuffer()
      : SharedBuffer(T{})
    { }
    
    __host__ explicit
    SharedBuffer(T t)
      : ptr_(nullptr)
    {
      checkCudaErrors(cudaMallocManaged((void **)&ptr_, sizeof(T)));
      new (ptr_) T(std::move(t));
    }

    __host__ SharedBuffer(SharedBuffer &&other)
      : ptr_(other.ptr_)
    {
      other.ptr_ = nullptr;
    }

    __host__ SharedBuffer &operator = (SharedBuffer &&other)
    {
      // prevent blowup on *this = std::move(*this);
      if (this != &other) {
	this->~SharedBuffer();
	new (this) SharedBuffer(std::move(other));
      }

      return *this;
    }
    
    __host__ T &value() { return *ptr_; }

    __host__ const T &value() const { return *ptr_; }

    __host__ ~SharedBuffer()
    {
      if (ptr_) {
	checkCudaErrors(cudaFree(ptr_));
      }
    }

  private:
    T *ptr_;
  };

  template <typename T>
  class DeviceSharedBufferView {
  public:

    __host__ DeviceSharedBufferView(SharedBuffer<T> &buff)
      : ptr_(buff.ptr_)
    { }

    __device__ T &value() { return *ptr_; }

    __device__ const T &value() const { return *ptr_; }

  private:
    T *ptr_;
  };

  template <typename T>
  SharedBuffer<T> make_gpu_shared(T t)
  {
    return SharedBuffer<T>(std::move(t));
  }

  template <typename T>
  DeviceSharedBufferView<T> make_view(SharedBuffer<T> &buff)
  {
    return DeviceSharedBufferView<T>{buff};
  }
}
