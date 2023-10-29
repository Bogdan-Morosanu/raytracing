#pragma once

#include <cstdint>
#include <new>

#include <Eigen/Dense>

#include "cuda_utils/check_error.cuh"

namespace rt {
  template <typename T>
  class VectorView;

  template <typename T>
  class Vector {
  public:

    template <typename SameT>
    friend class VectorView;
    
    using ValueType = T;

    __host__ Vector()
      : header_(nullptr)
    {
      checkCudaErrors(cudaMallocManaged((void **)&header_, sizeof(Header)));
      new (header_) Header{0u, 0u, nullptr};
    }

    __host__ Vector(std::initializer_list<ValueType> init_vals)
      : Vector()
    {
      for (auto &v : init_vals) {
	push_back(std::move(v));
      }
    }

    __host__ Vector(const Vector&) = delete;
    __host__ Vector& operator = (const Vector&) = delete;

    __host__ Vector(Vector &&other)
      : header_(other.header_)
    {
      other.header_ = nullptr;
    }

    __host__ Vector &operator = (Vector &&other)
    {
      // guard against this = std::move(*this) blowing our object up
      if (this != &other) {
	this->~Vector();
	new (this) Vector(std::move(other));
      }
      
      return *this;
    }

    __host__ __device__ ValueType *data()
    {
      assert(header_);
      return header_->buffer;
    }

    __host__ __device__ const ValueType *data() const
    {
      assert(header_);
      return header_->buffer;
    }

    __host__ __device__ ValueType *begin()
    {
      assert(header_);
      return header_->buffer;
    }

    __host__ __device__ ValueType *end()
    {
      assert(header_);
      return header_->buffer + header_->alloc_size;
    }

    __host__ __device__ const ValueType *begin() const
    {
      assert(header_);
      return header_->buffer;
    }

    __host__ __device__ const ValueType *end() const
    {
      assert(header_);
      return header_->buffer + header_->alloc_size;
    }

    __host__ __device__ ValueType &operator[](std::size_t i)
    {
      assert(header_);
      assert(header_->buffer);
      assert(i < header_->buffer.alloc_size);
      return header_->buffer[i];
    }

    __host__ __device__ ValueType &operator[](std::size_t i) const
    {
      assert(header_);
      assert(header_->buffer);
      assert(i < header_->buffer.alloc_size);
      return header_->buffer[i];
    }
    
    __host__ __device__ std::size_t size() const
    {
      assert(header_);
      return header_->alloc_size;
    }

    __host__ void push_back(ValueType val)
    {
      assert(header_);
      if (header_->capacity == header_->alloc_size) {
	increase_alloc();
      }

      new (&header_->buffer[header_->alloc_size]) ValueType(std::move(val));
      header_->alloc_size++;
    }
    
    __host__ ~Vector()
    {
      if (header_) {
	free_buffer();
	checkCudaErrors(cudaFree(header_));
      }
    }
    
  private:

    __host__ void increase_alloc()
    {
      assert(header_);

      auto new_capacity = 2 * (header_->capacity + 1);
      auto alloc_size = (new_capacity * sizeof(ValueType));
      ValueType *new_buffer;
      checkCudaErrors(cudaMallocManaged((void **)&(new_buffer), alloc_size));

      for (auto i = 0u; i < header_->alloc_size; ++i) {
	new (&new_buffer[i]) ValueType(std::move(header_->buffer[i]));
      }

      free_buffer();

      header_->capacity = new_capacity;
      header_->buffer = new_buffer;
    }

    __host__ void free_buffer()
    {
      assert(header_);
      
      if (header_->buffer) {
	// destruct in reverse order of construction
	for (auto i = header_->alloc_size; i > 0; --i) {
	  header_->buffer[i-1].~ValueType();
	}
      }
 
      checkCudaErrors(cudaFree(header_->buffer));
    }
    
    struct Header {
      std::size_t capacity;
      std::size_t alloc_size;
      ValueType *buffer;
    };

    Header *header_;
  };

  template <typename T>
  class VectorView {
  public:

    using ValueType = typename Vector<T>::ValueType;
    
    __host__ VectorView(Vector<T> &data)
      : header_(data.header_)
    { }
    
    __host__ __device__ ValueType *data()
    {
      assert(header_);
      return header_->buffer;
    }

    __host__ __device__ const ValueType *data() const
    {
      assert(header_);
      return header_->buffer;
    }

    __host__ __device__ ValueType *begin()
    {
      assert(header_);
      return header_->buffer;
    }

    __host__ __device__ ValueType *end()
    {
      assert(header_);
      return header_->buffer + header_->alloc_size;
    }

    __host__ __device__ const ValueType *begin() const
    {
      assert(header_);
      return header_->buffer;
    }

    __host__ __device__ const ValueType *end() const
    {
      assert(header_);
      return header_->buffer + header_->alloc_size;
    }

    __host__ __device__ ValueType &operator[](std::size_t i)
    {
      assert(header_);
      assert(header_->buffer);
      assert(i < header_->buffer.alloc_size);
      return header_->buffer[i];
    }

    __host__ __device__ const ValueType &operator[](std::size_t i) const
    {
      assert(header_);
      assert(header_->buffer);
      assert(i < header_->alloc_size);
      return header_->buffer[i];
    }
    
    __host__ __device__ std::size_t size() const
    {
      assert(header_);
      return header_->alloc_size;
    }

  private:
    const typename Vector<T>::Header *header_;
  };

  template <typename T>
  VectorView<T> make_view(Vector<T> &v)
  {
    return VectorView<T>{v};
  }
}
