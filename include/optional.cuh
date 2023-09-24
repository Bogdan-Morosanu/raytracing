#pragma once

#include <cstdint>

namespace rt {

  template <typename T>
  class Optional {
  public:

    __host__ __device__
    Optional()
      : is_valid_(false)
      , buff_()
    { }

    template <typename... Params>
    __host__ __device__
    Optional(Params... p)
      : is_valid_(true)
      , buff_()
    {
      new (value_ptr()) T(std::forward<Params>(p)...);
    }

    __host__ __device__
    Optional(const Optional<T> &other)
      : is_valid_(other.is_valid_)
      , buff_()
    {
      if (is_valid_) {
	new (value_ptr()) T(*other.value_ptr());
      }
    }

    __host__ __device__
    Optional(Optional<T> &&other)
      : is_valid_(other.is_valid_)
      , buff_()
    {
      if (is_valid_) {
	new (value_ptr()) T(std::move(*other.value_ptr()));
      }
    }

    __host__ __device__
    Optional &operator = (const Optional &other)
    {
      // prevent blowup on *this = *this
      if (this != &other) {
	this->~Optional();
	new (this) Optional(other);
      }
    }

    __host__ __device__
    Optional &operator = (Optional &&other)
    {
      // prevent blowup on *this = *this
      if (this != &other) {
	this->~Optional();
	new (this) Optional(std::move(other));
      }
    }

    __host__ __device__
    T &value()
    {
      assert(is_valid_);
      return *value_ptr();
    }

    __host__ __device__
    const T &value() const
    {
      assert(is_valid_);
      return *value_ptr();
    }

    __host__ __device__
    bool is_valid() const { return is_valid_; }

    __host__ __device__
    ~Optional()
    {
      if (is_valid_) {
	value_ptr()->~T();
      }
    }

  private:

    __host__ __device__
    T *value_ptr()
    {
      return reinterpret_cast<T*>(buff_);
    }

    __host__ __device__
    const T *value_ptr() const
    {
      return reinterpret_cast<T*>(buff_);
    }

    bool is_valid_;
    alignas(alignof(T)) char buff_[sizeof(T)];
  };
}
