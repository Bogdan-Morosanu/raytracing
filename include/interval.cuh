#pragma once

namespace rt {
  class Interval {

    enum Mode {
      LOWER_CLOSED_UPPER_CLOSED,
      LOWER_CLOSED_UPPER_OPEN,
      LOWER_OPEN_UPPER_CLOSED,
      LOWER_OPEN_UPPER_OPEN,
    };

  public:
    __host__ __device__
    Interval(float lower, float upper,
	     bool lower_closed = true,
	     bool upper_closed = true)
      : lower_(lower)
      , upper_(upper)
      , mode_((lower_closed) ?
	      (upper_closed ? LOWER_CLOSED_UPPER_CLOSED : LOWER_CLOSED_UPPER_OPEN) :
	      (upper_closed ? LOWER_OPEN_UPPER_CLOSED : LOWER_OPEN_UPPER_OPEN))
    {}

    __host__ __device__ bool contains(float f) const
    {
      switch(mode_) {
      case LOWER_CLOSED_UPPER_CLOSED:
	return (lower_ <= f) && (f <= upper_);

      case LOWER_CLOSED_UPPER_OPEN:
	return (lower_ <= f) && (f < upper_);

      case LOWER_OPEN_UPPER_CLOSED:
	return (lower_ < f) && (f <= upper_);

      default: // fallthrough, for nvcc compiler warning
      case LOWER_OPEN_UPPER_OPEN:
	return (lower_ < f) && (f < upper_);

      }
    }
    
  private:
    float lower_;
    float upper_;
    Mode mode_;
  };
}
