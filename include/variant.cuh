#pragma once

#include <cstdlib>
#include <type_traits>
#include <utility>

namespace rt {

  namespace internal {
    template <typename... Args>
    struct VariantStorageRequirements;

    template <typename Arg >
    struct VariantStorageRequirements<Arg> {
      static constexpr std::size_t size = sizeof(Arg);
      static constexpr std::size_t align = alignof(Arg);
    };

    template <typename Arg, typename... Args>
    struct VariantStorageRequirements<Arg, Args...> {
      static constexpr std::size_t sizeof_current = sizeof(Arg);
      static constexpr std::size_t alignof_current = alignof(Arg);
      static constexpr std::size_t sizeof_rest = VariantStorageRequirements<Args...>::size;
      static constexpr std::size_t alignof_rest = VariantStorageRequirements<Args...>::align;

    public:
      static constexpr std::size_t size = (sizeof_current > sizeof_rest) ? sizeof_current : sizeof_rest;
      static constexpr std::size_t align = (alignof_current > alignof_rest) ? alignof_current : alignof_rest;
      
    };

    template <std::size_t Idx, typename... Args>
    struct VariantIndexer;

    // recursive case
    template <std::size_t Idx, typename Arg, typename... Args>
    struct VariantIndexer<Idx, Arg, Args...> {
      using Type = typename VariantIndexer<Idx-1, Args...>::Type;
    };

    // base case
    template <typename Arg, typename... Args>
    struct VariantIndexer<0, Arg, Args...> {
      using Type = Arg;
    };

    // error check
    template <std::size_t Idx>
    struct VariantIndexer<Idx> {
      static_assert(Idx != Idx, "Variant type list index out of bounds!");
    };

    template <std::size_t Idx, typename... Args>
    struct IndexOfImpl;

    // recursive case
    template <std::size_t Idx, typename Needle, typename Head, typename... Tail>
    struct IndexOfImpl<Idx, Needle, Head, Tail...> {
      static constexpr std::size_t value = IndexOfImpl<Idx+1, Needle, Tail...>::value;
    };

    // base case
    template <std::size_t Idx, typename Needle, typename... Tail>
    struct IndexOfImpl<Idx, Needle, Needle, Tail...> {
      static constexpr std::size_t value = Idx;
    };

    // error check
    template <std::size_t Idx, typename Needle>
    struct IndexOfImpl<Idx, Needle> {
      static_assert(Idx != Idx, "Type not found in variant parameter pack!");
    };
    
    template <typename Needle, typename... Haystack>
    struct IndexOf {
      static constexpr std::size_t value = IndexOfImpl<0, Needle, Haystack...>::value;
    };

    template <typename A, typename B>
    struct IsSame {
      static constexpr bool value = false;
    };

    template <typename A>
    struct IsSame<A, A> {
      static constexpr bool value = true;
    };

    template <typename... Args>
    struct AllSame;

    // recursive case
    template <typename Arg1, typename Arg2, typename... Args>
    struct AllSame<Arg1, Arg2, Args...> {
      static constexpr bool value = IsSame<Arg1, Arg2>::value && AllSame<Arg2, Args...>::value;
    };

    // base case
    template <typename Arg>
    struct AllSame<Arg> {
      static constexpr bool value = true;
    };

    template <typename Function, typename Arg>
    struct InvokeResult {
      using Type = decltype(std::declval<Function>()(*(Arg*)nullptr));
    };
    
    template <typename Function, typename... Args>
    struct VisitResult {
      static_assert(AllSame<typename InvokeResult<Function, Args>::Type...>::value,
		    "Visitor function must return the same value across all argument types!");

      using Type = typename InvokeResult<Function, typename VariantIndexer<0, Args...>::Type>::Type;
    };
    
    template <typename... OtherArgs>
    struct VisitImpl;
  }
  
  template <typename... Args>
  class Variant {

    template <typename... OtherArgs>
    friend struct internal::VisitImpl;
    
    static constexpr std::size_t size = internal::VariantStorageRequirements<Args...>::size;
    static constexpr std::size_t align = internal::VariantStorageRequirements<Args...>::align;
    static constexpr std::size_t pack_length = sizeof...(Args);
  public:
    __host__ __device__ Variant();
    
    template <typename T, typename Guard = 
	      typename std::enable_if<!internal::AllSame<T, Variant<Args...>>::value>::type>
    __host__ __device__ Variant(T &&t);

    __host__ __device__ Variant(const Variant &);

    __host__ __device__ Variant &operator = (const Variant &);
    
    template <typename Function>
    __host__ __device__
    typename internal::VisitResult<Function, Args...>::Type
    visit(Function &&f);

    template <typename Function>
    __host__ __device__
    typename internal::VisitResult<Function, Args...>::Type
    visit(Function &&f) const;

    template <typename T>
    __host__ __device__ T &get();

    template <typename T>
    __host__ __device__ const T &get() const;

    __host__ __device__ std::size_t current() const;

    __host__ __device__ bool is_empty() const;
    
    __host__ __device__ ~Variant();
    
  private:

    template<typename Function, std::size_t Index>
    __host__ __device__
    typename internal::VisitResult<Function, Args...>::Type
    visit_impl(Function &&f);

    template<typename Function, std::size_t Index>
    __host__ __device__
    typename internal::VisitResult<Function, Args...>::Type
    visit_impl(Function &&f) const;

    template <std::size_t Index>
    __host__ __device__
    typename internal::VariantIndexer<Index, Args...>::Type *ptr();

    template <std::size_t Index>
    __host__ __device__
    const typename internal::VariantIndexer<Index, Args...>::Type *ptr() const;

    std::size_t current_;
    alignas(align) std::uint8_t buff_[size];
  };

  template <typename... Args>
  __host__ __device__ Variant<Args...>::Variant()
    : current_(pack_length)
  { }

  template <typename... Args>
  template <typename T, typename Guard>
  __host__ __device__ Variant<Args...>::Variant(T &&t)
    : current_(internal::IndexOf<typename std::decay<T>::type, Args...>::value) 
  {
    using ValueType = typename std::decay<T>::type;
    constexpr auto idx = internal::IndexOf<ValueType, Args...>::value;
    new (ptr<idx>()) ValueType(std::forward<T>(t));
  }

  template <typename... Args>
  __host__ __device__ Variant<Args...>::Variant(const Variant &other)
    : current_(other.current_)
  {
    auto copy_visitor = [&other](auto &place)
			{
			  using Type = typename std::decay<decltype(place)>::type;
			  constexpr auto idx = internal::IndexOf<Type, Args...>::value;
			  new (&place) Type(*other.ptr<idx>());
			};

    this->visit(copy_visitor);
  }

  template <typename... Args>
  __host__ __device__ Variant<Args...>&
  Variant<Args...>::operator = (const Variant &other)
  {
    // guard against blowup on *this = *this
    if (this != &other) {
      this->~Variant();
      new (this) Variant(other);
    }

    return *this;
  }
  

  namespace internal {
    template <typename ... Args>
    struct VisitImpl;

    template <typename Arg, typename... Args>
    struct VisitImpl<Arg, Args...> {

      template <typename Function, typename... VariantArgs>
      __host__ __device__
      static auto do_visit(Function &&f, Variant<VariantArgs...> &v)
      {
	static constexpr std::size_t CurrentTypeIndex = internal::IndexOf<Arg, VariantArgs...>::value;
	if (CurrentTypeIndex == v.current_) {
	  return std::forward<Function>(f).operator()(*v.template ptr<CurrentTypeIndex>());

	} else {
	  return VisitImpl<Args...>::do_visit(std::forward<Function>(f), v);
	}
      }

      template <typename Function, typename... VariantArgs>
      __host__ __device__
      static auto do_visit(Function &&f, const Variant<VariantArgs...> &v)
      {
	static constexpr std::size_t CurrentTypeIndex = internal::IndexOf<Arg, VariantArgs...>::value;
	if (CurrentTypeIndex == v.current_) {
	  return std::forward<Function>(f).operator()(*v.template ptr<CurrentTypeIndex>());

	} else {
	  return VisitImpl<Args...>::do_visit(std::forward<Function>(f), v);
	}
      }

    };

    template <typename Arg>
    struct VisitImpl<Arg> {
      template <typename Function, typename... VariantArgs>
      __host__ __device__
      static auto do_visit(Function &&f, Variant<VariantArgs...> &v)
      {
	constexpr std::size_t last_elem = sizeof...(VariantArgs) - 1;
	assert(v.current_ == last_elem);
	return std::forward<Function>(f).template operator()(*v.template ptr<last_elem>());
      }

      template <typename Function, typename... VariantArgs>
      __host__ __device__
      static auto do_visit(Function &&f, const Variant<VariantArgs...> &v)
      {
	constexpr std::size_t last_elem = sizeof...(VariantArgs) - 1;
	assert(v.current_ == last_elem);
	return std::forward<Function>(f).template operator()(*v.template ptr<last_elem>());
      }
    };  
  }
  template <typename... Args>
  template <typename Function>
  __host__ __device__
  typename internal::VisitResult<Function, Args...>::Type
  Variant<Args...>::visit(Function &&f)
  {
    return internal::VisitImpl<Args...>::
      template do_visit(std::forward<Function>(f), *this);
  }
  
  template <typename... Args>
  template <typename Function>
  __host__ __device__
  typename internal::VisitResult<Function, Args...>::Type
  Variant<Args...>::visit(Function &&f) const
  {
    return internal::VisitImpl<Args...>::
      template do_visit(std::forward<Function>(f), *this);
  }

  template <typename... Args>
  template <typename T>
  __host__ __device__
  T &Variant<Args...>::get()
  {
    return *ptr<internal::IndexOf<T, Args...>::value>();
  }

  template <typename... Args>
  template <typename T>
  __host__ __device__
  const T &Variant<Args...>::get() const
  {
    return *ptr<internal::IndexOf<T, Args...>::value>();
  }

  template <typename... Args>
  __host__ __device__
  bool Variant<Args...>::is_empty() const
  {
    return current_ == pack_length;
  }
  
  template <typename... Args>
  __host__ __device__
  std::size_t Variant<Args...>::current() const
  {
    return current_;
  }

  template <typename... Args>
  __host__ __device__
  Variant<Args...>::~Variant() {
    if (!is_empty()) {
      auto destruct_function = [](auto& v) -> bool
			       {
				 using ValueType = typename std::decay<decltype(v)>::type;
				 v.~ValueType();
				 return true;
			       };

      visit(destruct_function);
    }
  }

  template <typename... Args>
  template <std::size_t Index>
  __host__ __device__
  typename internal::VariantIndexer<Index, Args...>::Type *
  Variant<Args...>::ptr()
  {
    using Type = typename internal::VariantIndexer<Index, Args...>::Type;
    return reinterpret_cast<Type*> (&buff_);
  }

  template <typename... Args>
  template <std::size_t Index>
  __host__ __device__
  const typename internal::VariantIndexer<Index, Args...>::Type *
  Variant<Args...>::ptr() const
  {
    using Type = typename internal::VariantIndexer<Index, Args...>::Type;
    return reinterpret_cast<const Type*> (&buff_);
  }
}
