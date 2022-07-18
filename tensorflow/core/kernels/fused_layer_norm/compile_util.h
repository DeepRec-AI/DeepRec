#ifndef TENSORFLOW_CORE_KERNELS_FUSED_LAYER_NORMALIZE_COMPILE_UTIL_OP_H_
#define TENSORFLOW_CORE_KERNELS_FUSED_LAYER_NORMALIZE_COMPILE_UTIL_OP_H_

#include <type_traits>

// A class for forced loop unrolling at compile time
template <int i>
struct compile_time_for {
    template <typename Lambda, typename... Args>
    inline static void op(const Lambda& function, Args... args) {
        compile_time_for<i-1>::op(function, args...);
        function(std::integral_constant<int, i-1>{}, args...);
    }
};
template <>
struct compile_time_for<1> {
    template <typename Lambda, typename... Args>
    inline static void op(const Lambda& function, Args... args) {
        function(std::integral_constant<int, 0>{}, args...);
    }
};
template <>
struct compile_time_for<0> { 
    // 0 loops, do nothing
    template <typename Lambda, typename... Args>
    inline static void op(const Lambda& function, Args... args) {
    }
};

#endif  // TENSORFLOW_CORE_KERNELS_FUSED_LAYER_NORMALIZE_COMPILE_UTIL_OP_H_