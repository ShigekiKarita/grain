#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

float sum_thrust(float* in, unsigned int n) {
    thrust::plus<float> binary_op;
    // compute sum on the device
    thrust::device_ptr<float> begin = thrust::device_pointer_cast(in);
    return thrust::reduce(begin, begin + n, 0, binary_op);
}