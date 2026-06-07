#include <metal_stdlib>
using namespace metal;

kernel void dot_product(device float* db_vector [[buffer(0)]], device float* query_vector [[buffer(1)]], device float* results [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    float result = 0;
    for(int i = 0; i < 384; i++) {
        result += query_vector[i] * db_vector[(index * 384) + i];
    }
    results[index] = result;
}