#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <string>
#include <fstream>
#include <chrono>
#include <utility>
#include <memory>

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

using namespace std;

class Matrix {
    public:
        int rows;
        int cols;
        unique_ptr<float[]> data;
};

class Pair {
    public:
        int index;
        float value;

        Pair(int _index, float _value) : index(_index), value(_value) {}

        bool operator>(const Pair& other) const {
            return value > other.value;
        }
};

Matrix create_matrix(string file) {
    ifstream bin_file(file, ios::binary);

    if(!bin_file.is_open()) {
        cout << "Could not open file: " << file << endl;
        exit(1);
    }

    int header[2];
    bin_file.read(reinterpret_cast<char*>(header), sizeof(header));
    int rows = header[0];
    int cols = header[1];

    size_t size = rows * cols;
   
    unique_ptr<float[]> data(new float[size]);
    bin_file.read(reinterpret_cast<char*>(data.get()), size * sizeof(float));


    Matrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.data = std::move(data);

    return matrix;
}

int main() {

    Matrix db = create_matrix("data/vector_db.bin");

    int db_bytes = db.rows * db.cols * sizeof(float);
    int query_bytes = 384 * sizeof(float);
    int result_bytes = db.rows * sizeof(float);

    id<MTLDevice> gpu = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> command_queue = [gpu newCommandQueue];
    id<MTLBuffer> db_buffer = [gpu newBufferWithBytes:db.data.get() length:db_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> result_buffer = [gpu newBufferWithLength:result_bytes options:MTLResourceStorageModeShared];

    id<MTLLibrary> library = [gpu newDefaultLibrary];
    id<MTLFunction> dot_product = [library newFunctionWithName:@"dot_product"];
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline_state = [gpu newComputePipelineStateWithFunction:dot_product error:&error];
    if(!pipeline_state) {
        cout << "Failed to create pipeline state." << endl;
        exit(1);
    }

    vector<float> query(384);
    while(cin.read(reinterpret_cast<char*>(query.data()), 384 * sizeof(float))) {

        int query_bytes = 384 * sizeof(float);
        id<MTLBuffer> query_buffer = [gpu newBufferWithBytes:query.data() length:query_bytes options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline_state];
        [encoder setBuffer:db_buffer offset:0 atIndex:0];
        [encoder setBuffer:query_buffer offset:0 atIndex:1];
        [encoder setBuffer:result_buffer offset:0 atIndex:2];

        MTLSize grid_size = MTLSizeMake(db.rows, 1, 1);
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        float* results = (float*)result_buffer.contents;
        priority_queue<Pair, vector<Pair>, greater<Pair>> min_heap;

        for(int i = 0; i < 5; i++) {
            Pair pair = Pair(i, results[i]);
            min_heap.push(pair);
        }
        for(int i = 5; i < 50000; i++) {
            if(results[i] > min_heap.top().value) {
                Pair pair = Pair(i, results[i]);
                min_heap.pop();
                min_heap.push((pair));
            } else {
                continue;
            }
        }

        vector<int> top_indices(5);
        for(int i = 0; i < 5; i++) {
            top_indices[i] = min_heap.top().index;
            min_heap.pop();
        }

        cout.write(reinterpret_cast<char*>(top_indices.data()), 5 * sizeof(int));
        cout.flush();

    }

    return 0;
}