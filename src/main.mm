#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

using namespace std;

class Matrix {
    public:
        int rows;
        int cols;
        vector<float> data;
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
    vector<float> data;
    data.resize(size);
    bin_file.read(reinterpret_cast<char*>(data.data()), size * sizeof(float));

    Matrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.data = data;

    return matrix;
}

int main() {

    Matrix db = create_matrix("data/vector_db.bin");
    Matrix query = create_matrix("data/vector_db.bin");
    int db_bytes = db.data.size() * sizeof(float);
    int query_bytes = query.data.size() * sizeof(float);
    int result_bytes = db.rows * sizeof(float);

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> command_queue = [device newCommandQueue];
    id<MTLBuffer> db_buffer = [device newBufferWithBytes:db.data.data() length:db_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> query_buffer = [device newBufferWithBytes:query.data.data() length:query_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> result_buffer = [device newBufferWithLength:result_bytes options:MTLResourceStorageModeShared];

    id<MTLLibrary> library = [device newDefaultLibrary];
    id<MTLFunction> dot_product = [library newFunctionWithName:@"dot_product"];
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline_state = [device newComputePipelineStateWithFunction:dot_product error:&error];
    if(!pipeline_state) {
        cout << "Failed to create pipeline state." << endl;
        exit(1);
    }

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
    for(int i = 0; i < 5; i++) {
        cout << results[i] << endl;
    } 



    return 0;
}