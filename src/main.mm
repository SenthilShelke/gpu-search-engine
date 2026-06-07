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
    Matrix query = create_matrix("data/query.bin");

    return 0;
}