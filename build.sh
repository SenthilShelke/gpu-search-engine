#!/bin/bash

echo "[Compiler] Building Metal Library..."
xcrun -sdk macosx metal -c src/compute.metal -o compute.air
xcrun -sdk macosx metallib compute.air -o default.metallib

echo "[Compiler] Building C++ Engine..."
clang++ -std=c++17 -framework Metal -framework Foundation src/main.mm -o main

echo "[Compiler] Success! Executing..."
echo "-----------------------------------"
./main