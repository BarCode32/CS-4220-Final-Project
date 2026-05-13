# CUDA OpenGL N-Body Simulation 

This project implements a real time GPU based solution to a 2D N-body paericle simulation using CUDA for comuptations and OpenGL for rendering.

## How to Run

Compile the program with nvcc:

```bash
nvcc -O3 --use_fast_math nbody_sim.cu glad/glad.c -o sim -lglfw -lGL -ldl
```
Run the executable:
```bash
./sim
```

Controls:
- Press ESC to exit the program
- Press UP ARROW to zoom in
- Press DOWN ARROW to zoom out

## Required Packages / Dependencies

This project requires:

- CUDA Toolkit
- NVIDIA GPU driver
- OpenGL
- GLFW
- GLAD
- C++ compiler compatible with nvcc

Confirmed to work on Unntu 22.04 Linux, but may work on other systems

GLAD files must be included in the project directory. The file tree expects:

project/glad/glad.h
project/glad/glad.c

The code uses CUDA/OpenGL interoperability, so the CUDA device and OpenGL rendering device should refer to the same NVIDIA GPU.

## Main CUDA Kernel / Experiment Location

The main CUDA simulation code is located in nbody_sim.cu.
The main cuda kernels exist between lines: 81-159
- integrateKernel
- accelKernelTiled (function uses shared memory & Tiling)
- interaction (called inside accelKernelTiled)



### Main Loop
The main function starts on line 217

The main loop starts n like 358 and is inside 
```bash
while (!glfwWindowShouldClose(window)) {
    ...
}
```

## Benchmark code
The program prints benchmark results every 60 frames, including FPS, average frame time, CUDA computation time, render time, and billions of interactions per second.
The section where the calculations for the benchmarks is done is between lines: 418-459

The Cuda Event timers can be found curing the cuda function calls between lines: 384-398


## Notes

The number of particles is currently set with:

constexpr int N = 170000;

The CUDA block size is set with:

constexpr int BLOCK_SIZE = 256;

Changing these values will affect performance, memory usage, and simulation speed.
