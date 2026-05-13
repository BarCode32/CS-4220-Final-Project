// nvcc -O3 --use_fast_math nbody_sim.cu glad/glad.c -o sim -lglfw -lGL -ldl

#include "glad/glad.h"
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

#include <ctime>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

// ---------------- CUDA error checking ----------------

#define CHECK(call)                                           \
do {                                                          \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        std::cerr << "CUDA error at " << __FILE__ << ":"      \
                  << __LINE__ << " -> "                       \
                  << cudaGetErrorString(err) << std::endl;    \
        std::exit(EXIT_FAILURE);                              \
    }                                                         \
} while (0)

// ---------------- Constants ----------------

constexpr int N = 170000;

constexpr float G = 0.1f;
constexpr float PARTICLE_MASS = 1.0f;
constexpr float EPS2 = 1e-4f;
constexpr float DT = 1e-4f;

constexpr int BLOCK_SIZE = 256;

const unsigned int SCR_WIDTH = 2000;
const unsigned int SCR_HEIGHT = 1000;

// ---------------- OpenGL shader code ----------------

const char *vertexShaderSource = R"(
#version 330 core

layout (location = 0) in vec2 aPos;

uniform float scale;

void main()
{
    gl_Position = vec4(aPos.x * scale, aPos.y * scale, 0.0, 1.0);
    gl_PointSize = 3.0;
}
)";

const char *fragmentShaderSource = R"(
#version 330 core

out vec4 FragColor;

void main()
{
    float d = length(gl_PointCoord - vec2(0.5));

    if (d > 0.5)
        discard;

    float brightness = smoothstep(0.5, 0.0, d);
    brightness = brightness * brightness;

    FragColor = vec4(1.0, 0.65, 0.25, brightness);
}
)";


// ---------------- CUDA kernels ----------------
__device__  float2 interaction(float2 posi, float2 posj) {
    float dx = posj.x - posi.x;
    float dy = posj.y - posi.y;

    float distSqr = dx * dx + dy * dy + EPS2;
    float invDist = rsqrtf(distSqr);
    float invDist3 = invDist * invDist * invDist;

    float scalar = G * PARTICLE_MASS * invDist3;

    return make_float2(dx * scalar, dy * scalar);
}


__global__ void accelKernelTiled(const float2 *__restrict__ pos,float2 *__restrict__ acc) {
    __shared__ float2 sharedPos[BLOCK_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N) {
        return;
    }

    float2 pos_i = pos[i];
    float2 acc_i = make_float2(0.0f, 0.0f);

    int numTiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int tile = 0; tile < numTiles; ++tile) {
        int tileStart = tile * BLOCK_SIZE;
        int j = tileStart + threadIdx.x;

        if (j < N) {
            sharedPos[threadIdx.x] = pos[j];
        } else {
            sharedPos[threadIdx.x] = make_float2(0.0f, 0.0f);
        }

        __syncthreads();

        int tileCount = N - tileStart;

        if (tileCount >= BLOCK_SIZE) {
            #pragma unroll
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                float2 a = interaction(pos_i, sharedPos[k]);
                acc_i.x += a.x;
                acc_i.y += a.y;
            }
        } else {
            for (int k = 0; k < tileCount; ++k) {
                float2 a = interaction(pos_i, sharedPos[k]);
                acc_i.x += a.x;
                acc_i.y += a.y;
            }
        }

        __syncthreads();
    }

    acc[i] = acc_i;
}

// Convert acceleration into velocity into position
__global__ void integrateKernel(float2 *__restrict__ pos, float2 *__restrict__ vel, const float2 *__restrict__ acc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N) {
        return;
    }

    vel[i].x += acc[i].x * DT;
    vel[i].y += acc[i].y * DT;

    pos[i].x += vel[i].x * DT;
    pos[i].y += vel[i].y * DT;
}

// ---------------- Initialization ----------------

void randbodyInitialization(std::vector<float2> &h_pos, std::vector<float2> &h_vel) {
    float range = 2.0f;
    
    srand(time(nullptr));

    for (int i = 0; i < N; ++i) {
        float x = range*(2.0f * (float(rand()) / RAND_MAX) - 1.0f);
        float y = range*(2.0f * (float(rand()) / RAND_MAX) - 1.0f);

        float x_vel = range*(2.0f * (float(rand()) / RAND_MAX) - 1.0f);
        float y_vel = range*(2.0f * (float(rand()) / RAND_MAX) - 1.0f);

        h_pos[i] = make_float2(x, y);
        h_vel[i] = make_float2(x_vel, y_vel);
    }
}

// ---------------- GLFW callbacks ----------------

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void processInput(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
}

// ---------------- Shader helper ----------------

GLuint buildShaderProgram() {
    // Create and compile vertex shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);

    // Create and compile fragment shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);

    // Combine shader components
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}


int main() {
    // ---------------- OpenGL setup ----------------

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(
        SCR_WIDTH,
        SCR_HEIGHT,
        "CUDA OpenGL N-Body Constant Mass Benchmark",
        nullptr,
        nullptr
    );

    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    glfwSwapInterval(0);    // Disable vsync so FPS is not capped by monitor refresh rate.

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // Show if everying is using the correct GPU
    std::cout << "OpenGL Vendor: " << glGetString(GL_VENDOR) << std::endl;
    std::cout << "OpenGL Renderer: " << glGetString(GL_RENDERER) << std::endl;
    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;

    glEnable(GL_PROGRAM_POINT_SIZE);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    CHECK(cudaSetDevice(0));        //choose CUDA device after OpenGL context exists

    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, 0));

    std::cout << "CUDA Device: " << deviceProp.name << std::endl;
    std::cout << "N: " << N << std::endl;
    std::cout << "BLOCK_SIZE: " << BLOCK_SIZE << std::endl;

    GLuint shaderProgram = buildShaderProgram();
    GLint scaleLocation = glGetUniformLocation(shaderProgram, "scale");

    // ---------------- Initial particle data ----------------

    std::vector<float2> h_pos(N);
    std::vector<float2> h_vel(N);

    randbodyInitialization(h_pos, h_vel);

    // ---------------- Create OpenGL VBO for positions ----------------

    GLuint VAO;
    GLuint positionVBO;

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &positionVBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, positionVBO);
    glBufferData(
        GL_ARRAY_BUFFER,
        N * sizeof(float2),
        h_pos.data(),
        GL_DYNAMIC_DRAW
    );

    glVertexAttribPointer(
        0,
        2,
        GL_FLOAT,
        GL_FALSE,
        sizeof(float2),
        (void*)0
    );

    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // ---------------- Register OpenGL VBO with CUDA ----------------

    cudaGraphicsResource *cudaPositionResource = nullptr;
    CHECK(cudaGraphicsGLRegisterBuffer(&cudaPositionResource, positionVBO, cudaGraphicsMapFlagsNone));

    // ---------------- CUDA arrays ----------------

    float2 *d_vel = nullptr;
    float2 *d_acc = nullptr;

    CHECK(cudaMalloc(&d_vel, N * sizeof(float2)));
    CHECK(cudaMalloc(&d_acc, N * sizeof(float2)));

    CHECK(cudaMemcpy(d_vel, h_vel.data(), N * sizeof(float2), cudaMemcpyHostToDevice));

    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float cameraScale = 1.0f; // Zoom factor

    // ---------------- Benchmarking setup ----------------

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;

    CHECK(cudaEventCreate(&startEvent));
    CHECK(cudaEventCreate(&stopEvent));

    double frameMsAccum = 0.0;
    double simMsAccum = 0.0;
    double renderMsAccum = 0.0;

    int benchmarkFrameCount = 0;
    const int benchmarkReportInterval = 60;

    using clock_type = std::chrono::high_resolution_clock;

    std::cout << "\nBenchmark output every "
              << benchmarkReportInterval
              << " frames.\n"
              << std::endl;

    // ---------------- Simulation loop ----------------

    while (!glfwWindowShouldClose(window)) {
        auto frameStartCpu = clock_type::now();

        processInput(window);

        // Control camera zoom with arrow keys
        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
            cameraScale *= 1.1f;
        }
        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
            cameraScale *= 0.9f;
        }

        // ---------------- CUDA simulation ----------------

        CHECK(cudaGraphicsMapResources(1, &cudaPositionResource, 0));

        float2 *d_pos = nullptr;
        size_t numBytes = 0;

        CHECK(cudaGraphicsResourceGetMappedPointer(
            (void**)&d_pos,
            &numBytes,
            cudaPositionResource
        ));

        CHECK(cudaEventRecord(startEvent, 0));

        accelKernelTiled<<<gridSize, BLOCK_SIZE>>>(d_pos, d_acc);
        CHECK(cudaGetLastError());

        integrateKernel<<<gridSize, BLOCK_SIZE>>>(d_pos, d_vel, d_acc);
        CHECK(cudaGetLastError());

        CHECK(cudaEventRecord(stopEvent, 0));
        CHECK(cudaEventSynchronize(stopEvent));

        float simMs = 0.0f;
        CHECK(cudaEventElapsedTime(&simMs, startEvent, stopEvent));

        CHECK(cudaGraphicsUnmapResources(1, &cudaPositionResource, 0));

        // ---------------- OpenGL render ----------------

        auto renderStartCpu = clock_type::now();

        glClearColor(0.02f, 0.03f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shaderProgram);
        glUniform1f(scaleLocation, cameraScale);

        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, N);

        glfwSwapBuffers(window);
        glfwPollEvents();

        auto frameEndCpu = clock_type::now();

        // ---------------- Benchmark accumulation ----------------

        double frameMs = std::chrono::duration<double, std::milli>(
            frameEndCpu - frameStartCpu
        ).count();

        double renderMs = std::chrono::duration<double, std::milli>(
            frameEndCpu - renderStartCpu
        ).count();

        frameMsAccum += frameMs;
        simMsAccum += double(simMs);
        renderMsAccum += renderMs;
        benchmarkFrameCount++;

        if (benchmarkFrameCount >= benchmarkReportInterval) {
            double avgFrameMs = frameMsAccum / benchmarkFrameCount;
            double avgSimMs = simMsAccum / benchmarkFrameCount;
            double avgRenderMs = renderMsAccum / benchmarkFrameCount;

            double fps = 1000.0 / avgFrameMs;

            double interactions = double(N) * double(N);
            double interactionsPerSecond = interactions / (avgSimMs / 1000.0);
            double billionInteractionsPerSecond = interactionsPerSecond / 1.0e9;

            std::cout << std::fixed << std::setprecision(3);

            std::cout << "N: " << N
                      << " | FPS: " << fps
                      << " | frame: " << avgFrameMs << " ms"
                      << " | CUDA sim: " << avgSimMs << " ms"
                      << " | render: " << avgRenderMs << " ms"
                      << " | interaction(s): "
                      << billionInteractionsPerSecond
                      << " billion"
                      << std::endl;

            frameMsAccum = 0.0;
            simMsAccum = 0.0;
            renderMsAccum = 0.0;
            benchmarkFrameCount = 0;
        }
    }

    // ---------------- Cleanup ----------------

    CHECK(cudaGraphicsUnregisterResource(cudaPositionResource));

    CHECK(cudaFree(d_vel));
    CHECK(cudaFree(d_acc));

    CHECK(cudaEventDestroy(startEvent));
    CHECK(cudaEventDestroy(stopEvent));

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &positionVBO);
    glDeleteProgram(shaderProgram);

    glfwTerminate();

    return 0;
}