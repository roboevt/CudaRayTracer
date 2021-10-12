#include <iostream>
#include <fstream>
#include <math.h>
#include <limits>
#include "cuda_runtime.h"
#include <curand_kernel.h>
#include <curand.h>
#include "device_launch_parameters.h"

__constant__ float maxDistance = 3.40282346639e+38f;

using namespace std;

const int width = 1280;
const int height = 720;
int samples = 100;
int bounceLimit = 3;
const int tileHeight = 16;
const int tileWidth = 16;
const int numPixels = width * height;
const int stackSize = 65536;  // 64kb

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)


void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

struct vec3 {
    float x, y, z;
    __host__ __device__ vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    __host__ __device__ vec3() : x(0), y(0), z(0) {}
    __host__ __device__ vec3 operator+(const vec3& other) const { return vec3(this->x + other.x, this->y + other.y, this->z + other.z); }
    __host__ __device__ void operator+=(const vec3& other) { this->x += other.x; this->y += other.y; this->z += other.z; }
    __host__ __device__ vec3 operator-(const vec3& other) const { return vec3(this->x - other.x, this->y - other.y, this->z - other.z); }
    __host__ __device__ void operator-=(const vec3& other) { this->x -= other.x; this->y -= other.y; this->z -= other.z; }
    __host__ __device__ vec3 operator*(const float scale) const { return vec3(this->x * scale, this->y * scale, this->z * scale); };
    __host__ __device__ void operator*=(const float scale) { this->x *= scale; this->y *= scale; this->z *= scale; };
    __host__ __device__ float operator*(const vec3& other) const { return this->x * other.x + this->y * other.y + this->z * other.z; }
    __host__ __device__ float magnitudeSquared() const { return this->x * this->x + this->y * this->y + this->z * this->z;}
    __host__ __device__ float magnitude() const { return sqrt(this->magnitudeSquared()); }
    __host__ __device__ vec3 normalized() const { return (*this) * (1.0f / this->magnitude()); }
};

struct Color {
    float r, g, b;
    int samples;
    __host__ __device__ Color(float r, float g, float b) : r(r), g(g), b(b), samples(0) {}
    __host__ __device__ Color(float r, float g, float b, int samples) : r(r), g(g), b(b), samples(samples) {}
    __host__ __device__ Color operator+(const Color& other) const { return Color(this->r + other.r, this->g + other.g, this->b + other.b, this->samples + other.samples + 1); }
    __host__ __device__ void operator+=(const Color& other) { this->r += other.r; this->g += other.g; this->b += other.b; this->samples += other.samples + 1; }
    __host__ __device__ Color operator-(const Color& other) const { return Color(this->r - other.r, this->g - other.g, this->b - other.b, this->samples - other.samples + 1); }
    __host__ __device__ void operator-=(const Color& other) { this->r -= other.r; this->g -= other.g; this->b -= other.b; this->samples -= other.samples + 1; }
    __host__ __device__ Color operator*(const float scale) const { return Color(this->r * scale, this->g * scale, this->b * scale, this->samples); }
    __host__ __device__ Color output() { return Color(this->r / this->samples, this->g / this->samples, this->b / this->samples); }
};

__device__ vec3 randomInUnitSphere(curandState* state) {
    vec3 temp = vec3(curand_uniform(state), curand_uniform(state), curand_uniform(state));
    while (temp.magnitudeSquared() > 1) {
        float newX = curand_uniform(state) * 2.0f - 1.0f;
        float newY = curand_uniform(state) * 2.0f - 1.0f;
        float newZ = curand_uniform(state) * 2.0f - 1.0f;
        temp = vec3(newX, newY, newZ);
    }
    return temp.normalized();
}

class Ray {
public:
    vec3 origin, direction;
    __host__ __device__ Ray(vec3 origin, vec3 direction) : origin(origin), direction(direction) {}
    __host__ __device__ vec3 at(float distance) const { return this->origin + (this->direction.normalized() * distance); }
};

class Material {
public:
    Color color;
    __host__ __device__ Material() : color(Color(0, 0, 0)) {}
    __host__ __device__ Material(Color color) : color(color) {}
};

struct HitRecord {
    float distance;
    vec3 position, normal;
    Material hitMaterial;
   __device__ HitRecord() : distance(maxDistance), position(vec3(0, 0, 0)), normal(vec3(0, 0, 0)), hitMaterial(Material()) {}
};

class Sphere {
public:
    vec3 origin; 
    float radius;
    Material material;
    __host__ __device__ Sphere(vec3 origin, float radius) : origin(origin), radius(radius), material(Material()) {}
    __device__ void checkRay(const Ray& ray, HitRecord& record) {
        
        vec3 oc = ray.origin - this->origin;
        float a = ray.direction * ray.direction;
        float b = oc * ray.direction;
        float c = oc * oc - this->radius * this->radius;
        float discriminant = b * b - a * c;
        if (discriminant > 0) {
            float temp = (-b - sqrt(discriminant)) / a;
            if (temp > 0 && temp < record.distance) {
                record.distance = temp;
                record.position = ray.at(temp);
                record.normal = record.position - this->origin;
                record.hitMaterial = this->material;
            }
            temp = (-b + sqrt(discriminant)) / a;
            if (temp > 0 && temp < record.distance) {
                record.distance = temp;
                record.position = ray.at(temp);
                record.normal = record.position - this->origin;
                record.hitMaterial = this->material;
            }
        }
    }
};

__device__ void checkGroundPlane(const Ray& ray, HitRecord& record) {
    float t = -1.0f;
    if(ray.direction.y != 0)
        t = -ray.origin.y / ray.direction.normalized().y;
    if (t > 0 && t < record.distance) {
        record.distance = t;
        record.position = ray.at(t);
        record.normal = vec3(0, 1, 0);
        record.hitMaterial.color = Color(1, 1, 1);
    }
}

class Camera {
public:
    vec3 origin;
    float zoom;
    Camera(vec3 origin, float zoom) : origin(origin), zoom(zoom) {};
};

__device__ Color shade(const Ray& ray, Sphere& sphere, int bouncesRemaining, curandState* state) {
    if (bouncesRemaining <= 0) return Color(0, 0, 0);
    HitRecord record = HitRecord();

    sphere.checkRay(ray, record);
    checkGroundPlane(ray, record);

    if (record.distance < maxDistance) {
        Ray newRay = Ray(record.position, record.normal.normalized() + randomInUnitSphere(state));
        return record.hitMaterial.color + shade(newRay, sphere, --bouncesRemaining, state) * 0.5f;
    }

    return Color(.68f, .85f, .9f);  // Background color
}

__global__ void render(Color* buffer, curandState* states, Camera camera, int samples, int bounceLimit) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) return;  // Out of bounds of image, no point rendering
    int pixelIndex = y * width + x;

    Ray cameraRay = Ray(camera.origin, vec3((x-width/2)/static_cast<float>(width), (y-height/2)/ static_cast<float>(width), camera.zoom));

    Sphere sphere = Sphere(vec3(0, 1.0f, 5.0f), 1.0f);  // To be changed to parameter
    sphere.material.color = Color(1, 1, 1);

    for (int s = 0; s < samples; s++) {
        buffer[pixelIndex] += shade(cameraRay, sphere, bounceLimit, &states[pixelIndex]);  // Could add check for background and break
    }
}

__global__ void randomInit(int width, int height, curandState* state) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= width) || (y >= height)) return;
    int pixelIndex = y * width + x;
    curand_init(0, pixelIndex, 0, &state[pixelIndex]);
}

__global__ void processImage(Color* buffer, int samples) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) return;
    int pixelIndex = y * width + x;

    //Divide by number of samples to average colors across samples
    buffer[pixelIndex] = buffer[pixelIndex].output();

    //Clamp values between 0 and 1
    Color& pixelColor = buffer[pixelIndex];
    if (pixelColor.r < 0) pixelColor.r = 0;
    if (pixelColor.g < 0) pixelColor.g = 0;
    if (pixelColor.b < 0) pixelColor.b = 0;
    if (pixelColor.r > 1) pixelColor.r = 1;
    if (pixelColor.g > 1) pixelColor.g = 1;
    if (pixelColor.b > 1) pixelColor.b = 1;

}

__host__ void outputImage(Color* buffer) {
    ofstream imageOut("image.ppm");
    imageOut << "P3\n" << width << " " << height << "\n255\n";
    for (int i = height-1; i >= 0; i--) {
        for (int j = 0; j < width; j++) {
            int pixelIndex = i * width + j;
            int r = int(255.0f * (buffer[pixelIndex].r));
            int g = int(255.0f * (buffer[pixelIndex].g));
            int b = int(255.0f * (buffer[pixelIndex].b));
            imageOut << r << " " << g << " " << b << "\n";
        }
    }
}

int main(void)
{
    Color* imageBuffer;
    checkCudaErrors(cudaMallocManaged(&imageBuffer, numPixels * sizeof(Color)));

    curandState* curandStates;
    checkCudaErrors(cudaMallocManaged(&curandStates, numPixels * sizeof(curandState)));

    dim3 blocks(width / tileWidth + 1, height / tileHeight + 1);
    dim3 threads(tileWidth, tileHeight);

    size_t currentStackSize = 0;
    size_t * currentStackSizeP = &currentStackSize;
    cudaDeviceGetLimit(currentStackSizeP, cudaLimitStackSize);
    cout << "Default Stack Size: " << currentStackSize << endl;
    cudaDeviceSetLimit(cudaLimitStackSize, stackSize);  // Would be nice to calculate this, instead of guessing. Max value should be 512kb for compute 7.5
    cudaDeviceGetLimit(currentStackSizeP, cudaLimitStackSize);
    cout << "Current Stack Size: " << currentStackSize << endl;

    randomInit <<<blocks, threads >>> (width, height, curandStates);

    Camera camera = Camera(vec3(0, .75f, 0), 1.0f);

    cout << "Beginning render\n";

    render <<<blocks, threads >>> (imageBuffer, curandStates, camera, samples, bounceLimit);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    cout << "Done rendering";

    processImage <<<blocks, threads >>> (imageBuffer, samples);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    outputImage(imageBuffer);

    cudaFree(imageBuffer);

    return 0;
}