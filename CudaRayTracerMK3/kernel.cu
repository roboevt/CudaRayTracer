#include <iostream>
#include <fstream>
#include <math.h>
#include <limits>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__constant__ float maxDistance = 3.40282346639e+38f;

using namespace std;

const int width = 1280;
const int height = 720;
int samples = 100000;
int bounceLimit = 5;
const int tileHeight = 16;
const int tileWidth = 16;
const int numPixels = width * height;

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)


void check_cuda(cudaError_t result,
    char const* const func,
    const char* const file,
    int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

struct vec3 {
    float x, y, z;
    __host__ __device__ vec3(float x, float y, float z) : x(x), y(y), z(z) {};
    __host__ __device__ vec3 operator+(const vec3& other) const { return vec3(this->x + other.x, this->y + other.y, this->z + other.z); };
    __host__ __device__ vec3 operator-(const vec3& other) const { return vec3(this->x - other.x, this->y - other.y, this->z - other.z); };
    __host__ __device__ vec3 operator*(const float scale) const { return vec3(this->x * scale, this->y * scale, this->z * scale); };
    __host__ __device__ float operator*(const vec3& other) const { return this->x * other.x + this->y * other.y + this->z * other.z; };
    __host__ __device__ float magnitudeSquared() const { return *this * *this; };
    __host__ __device__ float magnitude() const { return sqrt(this->magnitudeSquared()); };
    __host__ __device__ vec3 normalized() const { return vec3(*this * (1.0f / this->magnitude())); };
};

class Ray {
public:
    vec3 origin, direction;
    __host__ __device__ Ray(vec3 origin, vec3 direction) : origin(origin), direction(direction) {};
    __host__ __device__ vec3 at(float distance) const { return this->origin + (this->direction.normalized() * distance); };
};

class Material {
public:
    vec3 color;
    __host__ __device__ Material() : color(vec3(0, 0, 0)) {};
    __host__ __device__ Material(vec3 color) : color(color) {};
};

struct HitRecord {
    float distance;
    vec3 position;
    vec3 normal;
    Material hitMaterial;
   __device__ HitRecord() : distance(maxDistance), position(vec3(0, 0, 0)), normal(vec3(0, 0, 0)), hitMaterial(Material()) {};
};

class Sphere {
public:
    vec3 origin; float radius;
    Material material;
    __host__ __device__ Sphere(vec3 origin, float radius) : origin(origin), radius(radius) {};
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
    float t = -ray.origin.y / ray.direction.y;
    if (t > 0 && t < record.distance) {
        record.distance = true;
        record.position = ray.at(t);
        record.normal = vec3(0, 1, 0);
        record.hitMaterial.color = vec3(0, 1, 0);
    }
}

class Camera {
public:
    vec3 origin;
    float zoom;
    Camera(vec3 origin, float zoom) : origin(origin), zoom(zoom) {};
};

__device__ vec3 shade(const Ray& ray, int bouncesRemaining) {
    if (bouncesRemaining <= 0) return vec3(0, 0, 0);
    Sphere sphere = Sphere(vec3(0, 0, 5), 1.0f);  // To be changed to parameter
    sphere.material.color = vec3(1, 0, 1);
    HitRecord record = HitRecord();

    sphere.checkRay(ray, record);
    checkGroundPlane(ray, record);

    if (record.distance < maxDistance) {
        Ray newRay = Ray(record.position, record.normal);  // Must be changed to accurate direction - random bouncing
        return record.hitMaterial.color + shade(newRay, bouncesRemaining--) * 0.8f;
    }

    return vec3(0, 0, 0);
}

__global__ void render(vec3* buffer, Camera camera, int samples, int bounceLimit) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) return;  // Out of bounds of image, no point rendering
    int pixelIndex = y * width + x;

    Ray cameraRay = Ray(camera.origin, vec3((x-width/2)/static_cast<float>(width), (y-height/2)/ static_cast<float>(width), camera.zoom));

    buffer[pixelIndex] = shade(cameraRay, bounceLimit);
}

__global__ void process(vec3* buffer) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) return;  // Out of bounds of image, no point rendering
    int pixelIndex = y * width + x;

    vec3& pixelColor = buffer[pixelIndex];
    if (pixelColor.x < 0) pixelColor.x = 0;
    if (pixelColor.y < 0) pixelColor.y = 0;
    if (pixelColor.z < 0) pixelColor.z = 0;
    if (pixelColor.x > 1) pixelColor.x = 1;
    if (pixelColor.y > 1) pixelColor.y = 1;
    if (pixelColor.z > 1) pixelColor.z = 1;

}

__host__ void outputImage(vec3* buffer) {
    ofstream imageOut("image.ppm");
    imageOut << "P3\n" << width << " " << height << "\n255\n";
    for (int i = height-1; i >= 0; i--) {
        for (int j = 0; j < width; j++) {
            int pixelIndex = i * width + j;
            int r = int(255.0f * (buffer[pixelIndex].x));
            int g = int(255.0f * (buffer[pixelIndex].y));
            int b = int(255.0f * (buffer[pixelIndex].z));
            imageOut << r << " " << g << " " << b << "\n";
        }
    }
}

// Kernel function to add the elements of two arrays
__global__
void add(int n, vec3* x, vec3* y)
{
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

int main(void)
{
    vec3* imageBuffer;
    checkCudaErrors(cudaMallocManaged(&imageBuffer, numPixels * sizeof(vec3)));

    dim3 blocks(width / tileWidth + 1, height / tileHeight + 1);
    dim3 threads(tileWidth, tileHeight);

    Camera camera = Camera(vec3(0, .5f, 0), 1.0f);

    cout << "Beginning render\n";

    render << <blocks, threads >> > (imageBuffer, camera, samples, bounceLimit);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    cout << "Done rendering";

    process <<<blocks, threads >>> (imageBuffer);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    outputImage(imageBuffer);

    cudaFree(imageBuffer);

    return 0;
}