#include <stdio.h>
#include <string>
#include <cuda.h>

void checkCudaErrors(CUresult err) {
    if (err != CUDA_SUCCESS) {
        const char* name;
        const char* content;
        cuGetErrorName(err, &name);
        cuGetErrorString(err, &content);
        fprintf(stderr, "%s: %s\n", name, content);
        exit(err);
    }
}

int main(int argc, char** argv) {
    int device_id = 0;
    if (argc == 2) {
        device_id = std::stoi(argv[1]);
    }
    fprintf(stderr, "device_id: %d\n", device_id);

    CUdevice device;
    CUcontext context;
    // Initialize the driver API
    cuInit(0);
    // Get a handle to the first compute device
    cuDeviceGet(&device, device_id);
    // Create a compute device context
    cuCtxCreate(&context, 0, device);

    int devMajor, devMinor;
    checkCudaErrors(cuDeviceGetAttribute(
                        &devMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    checkCudaErrors(cuDeviceGetAttribute(
                        &devMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    printf("%d%d\n", devMajor, devMinor);
}