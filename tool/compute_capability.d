import std.stdio : writeln, writefln;
import std.format : format;
import std.conv : to;
import std.file : readText;
import std.string : toStringz, fromStringz;

import derelict.cuda;


void checkCudaErrors(CUresult err) {
    const(char)* name, content;
    cuGetErrorName(err, &name);
    cuGetErrorString(err, &content);

    assert(err == CUDA_SUCCESS, name.fromStringz ~ ": " ~ content.fromStringz);
}


void main()
{
    DerelictCUDADriver.load();
    CUdevice device;
    CUcontext context;

    // Initialize the driver API
    cuInit(0);
    // Get a handle to the first compute device
    cuDeviceGet(&device, 0);
    // Create a compute device context
    cuCtxCreate(&context, 0, device);

    int devMajor, devMinor;
    checkCudaErrors(cuDeviceGetAttribute(
                        &devMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    checkCudaErrors(cuDeviceGetAttribute(
                        &devMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    writefln!"%d%d"(devMajor, devMinor);
}
