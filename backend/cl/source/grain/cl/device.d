module grain.cl.device;

import grain.dpp.cl;
// import grain.dpp.cl_enum : CL_DEVICE_TYPE_DEFAULT, CL_MEM_READ_WRITE;

import grain.cl.testing;

@nogc nothrow
void printDeviceInfo(cl_device_id device)
{
    import core.stdc.stdio : printf;
    import std.typecons : tuple;

    // print string info
    cl_ulong len;
    char* p;
    static foreach (s; tuple(
        "CL_DEVICE_NAME",
        "CL_DEVICE_VERSION",
        "CL_DRIVER_VERSION",
        "CL_DEVICE_OPENCL_C_VERSION"))
    {
        {
            mixin("enum d = " ~ s ~ ";");
            // print device name
            clGetDeviceInfo(device, d, 0, null, &len);
            p = cast(char*) malloc(len);
            clGetDeviceInfo(device, d, len, p, null);
            printf("%-40s: %s\n", s.ptr, p);
            free(p);
        }
    }

    // print size info
    static foreach (s; tuple(
        "CL_DEVICE_MAX_CLOCK_FREQUENCY",
        "CL_DEVICE_MAX_COMPUTE_UNITS",
        "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS",
        "CL_DEVICE_GLOBAL_MEM_SIZE",
        "CL_DEVICE_LOCAL_MEM_SIZE",
        "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE",
        "CL_DEVICE_MAX_MEM_ALLOC_SIZE"))
    {
        {
            mixin("enum d = " ~ s ~ ";");
            clGetDeviceInfo(device, d, len.sizeof, &len, null);
            printf("%-40s: %ld\n", s.ptr, len);
        }
    }
}

struct ClDevice
{
    @nogc nothrow:

    static cl_platform_id[] platformIds;
    static typeof(this)[] platforms;

    cl_platform_id platformId;
    cl_device_id[] deviceIds;

    static this()
    {
        import core.memory : pureMalloc;
        // init device
        cl_uint numPlatform;
        checkCl!clGetPlatformIDs(0, null, &numPlatform);
        auto platformIdsPtr = cast(cl_platform_id*)
            pureMalloc(cl_platform_id.sizeof * numPlatform);
        platformIds = platformIdsPtr[0 .. numPlatform];
        checkCl!clGetPlatformIDs(numPlatform, platformIdsPtr, null);

        auto platformsPtr = cast(typeof(this)*)
            pureMalloc(typeof(this).sizeof * numPlatform);
        platforms = platformsPtr[0 .. numPlatform];
        platforms[] = typeof(this).init;
        foreach (pid, ref p; platforms)
        {
            p.platformId = platformIds[pid];
        }

        foreach (pid, ref p; platforms)
        {
            cl_device_id device_id = null;
            cl_uint numDevice;
            checkCl!clGetDeviceIDs(p.platformId, CL_DEVICE_TYPE_DEFAULT,
                                   0, null, &numDevice);
            auto deviceIdsPtr = cast(cl_device_id*)
                pureMalloc(cl_device_id.sizeof * numDevice);
            checkCl!clGetDeviceIDs(p.platformId, CL_DEVICE_TYPE_DEFAULT,
                                   numDevice, deviceIdsPtr, null);
            p.deviceIds = deviceIdsPtr[0 .. numDevice];
        }
        printInfo();
    }

    static cl_device_id get(cl_uint device, cl_uint platform=0)
    {
        return platforms[platform].deviceIds[device];
    }

    static void printInfo()
    {
        import core.stdc.stdio : printf;
        foreach (pid, p; platforms)
        {
            foreach (did, d; p.deviceIds)
            {
                debug printf("platform id: %d, device id: %d\n", pid, did);
                printDeviceInfo(d);
            }
        }
    }

    static ~this()
    {
        import core.memory : pureFree;
        foreach (p; platforms)
        {
            pureFree(p.deviceIds.ptr);
        }
        pureFree(platformIds.ptr);
        pureFree(platforms.ptr);
    }
}
