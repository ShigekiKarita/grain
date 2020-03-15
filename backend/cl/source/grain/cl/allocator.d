module grain.cl.allocator;

// import grain.tensor : Opt;
import grain.cl.testing;
import grain.cl.device : ClDevice;

import grain.dpp.cl;


struct ClMem
{
    @nogc nothrow:

    cl_mem ptr;
    size_t length;

    alias ptr this;
}

struct ClMallocator
{
    // Opt opt;
    // alias opt this;

    /// device indicator
    enum deviceof = "cl";
    enum pinned = false;

    ///
    @trusted @nogc nothrow
    ClMem allocate()(size_t bytes)
    {
        // import grain.dpp.cuda_driver : cuMemAlloc_v2, CUdeviceptr;
        if (!bytes) return ClMem(null, 0);

        auto id = ClDevice.get(0, 0); // TODO(karita): opt.deviceId, opt.platformId);
        auto context = checkClFun!clCreateContext(null, 1, &id, null, null);
        auto da = checkClFun!clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, null);
        return ClMem(da, bytes);
    }

    ///
    @system @nogc nothrow
    bool deallocate()(ClMem b)
    {
        checkCl(clReleaseMemObject(b.ptr));
        return true;
    }

    enum instance = ClMallocator();
}
