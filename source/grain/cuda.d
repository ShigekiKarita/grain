module grain.cuda;
version(grain_cuda):

import std.traits : ReturnType, arity;
import std.stdio : writeln, writefln;
import std.string : toStringz, fromStringz;

import derelict.cuda;
import derelict.cudnn7;
import grain.cublas;
import grain.utility;




// TODO: support multiple GPU devices (context)
__gshared CUcontext context;
__gshared cublasHandle_t cublasHandle;
__gshared cudnnHandle_t cudnnHandle;

/// global cuda init
shared static this() {
    // Initialize the driver API
    DerelictCUDADriver.load();
    CUdevice device;
    cuInit(0);
    // Get a handle to the first compute device
    cuDeviceGet(&device, 0);
    // Create a compute device context
    cuCtxCreate(&context, 0, device);


    // init CUDA libraries
    checkCublasErrors(cublasCreate_v2(&cublasHandle));
    DerelictCuDNN7.load();
    checkCUDNN( cudnnCreate(&cudnnHandle) );
}

/// global cuda exit
shared static ~this() {
    import core.memory : GC;
    GC.collect();
    cublasDestroy_v2(cublasHandle);
    checkCUDNN( cudnnDestroy(cudnnHandle) );
    checkCudaErrors(cuCtxDestroy(context));
}

/// cuda module compiled from ptx string
struct CuModule {
    CUmodule cuModule;

    this(string ptxstr) {
        // JIT compile a null-terminated PTX string
        checkCudaErrors(cuModuleLoadData(&cuModule, cast(void*) ptxstr.toStringz));
    }

    ~this() {
        checkCudaErrors(cuModuleUnload(cuModule));
    }

    auto kernel(alias F)() {
        return Kernel!F(cuModule);
    }
}

/// global accessor for the cuda module in grain
class Global {
    import K = grain.kernel;
    private this() {}

    // Cache instantiation flag in thread-local bool
    // Thread local
    private static bool instantiated_ = false, cxxInstantiated_ = false;

    // Thread global
    private __gshared CuModule* module_, cxxModule_;

    static get()
    {
        if (!instantiated_)
        {
            synchronized(Global.classinfo)
            {
                module_ = new CuModule(K.cxxptx);
                instantiated_ = true;
            }
        }

        return module_;
    }

    static getCxx()
    {
        if (!cxxInstantiated_)
        {
            synchronized(Global.classinfo)
            {
                cxxModule_ = new CuModule(K.cxxptx);
                cxxInstantiated_ = true;
            }
        }
        return cxxModule_;
    }

    static cxxKernel(T...)(string name, T args)
    {
        CUfunction cuFunction;
        writeln("getFunction...");
        checkCudaErrors(cuModuleGetFunction(&cuFunction, getCxx(), name.toStringz));
        writeln("getFunction...");
        return Launcher!T(cuFunction, args);
    }


    static kernel(alias F)() {
        return get().kernel!F;
    }

}

/// ditto
auto global() {
    return Global.get();
}

// pthread error ?
// auto CUDA_POST_KERNEL_CHECK() {
//     checkCudaErrors(cudaPeekAtLastError());
// }

/// cuda kernel function launcher with runtime numbers of blocks/threads
struct Launcher(Args...) {
    CUfunction cuFunction;
    Args args;

    /// create kernel function as void[Args.length]
    auto kernelParams(T...)(T args) {
        void*[args.length] ret;
        foreach (i, a; args) {
            ret[i] = &a;
        }
        return ret;
    }

    /// detailed launch function
    void launch(uint[3] grid, uint[3] block, uint sharedMemBytes=0, CUstream stream=null) {
        checkCudaErrors(cuLaunchKernel(
                            cuFunction,
                            grid[0], grid[1], grid[2],
                            block[0], block[1], block[2],
                            sharedMemBytes, stream,
                            kernelParams(args).ptr, null));
        // CUDA_POST_KERNEL_CHECK();
    }

    // TODO __CUDA_ARCH__ < 200 512
    enum CUDA_NUM_THREADS = 1024;

    static getBlocks(uint n) {
        return (n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    }

    /// convinient launch function
    void launch(uint n=1, uint sharedMemBytes=0, CUstream stream=null) {
        checkCudaErrors(cuLaunchKernel(
                            cuFunction,
                            getBlocks(n), 1, 1,
                            CUDA_NUM_THREADS, 1, 1,
                            sharedMemBytes, stream,
                            kernelParams(args).ptr, null));
        // CUDA_POST_KERNEL_CHECK();
    }
}

/// cuda function object called by mangled name of C++/D device function F
struct Kernel(alias F) if (is(ReturnType!F == void)) {
    // enum name = __traits(identifier, F);
    enum name = F.mangleof;
    CUfunction cuFunction;

    this(CUmodule m) {
        // writeln("mangled: ", name);
        checkCudaErrors(cuModuleGetFunction(&cuFunction, m, name.toStringz));
    }

    // TODO: compile-time type check like d-nv
    // TODO: separate this to struct Launcher
    auto call(T...)(T args) {
        static assert(args.length == arity!F);
        // Kernel launch
        // checkCudaErrors(cuCtxSynchronize());
        return Launcher!T(cuFunction, args);
    }
}

/// fat pointer in CUDA
struct CuPtr(T) {
    CUdeviceptr ptr = 0;
    const size_t length = 0;

    /// create copy of host array into device
    this(T[] host) {
        this(host.length);
        checkCudaErrors(cuMemcpyHtoD(ptr, &host[0], T.sizeof * length));
    }

    @disable this(this); // not copyable

    /// create uninitialized T.sizeof * n array in device
    this(size_t n) {
        this.length = n;
        if (n > 0) {
            checkCudaErrors(cuMemAlloc(&this.ptr, T.sizeof * this.length));
        }
    }

    /// create fat pointer from raw pointer and its length
    this(CUdeviceptr p, size_t l) {
        this.ptr = p;
        this.length = l;
    }

    /// dtor calling cuMemFree
    ~this() {
        if (ptr != 0x0) checkCudaErrors(cuMemFree(ptr));
        ptr = 0x0;
    }

    /// true if length == 0
    @property
    bool empty() {
        return this.length == 0;
    }

    /// copy device memory to host (maybe reallocate in host)
    ref toHost(scope ref T[] host) {
        host.length = length;
        checkCudaErrors(cuMemcpyDtoH(host.ptr, this.ptr, T.sizeof * length));
        return host;
    }

    /// copy device memory to host (CAUTION: no reallocation here)
    auto toHost(T* host) {
        checkCudaErrors(cuMemcpyDtoH(host, this.ptr, T.sizeof * length));
        return host;
    }

    /// allocate host memory and copy device memory content
    auto toHost() {
        auto host = new T[this.length];
        checkCudaErrors(cuMemcpyDtoH(host.ptr, this.ptr, T.sizeof * this.length));
        return host;
    }

    /// duplicate cuda memory (deep copy)
    auto dup() {
        CUdeviceptr ret;
        if (this.length > 0) {
            checkCudaErrors(cuMemAlloc(&ret, T.sizeof * length));
            checkCudaErrors(cuMemcpyDtoD(ret, ptr, T.sizeof * length));
        }
        return typeof(this)(ret, length);
    }

    /// fill value for N elements from the first position
    ref fill_(T value, size_t N) {
        import std.conv : to;
        import std.traits : Parameters;
        mixin("alias _memset = cuMemsetD" ~  to!string(T.sizeof * 8) ~ ";");
        alias Bytes = Parameters!(_memset)[1];
        static assert(Bytes.sizeof == T.sizeof);
        _memset(this.ptr, *(cast(Bytes*) &value), N);
        return this;
    }

    /// fill value for all the element in device array
    ref fill_(T value) {
        return this.fill_(value, this.length);
    }

    /// fill zero for all the element in device array
    ref zero_() {
        return this.fill_(0);
    }
}

///
unittest {
    foreach (i; 0 .. 100) {
        auto d = CuPtr!float([3.0]);
        assert(d.toHost() == [3.0]);
    }
}


/// deep copy inter device memory without allocation
void copy(T)(ref CuPtr!T src, ref CuPtr!T dst)
    in { assert(src.length == dst.length); }
do {
    checkCudaErrors(cuMemcpyDtoD(dst.ptr, src.ptr, T.sizeof * src.length));
}

/// create zero filled N elements array
auto zeros(S: CuPtr!T, T)(size_t N) {
    import std.algorithm : move;
    return move(CuPtr!T(N).zero_());
}

/// cuda error checker
void checkCudaErrors(string file = __FILE__, size_t line = __LINE__,
                     string mod = __MODULE__, string func = __FUNCTION__)(CUresult err) {
    import std.format;
    const(char)* name, content;
    cuGetErrorName(err, &name);
    cuGetErrorString(err, &content);
    assert(err == CUDA_SUCCESS,
           format!"%s: %s from %s @%s:%s"(
               name.fromStringz,  content.fromStringz,
               func, file, line));
}

/// cublas error checker
void checkCublasErrors(cublasStatus_t err) {
    assert(err == CUBLAS_STATUS_SUCCESS, cublasGetErrorEnum(err));
}

/// cudnn error checker
void checkCUDNN(string file = __FILE__, size_t line = __LINE__)(cudnnStatus_t err) {
    import std.conv : to;
    import std.format : format;
    assert(err == CUDNN_STATUS_SUCCESS, cudnnGetErrorString(err).fromStringz ~ format!" at %s (%d)"(file, line));
}

/// example to launch kernel
unittest
{
    import grain.kernel; // : saxpy;

    // Populate input
    uint n = 16;
    auto hostA = new float[n];
    auto hostB = new float[n];
    auto hostC = new float[n];
    foreach (i; 0 .. n) {
        hostA[i] = i;
        hostB[i] = 2 * i;
        hostC[i] = 0;
    }

    // Device data
    auto devA = new CuPtr!float(hostA);
    auto devB = CuPtr!float(hostB);
    auto devC = CuPtr!float(n);

    // Kernel launch
    Global.kernel!(saxpy).call(devC.ptr, devA.ptr, devB.ptr, n).launch(n);

    // Validation
    devC.toHost(hostC);
    foreach (i; 0 .. n) {
        // writefln!"%f + %f = %f"(hostA[i], hostB[i], hostC[i]);
        assert(hostA[i] + hostB[i] == hostC[i]);
    }
}

/// test sum
unittest {
    import grain.kernel : sum;
    auto a = CuPtr!float([3, 4, 5]);
    auto b = CuPtr!float([0]);
    auto N = cast(int) a.length;
    assert(N == 3);
    Global.kernel!sum.call(a.ptr, b.ptr, N)
        .launch(cast(uint[3]) [1U,1,1], cast(uint[3]) [1U,1,1], 0U);
    checkCudaErrors(cuCtxSynchronize());
    assert(b.toHost()[0] == 3+4+5);
}

/*
// test cxx kernel
unittest {
    auto a = CuPtr!float([3, 4, 5]);
    auto b = CuPtr!float([0]);
    auto N = cast(int) a.length;
    assert(N == 3);
    Global.cxxKernel("sum_naive", a.ptr, b.ptr, N)
        .launch(cast(uint[3]) [1U,1,1], cast(uint[3]) [1U,1,1], 0U);
    // checkCudaErrors(cuCtxSynchronize());
    writeln(b.toHost());
    assert(b.toHost()[0] == 3+4+5);
}
*/

/// example to fill value
unittest {
    auto d = CuPtr!float(3);
    d.zero_();
    auto h = d.toHost();
    assert(h == [0, 0, 0]);
    assert(zeros!(CuPtr!float)(3).toHost() == [0, 0, 0]);
    assert(d.fill_(3).toHost() == [3, 3, 3]);
}


/// high-level axpy wrapper for CuPtr
void axpy(T)(const ref CuPtr!T x, ref CuPtr!T y, T alpha=1, int incx=1, int incy=1)  {
    static if (is(T == float)) {
        alias axpy_ = cublasSaxpy_v2;
    } else static if (is(T == double)) {
        alias axpy_ = cublasDaxpy_v2;
    } else {
        static assert(false, "unsupported type: " ~ T.stringof);
    }
    auto status = axpy_(cublasHandle, cast(int) x.length, &alpha,
                        cast(const T*) x.ptr, incx,
                        cast(T*) y.ptr, incy);
    assert(status == CUBLAS_STATUS_SUCCESS, cublasGetErrorEnum(status));
}

/// cublas tests
unittest {
    auto a = CuPtr!float([3, 4, 5]);
    auto b = CuPtr!float([1, 2, 3]);
    axpy(a, b, 2.0);
    assert(a.toHost() == [3, 4, 5]);
    assert(b.toHost() == [7, 10, 13]);
}

