module grain.cuda;
version(grain_cuda):

import std.traits : ReturnType, arity;
import std.stdio : writeln, writefln;
import std.string : toStringz, fromStringz;

import derelict.cuda;
import grain.cublas;

// TODO: support multiple GPU devices (context)
__gshared CUcontext context;
__gshared cublasHandle_t cublasHandle;

shared static this() {
    DerelictCUDADriver.load();
    CUdevice device;

    // Initialize the driver API
    cuInit(0);
    // Get a handle to the first compute device
    cuDeviceGet(&device, 0);
    // Create a compute device context
    cuCtxCreate(&context, 0, device);
    checkCublasErrors(cublasCreate_v2(&cublasHandle));
}

shared static ~this() {
    import core.memory : GC;
    GC.collect();
    cublasDestroy_v2(cublasHandle);
    checkCudaErrors(cuCtxDestroy(context));
}


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


class Global {
    import K = grain.kernel;
    private this() {}

    // Cache instantiation flag in thread-local bool
    // Thread local
    private static bool instantiated_, cxxInstantiated_;

    // Thread global
    private __gshared CuModule* module_, cxxModule_;

    static get()
    {
        if (!instantiated_)
        {
            synchronized(Global.classinfo)
            {
                module_ = new CuModule(K.ptx);
                instantiated_ = true;
            }
        }

        return module_;
    }

    static cxxKernel(T...)(string name, T args)
    {
        if (!cxxInstantiated_)
        {
            synchronized(Global.classinfo)
            {
                cxxModule_ = new CuModule(K.cxxptx);
                cxxInstantiated_ = true;
            }
        }
        CUfunction cuFunction;
        writeln("getFunction...");
        checkCudaErrors(cuModuleGetFunction(&cuFunction, cxxModule_, name.toStringz));
        writeln("getFunction...");
        return Launcher!T(cuFunction, args);
    }


    static kernel(alias F)() {
        return get().kernel!F;
    }

}

auto global() {
    return Global.get();
}

struct Launcher(Args...) {
    CUfunction cuFunction;
    Args args;

    auto kernelParams(T...)(T args) {
        void*[args.length] ret;
        foreach (i, a; args) {
            ret[i] = &a;
        }
        return ret;
    }

    void launch(uint[3] grid, uint[3] block, uint sharedMemBytes=0, CUstream stream=null) {
        checkCudaErrors(cuLaunchKernel(
                            cuFunction,
                            grid[0], grid[1], grid[2],
                            block[0], block[1], block[2],
                            sharedMemBytes, stream,
                            kernelParams(args).ptr, null));
    }
}

struct Kernel(alias F) if (is(ReturnType!F == void)) {
    enum name = __traits(identifier, F);
    CUfunction cuFunction;

    this(CUmodule m) {
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


struct CuPtr(T) {
    CUdeviceptr ptr;
    const size_t length;

    this(T[] host) {
        this(host.length);
        checkCudaErrors(cuMemcpyHtoD(ptr, &host[0], T.sizeof * length));
    }

    @disable this(this); // not copyable

    this(size_t n) {
        length = n;
        checkCudaErrors(cuMemAlloc(&ptr, T.sizeof * n));
    }

    this(CUdeviceptr p, size_t l) {
        this.ptr = p;
        this.length = l;
    }

    ~this() {
        if (ptr != 0x0) checkCudaErrors(cuMemFree(ptr));
        ptr = 0x0;
    }

    ref toHost(scope ref T[] host) {
        host.length = length;
        checkCudaErrors(cuMemcpyDtoH(&host[0], ptr, T.sizeof * length));
        return host;
    }

    auto toHost() {
        auto host = new T[length];
        checkCudaErrors(cuMemcpyDtoH(&host[0], ptr, T.sizeof * length));
        return host;
    }

    auto dup() {
        CUdeviceptr ret;
        checkCudaErrors(cuMemAlloc(&ret, T.sizeof * length));
        checkCudaErrors(cuMemcpyDtoD(ret, ptr, T.sizeof * length));
        return typeof(this)(ret, length);
    }

    ref fill_(T value, size_t N) {
        import std.conv : to;
        import std.traits : Parameters;
        mixin("alias _memset = cuMemsetD" ~  to!string(T.sizeof * 8) ~ ";");
        alias Bytes = Parameters!(_memset)[1];
        static assert(Bytes.sizeof == T.sizeof);
        _memset(this.ptr, *(cast(Bytes*) &value), N);
        return this;
    }

    ref fill_(T value) {
        return this.fill_(value, this.length);
    }

    ref zero_() {
        return this.fill_(0);
    }
}

void copy(T)(ref CuPtr!T src, ref CuPtr!T dst)
    in { assert(src.length == dst.length); }
do {
    checkCudaErrors(cuMemcpyDtoD(dst.ptr, src.ptr, T.sizeof * src.length));
}

auto zeros(S: CuPtr!T, T)(size_t N) {
    import std.algorithm : move;
    return move(CuPtr!T(N).zero_());
}


void checkCudaErrors(CUresult err) {
    const(char)* name, content;
    cuGetErrorName(err, &name);
    cuGetErrorString(err, &content);
    assert(err == CUDA_SUCCESS, name.fromStringz ~ ": " ~ content.fromStringz);
}


void checkCublasErrors(cublasStatus_t err) {
    assert(err == CUBLAS_STATUS_SUCCESS, cublasGetErrorEnum(err));
}

unittest
{
    import grain.kernel : saxpy;

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
    Global.kernel!saxpy.call(devC.ptr, devA.ptr, devB.ptr, n).launch([1,1,1], [n,1,1]);

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
    writeln(b.toHost());
    assert(b.toHost()[0] == 3+4+5);
}

// test cxx kernel
unittest {
    auto a = CuPtr!float([3, 4, 5]);
    auto b = CuPtr!float([0]);
    auto N = cast(int) a.length;
    assert(N == 3);
    Global.cxxKernel("sum_naive", a.ptr, b.ptr, N)
        .launch(cast(uint[3]) [1U,1,1], cast(uint[3]) [1U,1,1], 0U);
    checkCudaErrors(cuCtxSynchronize());
    writeln(b.toHost());
    assert(b.toHost()[0] == 3+4+5);
}

unittest {
    auto d = CuPtr!float(3);
    d.zero_();
    auto h = d.toHost();
    assert(h == [0, 0, 0]);
    assert(zeros!(CuPtr!float)(3).toHost() == [0, 0, 0]);
    assert(d.fill_(3).toHost() == [3, 3, 3]);
}


void axpy(T)(const ref CuPtr!T x, ref CuPtr!T y, T alpha=1.0, int incx=1, int incy=1)  {
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
