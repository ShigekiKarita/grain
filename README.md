# grain

high-level CUDA wrapper for D

```d
unittest
{
    import grain.cuda;
    import grain.kernel : saxpy;

    // Get a handle to the kernel function in kernel/kernel.d
    // See Makefile how to create kernel/kernel.ptx
    auto cuModule = CuModule("kernel/kernel.ptx");
    auto ksaxpy = cuModule.kernel!saxpy;

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
    auto devA = CuPtr!float(hostA);
    auto devB = CuPtr!float(hostB);
    auto devC = CuPtr!float(n);

    // Kernel launch
    ksaxpy.launch(devC.ptr, devA.ptr, devB.ptr, n, [1,1,1], [n,1,1]);

    // Validation
    devC.toCPU(hostC);
    foreach (i; 0 .. n) {
        writefln!"%f + %f = %f"(hostA[i], hostB[i], hostC[i]);
        assert(hostA[i] + hostB[i] == hostC[i]);
    }
}
```

## how to run

```d
$ curl -fsS https://dlang.org/install.sh | bash -s ldc-1.9.0
$ source ~/dlang/ldc-1.9.0/activate
$ make test
```

I have tested with

- LDC1.9.0 (prebuilt binary)
- CUDA9.1
- NVidia GTX760

## links

- https://github.com/ldc-developers/ldc
- https://llvm.org/docs/NVPTXUsage.html
- https://llvm.org/docs/CompileCudaWithLLVM.html


## todo

- axpy in grad register
- kernel type check
