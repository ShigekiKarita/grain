# grain

high-level CUDA wrapper for D

```d
import grain.cuda;
import grain.kernel : saxpy; // auto generated header of kernel/kernel.d

// Get a handle to the "myfunction" kernel function
// See Makefile how to create kernel/kernel.ptx
auto cuModule = loadModule("kernel/kernel.ptx");
scope(exit) checkCudaErrors(cuModuleUnload(cuModule));
auto ksaxpy = Kernel!saxpy(cuModule);

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

## Questions

- how to specify CUDA compute capability as `ldc2` compiler options? now I'm rewriting `generic` in ptx with `sm_30` by sed (see kernel/Makefile).
- should `.func` in generated ptx be `.entry`? I also apply sed (see kernel/Makefile)

If you know answers or much better ways, open issues for me.

```diff
diff raw.ptx processed.ptx 
6c6
< .target generic
---
> .target sm_30
12c12
< .visible .func saxpy(
---
> .visible .entry saxpy(
```

see entire ptx in `tmp` dir.

## Links

- https://github.com/ldc-developers/ldc
- https://llvm.org/docs/NVPTXUsage.html
- https://llvm.org/docs/CompileCudaWithLLVM.html
