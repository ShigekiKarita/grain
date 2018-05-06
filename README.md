# Demo: Writing CUDA Kernel in D

Thanks to DCompute and LDC1.8.0-, we can write CUDA kernel in D.
This project aims to be minimal for writing CUDA kernels and calling CUDA Driver APIs.

## how to run

```d
$ curl -fsS https://dlang.org/install.sh | bash -s ldc-1.9.0
$ source ~/dlang/ldc-1.9.0/activate
$ make test
```

I have tested with

- LDC1.8.0/1.9.0 (prebuilt binary)
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
