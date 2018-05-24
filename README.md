# grain (WIP)

Grain automatically differentiates native [mir](https://github.com/libmir/mir-algorithm)-like and high-level CUDA functions in D. Inspired by PyTorch.

## how to test

```d
$ curl -fsS https://dlang.org/install.sh | bash -s ldc-1.9.0
$ source ~/dlang/ldc-1.9.0/activate
$ make test # with cuda
$ make test NO_CUDA=true # without cuda
```

I have tested with

- LDC1.9.0 (prebuilt binary)
- CUDA9.1 (optional)
- NVidia GTX760, GTX1080 (optional)

## links

- https://github.com/ldc-developers/ldc
- https://llvm.org/docs/NVPTXUsage.html
- https://llvm.org/docs/CompileCudaWithLLVM.html


## todo

- kernel type check
- more cuda intrinsics like exp/log
- more functions
  - matmul
  - softmax cross entropy
  - optimizer (SGD, Adadelta, Adam)
  - basic ops
    - add
    - mul
    - concat
    - view
    - transpose
- practical examples (MNIST, CIFAR10, WordLM)

