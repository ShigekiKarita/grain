# grain (WIP)

differentiates native [mir](https://github.com/libmir/mir-algorithm) and CUDA functions in D.

## features

- dynamic computation graph like chainer or pytorch
- statically typed tensor `Variable(T, size_t dim, alias Storage)` unlike numpy
- CPU (mir) and CUDA (cublas/cudnn) backend
- extensible (i.e., user-defined) autograd function

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
- CUDNN7 (optional but required if you use CUDA)
- NVidia GTX760, GTX1080 (optional)

## links

- CUDA in D
  - https://github.com/ldc-developers/ldc
  - https://llvm.org/docs/NVPTXUsage.html
  - https://llvm.org/docs/CompileCudaWithLLVM.html

- Referenced autograd libraries
  - Pytorch https://github.com/pytorch/pytorch
  - Chainer v1 https://github.com/chainer/chainer/tree/v1


## todo

sorted by higher priority for me

- kernel type check
- test CPU/CUDA function equality
- practical examples (MNIST, CIFAR10, WordLM)
- more autograd functions
  - matmul (done)
  - logsoftmax (done)
  - nlloss
  - softmax cross entropy
  - convolution/cross correlation
  - optimizer (SGD, Adadelta, Adam)
  - basic ops
    - add/sub/mul/div
    - concat
    - view
    - transpose
  - cudnn compatible activations (done in grain.cudnn)
    - relu (done by mir, custom kernel and cudnn)
    - sigmoid
    - tanh
    - clipped relu
- more cuda intrinsics like exp/log
- multi GPU

