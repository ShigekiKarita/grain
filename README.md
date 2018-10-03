<p align="center">
  <img height="160" src="https://raw.githubusercontent.com/ShigekiKarita/grain/master/grain.png">
</p>

# grain
[![Build Status](https://travis-ci.org/ShigekiKarita/grain.svg?branch=master)](https://travis-ci.org/ShigekiKarita/grain)
[![codecov](https://codecov.io/gh/ShigekiKarita/grain/branch/master/graph/badge.svg)](https://codecov.io/gh/ShigekiKarita/grain)
[![Dub version](https://img.shields.io/dub/v/grain.svg)](https://code.dlang.org/packages/grain)

autograd [mir](https://github.com/libmir/mir-algorithm) and CUDA library for dynamic neural networks in D.

- documentaion https://shigekikarita.github.io/grain
- introduction (PDF) https://github.com/ShigekiKarita/grain-talk/blob/master/slide.pdf

## features

- dynamic computation graph like chainer or pytorch
- statically typed tensor `Variable(T, size_t dim, alias Storage)` unlike numpy
- CPU (mir) and CUDA (cublas/cudnn) backend
- extensible (i.e., user-defined) autograd function
- LDC2 (CPU/CUDA) and DMD (CPU only) support

## how to run MNIST


```d
$ dub --config=example-mnist -b=cuda-release # with cuda
$ dub --config=example-mnist -b=release      # without cuda
```

it results as following (may take several seconds without cuda)

```
Running ./grain-example-mnist
loading data/train-images-idx3-ubyte.gz
loading data/train-labels-idx1-ubyte.gz
loading data/t10k-images-idx3-ubyte.gz
loading data/t10k-labels-idx1-ubyte.gz
train loss: 0.538635, acc: 0.864311
test loss: 0.299959, acc: 0.915264
train loss: 0.277901, acc: 0.920858
test loss: 0.241783, acc: 0.930589
train loss: 0.229879, acc: 0.934999
test loss: 0.206087, acc: 0.939704
train loss: 0.198716, acc: 0.943937
test loss: 0.181938, acc: 0.945613
train loss: 0.175066, acc: 0.950957
test loss: 0.163919, acc: 0.951022
```


## how to test

```d
$ curl -fsS https://dlang.org/install.sh | bash -s ldc-1.9.0
$ source ~/dlang/ldc-1.9.0/activate
$ dub test -b=cuda-unittest # with cuda
$ dub test                  # without cuda
```

I have tested with

- LDC1.9.0 (prebuilt binary)
- CUDA9.1 (optional)
- CUDNN7 (optional but required if you use CUDA)
- NVidia GTX760, GTX1080 (https://grain.dpldocs.info/grain.htmloptional)

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

- practical examples (MNIST, CIFAR10, WordLM). see example/
  - `dub --config=example-mnist`
  - (wip) `dub --config=example-char-rnn`
- more autograd functions. see source/grain/functions/ TODO
- multi GPU
- curand wrappers
- statically linked kernel module instead of ptx string
- dmd support
- double backward (implement Function.backward with Chain)
