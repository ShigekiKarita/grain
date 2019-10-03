module grain.autograd;

import std.stdio;

import grain.tensor : Tensor, AnyTensor;

abstract class AnyOp {
    abstract AnyTensor[] forwardAny(AnyTensor[]);
}

abstract class Op(Impl) : AnyOp {
    override AnyTensor[] forwardAny(AnyTensor[] xs) {
        return xs;
    }
}

class ReLU(T, size_t dim, Allocator) : Op!ReLU {
    // Tensor!(T, dim, CPUAllocator) forward(Tensor!(T, dim, CPUAllocator) x)
    // {
    //     return
    // }
}

// ///
// unittest
// {

// }
