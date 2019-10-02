module grain.autograd;

import std.stdio;
import mir.rc.ptr : RCPtr;

import grain.dlpack : DLManagedTensor;
import grain.allocator : CPUAllocator;
import grain.buffer : Buffer;

/// dynamic tensor
struct AnyTensor
{
    DLManagedTensor data = void;
    alias data this;
}

/// typed tensor
struct Tensor(T, size_t dim, Allocator = CPUAllocator) {
    RCPtr!(Buffer!T) buffer;
    long[dim] shape;
    long[dim] strides;

    auto toDLPack()
    {
        import grain.dlpack;
        DLTensor t = {};
        t.data = this.buffer.buffer;
        DLManagedTensor m;
        m.dl_tensor = t;
        m.manager_ctx = cast(void*) &this.buffer;
        // TODO(karita): register deleter
        m.deleter = (DLManagedTensor* self) {};
        return AnyTensor(m);
    }
}

abstract class AnyOp {
    abstract AnyTensor[] forwardAny(AnyTensor[]);
}

abstract class Op(Impl) : AnyOp {
    override AnyTensor[] forwardAny(AnyTensor[] xs) {
        return xs;
    }
}

class ReLU : Op!ReLU {
}

unittest
{
    Tensor!(float, 2) matrix;
}
