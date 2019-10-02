module grain.autograd;

import std.stdio;
import mir.rc.ptr : RCPtr, castTo;

import grain.dlpack : DLManagedTensor, DLContext, DLDataType;
import grain.allocator : CPUAllocator;
import grain.buffer : AnyBuffer, Buffer;

/// dynamic tensor
struct AnyTensor
{
    import std.container.array : Array;

    /// RC pointer to allocated buffer
    RCPtr!AnyBuffer buffer;
    /// shape on dynamic memory
    Array!long shape;
    /// strides on dynamic memory
    Array!long strides;

    /// DLPack context
    DLContext dlContext;
    /// DLPack element type
    DLDataType dlType;

    /// assumes consumed exactly once (i.e., call deleter once)
    auto toDLPack()
    {
        import grain.dlpack;
        DLTensor t;
        t.data = this.buffer.payload.ptr;
        DLManagedTensor m;
        m.dl_tensor = t;
        m.manager_ctx = cast(void*) &this;
        // TODO(karita): register deleter
        m.deleter = (DLManagedTensor* self) {};
        return t;
    }
}

/// typed tensor
struct Tensor(T, size_t dim, Allocator = CPUAllocator)
{
    /// buffer to store numeric values
    RCPtr!(Buffer!Allocator) buffer;
    ///
    long[dim] shape;
    ///
    long[dim] strides;

    AnyTensor toAny()
    {
        AnyTensor ret = {
            buffer: buffer.castTo!AnyBuffer,
        };
        return ret;
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
