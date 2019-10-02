module grain.tensor;

import std.container.array : Array;

import grain.rc : RC;
import grain.dlpack : DLManagedTensor, DLContext, DLDataType;
import grain.allocator : CPUAllocator;
import grain.buffer : AnyBuffer, Buffer;

/// dynamic tensor
struct AnyTensor
{
    /// RC pointer to allocated buffer
    RC!AnyBuffer buffer;
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
        this.buffer._incRef();
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
    RC!(Buffer!Allocator) buffer;
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

@nogc nothrow @system
unittest
{
    Tensor!(float, 2) matrix;
}
