module grain.tensor;

import std.container.array : Array;

import grain.dlpack.header : DLManagedTensor;
import grain.rc : RC;
import grain.allocator : CPUAllocator;
import grain.buffer : AnyBuffer, Buffer;

/// dynamic tensor for type erasure and DLPack conversion
struct AnyTensor
{
    nothrow @nogc:

    import grain.dlpack.header : DLManagedTensor, DLContext, DLDataType;

    /// RC pointer to allocated buffer
    RC!AnyBuffer buffer;
    ulong offset;
    /// shape on dynamic memory
    RC!(Array!long) shape;
    /// strides on dynamic memory
    RC!(Array!long) strides;

    /// DLPack context
    DLContext context;
    /// DLPack element type
    DLDataType dataType;
}

/// typed tensor
struct Tensor(T, size_t dim, Allocator = CPUAllocator)
{
    import grain.dlpack.conv : dataTypeOf;

    ///
    Allocator allocator;
    /// buffer to store numeric values
    RC!(Buffer!Allocator) buffer;
    /// offset of the strided tensor on buffer
    ulong offset;
    /// shape of tensor
    long[dim] shape;
    /// strides of tensor
    long[dim] strides;

    /// erase type
    inout(AnyTensor) toAny() inout
    {
        alias A = RC!(Array!long);
        inout(AnyTensor) ret = {
            buffer: this.buffer.castTo!AnyBuffer,
            offset: this.offset,
            shape: cast(inout A) A.create(cast(long[]) this.shape[]),
            strides: cast(inout A) A.create(cast(long[]) this.strides[]),
            context: allocator.context,
            dataType: dataTypeOf!T
        };
        return ret;
    }

    /// revert type
    void fromAny(ref AnyTensor src)
    {
        // dynamic type check
        assert(allocator.context == src.context);
        assert(dataTypeOf!T == src.dataType);
        assert(dim == src.shape.length);
        this.buffer = typeof(this.buffer).create();
        foreach (i; 0 .. dim)
        {
            this.shape[i] = (*src.shape)[i];
            this.strides[i] = (*src.strides)[i];
        }
        if (src.buffer)
        {
            this.offset = src.offset;
            src.buffer._incRef();
            this.buffer.payload = src.buffer.payload;
            this.buffer._context = src.buffer._context;
        }
    }

    inout(T)* dataPtr() inout { return cast(T*) (this.buffer.ptr + offset); }

    alias toAny this;
}

@nogc nothrow @system
unittest
{
    Tensor!(float, 2) matrix;
    matrix.buffer = matrix.buffer.create(float.sizeof * 6);
    auto s = (cast(float*) matrix.buffer.ptr)[0 .. 6];
    s[0] = 123;
    s[5] = 0.1f;
    matrix.shape[0] = 2;
    matrix.shape[1] = 3;
    matrix.strides[0] = 3;
    matrix.strides[1] = 1;

    auto any = matrix.toAny;
    assert(matrix.buffer._counter == 2);
    {
        Tensor!(float, 2) a;
        a.buffer = a.buffer.create(float.sizeof * 6);
        a.fromAny(any);
        assert(matrix.buffer._counter == 3);
        assert(a.dataPtr[0] == 123);
        assert(a.dataPtr[5] == 0.1f);
    }
    matrix.fromAny(any);
    assert(matrix.buffer._counter == 2);
}
