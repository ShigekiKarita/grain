module grain.tensor;

import std.container.array : Array;

import grain.dlpack.header : DLManagedTensor;
import grain.rc : RC;
import grain.allocator : CPUAllocator, isCPU;
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
    import grain.dlpack.header : DLContext;
    import grain.dlpack.conv : dataTypeOf;

    /// buffer to store numeric values
    RC!(Buffer!Allocator) buffer;
    /// offset of the strided tensor on buffer
    ulong offset;
    /// shape of tensor
    long[dim] shape;
    /// strides of tensor
    long[dim] strides;

    /// device context (device id, device type, data type, etc)
    DLContext context() const { return this.buffer.allocator.context; }

    /// erase type
    inout(AnyTensor) toAny() inout
    {
        alias A = RC!(Array!long);
        inout(AnyTensor) ret = {
            buffer: this.buffer.castTo!AnyBuffer,
            offset: this.offset,
            shape: cast(inout A) A.create(cast(long[]) this.shape[]),
            strides: cast(inout A) A.create(cast(long[]) this.strides[]),
            context: this.context,
            dataType: dataTypeOf!T
        };
        return ret;
    }

    alias toAny this;

    /// revert type
    void fromAny(ref AnyTensor src)
    {
        // dynamic type check
        assert(this.context == src.context);
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
            // TODO(karita): provide a way to share members in RC
            src.buffer._incRef();
            this.buffer.payload = src.buffer.payload;
            this.buffer._context = src.buffer._context;
        }
    }

    inout(T)* dataPtr() inout { return cast(T*) (this.buffer.ptr + offset); }

    import mir.ndslice.slice : mir_slice, Universal;

    static if (isCPU!Allocator)
    {
        /// tensor as slice
        mir_slice!(T*, dim, Universal) asSlice()
        {
            size_t[dim] ushape;
            foreach (i; 0 .. dim)
            {
                ushape[i] = this.shape[i];
            }
            return typeof(return)(ushape, this.strides, this.dataPtr);
        }

        /// tensor as slice
        mir_slice!(const(T)*, dim, Universal) asSlice() const
        {
            size_t[dim] ushape;
            foreach (i; 0 .. dim)
            {
                ushape[i] = this.shape[i];
            }
            return typeof(return)(ushape, this.strides, this.dataPtr);
        }
    }

    /// allocate uninitialized tensor
    static empty(long[dim] shape...)
    {
        typeof(this) ret;
        size_t n = 1;
        foreach (i, s; shape)
        {
            assert(s > 0);
            n *= s;
            ret.shape[i] = s;
            ret.strides[i] = i == dim - 1 ? 1 : shape[i + 1];
        }
        ret.buffer = ret.buffer.create(T.sizeof * n);
        return ret;
    }
}

///
@nogc nothrow @system
unittest
{
    auto matrix = Tensor!(float, 2).empty(2, 3);
    assert(matrix.shape[0] == 2);
    assert(matrix.shape[1] == 3);
    assert(matrix.strides[0] == 3);
    assert(matrix.strides[1] == 1);

    auto s = matrix.dataPtr[0 .. 6];
    s[0] = 123;
    s[1] = 122;
    s[3] = 121;
    s[5] = 0.1f;

    // Tensor <-> AnyTensor
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

    // CPU slice
    const cs = matrix.asSlice;
    auto ms = matrix.asSlice;
    assert(ms[0, 0] == 123);
    assert(ms[0, 1] == 122);
    assert(ms[1, 0] == 121);
    assert(ms[1, 2] == 0.1f);
}
