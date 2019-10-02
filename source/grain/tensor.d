module grain.tensor;

import std.container.array : Array;

import grain.dlpack : DLManagedTensor;
import grain.rc : RC;
import grain.allocator : CPUAllocator;
import grain.buffer : AnyBuffer, Buffer;


/// DLPack resource export manager(_ctx)
private struct DLManager
{
    AnyTensor handle;
    DLManagedTensor tensor;
}

/// DLPack resource import buffer
private struct DLBuffer
{
    @nogc nothrow:

    AnyBuffer base;
    alias base this;

    DLManagedTensor* src;

    this(DLManagedTensor* src)
    {
        this.src = src;
        auto t = src.dl_tensor;
        this.base.payload = t.data[0 .. 0]; // dummy length
    }

    ~this() { src.deleter(src); }
}


/// dynamic tensor for type erasure and DLPack conversion
struct AnyTensor
{
    nothrow @nogc:

    import grain.dlpack : DLManagedTensor, DLContext, DLDataType;

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

    /** Convert AnyTensor to DLManagedTensor

        Returns:
            The DLManagedTensor to be consumed exactly once (i.e., call deleter once)

        See_also:
            https://github.com/pytorch/pytorch/blob/v1.2.0/aten/src/ATen/DLConvertor.cpp
     */
    DLManagedTensor* toDLPack()
    {
        import std.experimental.allocator : make, dispose;

        auto manager = CPUAllocator.instance.make!DLManager;
        manager.handle = this; // inc ref counts by copy
        auto ret = &manager.tensor;
        with (ret.dl_tensor)
        {
            ndim = cast(int) this.shape.length;
            byte_offset = this.offset;
            data = this.buffer.payload.ptr;
            shape = &(*this.shape)[0];
            strides = &(*this.strides)[0];
            ctx = this.context;
            dtype = this.dataType;
        }
        ret.manager_ctx = manager;
        ret.deleter = (DLManagedTensor* self) @trusted {
            // dec ref count by dtor in DLManager.handle
            CPUAllocator.instance.dispose(cast(DLManager*) self.manager_ctx);
        };
        return ret;
    }


    /// Load a new tensor from DLManagedTensor
    void fromDLPack(DLManagedTensor* src)
    {
        this.buffer = RC!DLBuffer.create(src).castTo!AnyBuffer;
        with (src.dl_tensor)
        {
            this.offset = byte_offset;
            this.shape = RC!(Array!long).create(shape[0 .. ndim]);
            this.strides = RC!(Array!long).create(strides[0 .. ndim]);
            this.context = ctx;
            this.dataType = dtype;
        }
    }
}

@nogc nothrow
unittest
{
    Array!byte bs;
    bs.length = 6;
    bs[] = 123;
    auto br = (&bs[0])[0 .. bs.length];
    AnyTensor a = { RC!AnyBuffer.create(br), 0, RC!(Array!long).create(2, 3), RC!(Array!long).create(3, 1) };
    assert(a.buffer._counter == 1);
    {
        auto d = a.toDLPack();
        // check contents equal without copy
        assert(d.dl_tensor.data == br.ptr);
        assert(d.dl_tensor.ndim == 2);
        assert(d.dl_tensor.shape == &(*a.shape)[0]);
        assert(d.dl_tensor.strides == &(*a.strides)[0]);

        assert(a.buffer._counter == 2);
        d.deleter(d);  // manually disposed
        assert(a.buffer._counter == 1);
    }

    {
        auto d = a.toDLPack();
        assert(a.buffer._counter == 2);
        AnyTensor b;
        b.fromDLPack(d);  // automatically disposed
        assert(a.buffer.payload.ptr == b.buffer.payload.ptr);
        assert((*a.shape) == (*b.shape));
        assert((*a.strides) == (*b.strides));
        assert(a.context == b.context);
        assert(a.dataType == b.dataType);

        assert(a.buffer._counter == 2);  // no increase
    }
    assert(a.buffer._counter == 1);
}

import grain.dlpack : DLDataType, kDLInt, kDLUInt, kDLFloat;
import std.traits : isIntegral, isFloatingPoint, isUnsigned, isSIMDVector;
enum DLDataType dataTypeOf(T) = {
    DLDataType ret;
    ret.bits = T.sizeof * 8;
    ret.lanes = 1;
    static if (is(T : __vector(V[N]), V, size_t N))
    {
        ret = dataTypeOf!V;
        ret.bits = V.sizeof * 8;
        ret.lanes = N;
    }
    else static if (isFloatingPoint!T)
    {
        ret.code = kDLFloat;
    }
    else static if (isUnsigned!T)
    {
        ret.code = kDLUInt;
    }
    else static if (isIntegral)
    {
        ret.code = kDLInt;
    }
    else
    {
        static assert(false, "cannot convert to DLDataType: " ~ T.stringof);
    }
    return ret;
}();

///
@nogc nothrow pure @safe unittest
{
    static assert(dataTypeOf!float == DLDataType(kDLFloat, 32, 1));

    alias float4 = __vector(float[4]);
    static assert(dataTypeOf!float4 == DLDataType(kDLFloat, 32, 4));
}

/// typed tensor
struct Tensor(T, size_t dim, Allocator = CPUAllocator)
{
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
    /// reference for any tensor source
    AnyTensor source;

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

// @nogc nothrow @system
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
