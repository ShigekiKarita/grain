module grain.dlpack.conv;

import std.container.array : Array;
import std.traits : isIntegral, isFloatingPoint, isUnsigned, isSIMDVector;

import grain.buffer : AnyBuffer;
import grain.tensor : AnyTensor;
import grain.dlpack.header;
import grain.dlpack.header : DLDataType, kDLInt, kDLUInt, kDLFloat;


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

/** Convert AnyTensor to DLManagedTensor

    Returns:
    The DLManagedTensor to be consumed exactly once (i.e., call deleter once)

    See_also:
    https://github.com/pytorch/pytorch/blob/v1.2.0/aten/src/ATen/DLConvertor.cpp
*/
@nogc nothrow
DLManagedTensor* toDLPack(AnyTensor src)
{
    import std.experimental.allocator : make, dispose;
    import std.experimental.allocator.mallocator : Mallocator;

    auto manager = Mallocator.instance.make!DLManager;
    manager.handle = src; // inc ref counts by copy
    auto ret = &manager.tensor;
    with (ret.dl_tensor)
    {
        ndim = cast(int) src.shape.length;
        byte_offset = src.offset;
        data = src.buffer.payload.ptr;
        shape = &(*src.shape)[0];
        strides = &(*src.strides)[0];
        ctx = src.context;
        dtype = src.dataType;
    }
    ret.manager_ctx = manager;
    ret.deleter = (DLManagedTensor* self) @trusted {
        // dec ref count by dtor in DLManager.handle
        Mallocator.instance.dispose(cast(DLManager*) self.manager_ctx);
    };
    return ret;
}

/// Load a new AnyTensor from DLManagedTensor
@nogc nothrow
AnyTensor toAny(DLManagedTensor* src)
{
    import grain.rc : RC;
    AnyTensor ret;
    ret.buffer = RC!DLBuffer.create(src).castTo!AnyBuffer;
    with (src.dl_tensor)
    {
        ret.offset = byte_offset;
        ret.shape = RC!(Array!long).create(shape[0 .. ndim]);
        ret.strides = RC!(Array!long).create(strides[0 .. ndim]);
        ret.context = ctx;
        ret.dataType = dtype;
    }
    return ret;
}

@nogc nothrow
unittest
{
    import grain.rc : RC;

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
        AnyTensor b = d.toAny;  // automatically disposed
        assert(a.buffer.payload.ptr == b.buffer.payload.ptr);
        assert((*a.shape) == (*b.shape));
        assert((*a.strides) == (*b.strides));
        assert(a.context == b.context);
        assert(a.dataType == b.dataType);

        assert(a.buffer._counter == 2);  // no increase
    }
    assert(a.buffer._counter == 1);
}
