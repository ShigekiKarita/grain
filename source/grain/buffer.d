/// Reference-counted buffer
module grain.buffer;

import grain.allocator : CPUAllocator;

/// dynamic buffer for type erasure
struct AnyBuffer
{
    /// pointer to allocated buffer
    void[] payload;
}

/// element/allocator-generic buffer. Allocator should satisfy `isAllocator`
struct Buffer(Allocator = CPUAllocator)
{
    AnyBuffer base;
    alias base this;
    shared Allocator* allocator;

    /// allocate uninitalized T values of `length`
    this(size_t length, ref shared Allocator allocator = Allocator.instance)
    {
        if (length == 0) return;
        this.allocator = &allocator;
        this.payload = allocator.allocate(length);
    }

    /// free allocated buffer
    nothrow ~this()
    {
        if (this.payload.length == 0) return;
        this.allocator.deallocate(this.payload);
        this.payload = [];
    }

    inout ptr()() { return this.payload.ptr; }

    size_t length() const { return this.payload.length; }

    nothrow @nogc inout(byte)[] bytes() inout { return (cast(inout(byte)*) this.ptr)[0 .. this.length]; }
}


/// combination with RC
@nogc nothrow @system
unittest
{
    import grain.rc : RC;

    auto a = RC!(Buffer!()).create(3);
    assert(a.length == 3);
    assert(a.ptr !is null);
    {
        auto b = a;
        b.bytes[0] = 1;
        assert(a._counter == 2);
        const c = a.bytes[0]; // NOTE: not increase counter
        assert(a._counter == 2);
        const i = a.castTo!AnyBuffer;
    }
    assert(a.bytes[0] == 1);

    auto z = RC!(Buffer!()).create(0);
    assert(z.length == 0);
    assert(z.ptr is null);
}
