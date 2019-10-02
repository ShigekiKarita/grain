/// Reference-counted buffer
module grain.buffer;

import std.experimental.allocator.mallocator : Mallocator;


/// element/allocator-generic buffer. Allocator should satisfy `isAllocator`
struct Buffer(T, Allocator = Mallocator)
{
    import grain.traits : hasDestructor;

    /// pointer to allocated buffer
    T* buffer;
    /// length of allocated buffer
    size_t length;
    shared Allocator* allocator;

    /// allocate uninitalized T values of `length`
    this(size_t length, ref shared Allocator allocator = Allocator.instance)
    {
        if (length == 0) return;

        this.allocator = &allocator;
        this.length = length;
        this.buffer = cast(T*) allocator.allocate(T.sizeof * length).ptr;
    }

    /// free allocated buffer
    nothrow ~this()
    {
        if (length == 0) return;

        static if (hasDestructor!T)
        {
            foreach (ref x; this.asSlice)
            {
                object.destroy(x);
            }
        }

        auto bufferPtr = cast(void*) this.ptr;
        auto n = T.sizeof * this.length;
        this.allocator.deallocate(bufferPtr[0 .. n]);
        this.buffer = null;
        this.length = 0;
    }

    /// mir slice
    auto asSlice()
    {
        import mir.ndslice.slice: mir_slice;
        return mir_slice!(T*)([length], buffer);
    }

    const asSlice() pure
    {
        import mir.ndslice.slice: mir_slice;
        return mir_slice!(const(T)*)([length], buffer);
    }

    alias asSlice this;
}


/// combination with mir.rc.ptr.RCPtr
@nogc nothrow @system
unittest
{
    import mir.rc.ptr : createRC;

    auto a = createRC!(Buffer!int)(3);
    {
        auto b = a;
        b[0] = 1;
        assert(a._counter == 2);
        auto c = a[0]; // not increase counter
        assert(a._counter == 2);
    }
    assert(a[0] == 1);
}

/// const buffer test
@nogc nothrow @system pure
unittest
{
    import mir.rc.ptr : createRC;

    const a = createRC!(Buffer!int)(3);
    assert(a._counter == 1);
    {
        const b = a;
        assert(a._counter == 2);
        const c = b;
        assert(a._counter == 3);
    }
    assert(a._counter == 1);
}

/// mutable buffer with custom type test
@nogc nothrow @system
unittest
{
    import std.conv : emplace;
    import mir.rc.ptr : createRC;

    struct A
    {
        int i = 123;
        double count = 20;
        static size_t dtor = 0;

        nothrow @nogc ~this()
        {
            ++dtor;
        }
    }

    const n = A.dtor;
    {
        auto a = createRC!(Buffer!A)(10);

        foreach (ref chunk; a)
        {
            emplace!A(&chunk);
        }

        assert(a._counter == 1);
        {
            auto b = a;
            assert(a._counter == 2);
            const c = b;
            assert(a._counter == 3);
            b.asSlice[1].i = 1;
        }
        assert(a.asSlice[0].i == 123);
        assert(a.asSlice[1].i == 1);
        assert(a._counter == 1);
    }
    assert(A.dtor == n + 10);
}
