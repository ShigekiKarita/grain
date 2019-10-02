/// reference count
// deprecated("use mir.rc.ptr instead")
module grain.rc;

import std.experimental.allocator.mallocator : Mallocator;

import grain.buffer : Buffer;


/// Reference-counted pointer
struct RC(T, Allocator = Mallocator)
{
    import core.atomic : atomicOp;

    /// storage for T and count
    private struct Payload
    {
        T data;
        shared long count;
    }

    ///
    private Payload* payload;

    alias get this;

    /// get raw pointer to payload
    inout(T)* get() inout
    {
        if (this.payload is null) return null;
        return &this.payload.data;
    }

    /// static ctor
    static create(Args ...)(auto ref Args args)
    {
        import std.conv : emplace;
        import std.functional : forward;

        RC ret = void;
        ret.payload = cast(Payload*) Allocator.instance.allocate(Payload.sizeof);
        emplace!T(&ret.payload.data, forward!args);
        ret.payload.count = 1;
        return ret;
    }

    /// copy ctor
    this(ref return scope RC a)
    {
        if (a.payload is null) return;
        a.payload.count.atomicOp!"+="(1);
        this.payload = a.payload;
    }

    /// ditto
    this(ref return scope const RC a) const
    {
        // NOTE: `this(cast(RC) a)` requires another copy ctor from 2.088.0
        this(*(cast(RC*) &a));
    }

    /// dtor
    nothrow ~this()
    {
        import grain.traits : hasDestructor;

        if (this.payload is null) return;

        this.payload.count.atomicOp!"-="(1);

        if (this.payload.count > 0) return;

        static if (hasDestructor!T)
        {
            object.destroy(this.payload.data);
        }

        auto bufferPtr = cast(void*) this.payload;
        Allocator.instance.deallocate(bufferPtr[0 .. T.sizeof]);
        this.payload = null;
    }
}

version (unittest)
private struct A
{
    int i = 123;
    double count = 20;
    static size_t dtor = 0;

    nothrow @nogc ~this() {
        ++dtor;
    }
}

/// mutable test
@system @nogc
unittest
{
    const n = A.dtor;
    {
        auto a = RC!A.create;
        assert(a.i == 123);
        {
            auto b = a;
            assert(a.payload.count == 2);
            b.i = 1;
            const c = b;
            assert(a.payload.count == 3);
        }
        assert(a.i == 1);
        assert(a.payload.count == 1);
        assert(A.dtor == n);
    }
    assert(A.dtor == n + 1);
}

/// const test
@system @nogc
unittest
{
    const n = A.dtor;
    {
        const a = RC!A.create(3);
        {
            const b = a;
            assert(a.payload.count == 2);
            const c = b;
            assert(a.payload.count == 3);
        }
        assert(a.i == 3);
        assert(a.payload.count == 1);
        assert(A.dtor == n);
    }
    assert(A.dtor == n + 1);
}


/// alias for reference-counted buffer
alias RCBuffer(T, BufferAllocator = Mallocator, PtrAllocator = Mallocator) =
        RC!(Buffer!(T, BufferAllocator), PtrAllocator);

/// mutable buffer test
pure @system @nogc
unittest
{
    auto a = RCBuffer!int.create(10);
    a.asSlice[] = 0;
    assert(a.payload.count == 1);
    {
        auto b = a;
        assert(a.payload.count == 2);
        const c = b;
        assert(a.payload.count == 3);
        b.asSlice[1] = 1;
    }
    assert(a.asSlice[0] == 0);
    assert(a.asSlice[1] == 1);
    assert(a.payload.count == 1);
}

/// const buffer test
pure @system @nogc
unittest
{
    const a = RCBuffer!int.create(10);
    assert(a.payload.count == 1);
    {
        const b = a;
        assert(a.payload.count == 2);
        const c = b;
        assert(a.payload.count == 3);
    }
    assert(a.payload.count == 1);
}

/// mutable buffer with custom type test
@system @nogc
unittest
{
    import std.conv : emplace;

    const n = A.dtor;
    {
        auto a = RCBuffer!A.create(10);
        foreach (ref chunk; a.asSlice)
        {
            emplace!A(&chunk);
        }

        assert(a.payload.count == 1);
        {
            auto b = a;
            assert(a.payload.count == 2);
            const c = b;
            assert(a.payload.count == 3);
            b.asSlice[1].i = 1;
        }
        assert(a.asSlice[0].i == 123);
        assert(a.asSlice[1].i == 1);
        assert(a.payload.count == 1);
    }
    assert(A.dtor == n + 10);
}
