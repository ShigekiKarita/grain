/// reference count
module grain.rc;

import std.experimental.allocator.mallocator : Mallocator;

import grain.array : Array;


/// Reference-counted pointer
struct RC(T, Allocator = Mallocator)
{
    import core.atomic : atomicOp;
    import grain.traits : isAllocator, hasDestructor;

    static assert(isAllocator!Allocator, "Allocator does not satisfy isAllocator concept");

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

        RC ret = void;
        ret.payload = cast(Payload*) Allocator.instance.allocate(Payload.sizeof);
        emplace!(T, Args)(&ret.payload.data, args);
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
        this(cast(RC) a);
    }

    /// dtor
    ~this()
    {
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

    @nogc ~this() {
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


/// alias for reference-counted array
alias RCArray(T, ArrayAllocator = Mallocator, PtrAllocator = Mallocator) =
        RC!(Array!(T, ArrayAllocator), PtrAllocator);

/// mutable array test
pure @system @nogc
unittest
{
    auto a = RCArray!int.create(10);
    a.slice[] = 0;
    assert(a.payload.count == 1);
    {
        auto b = a;
        assert(a.payload.count == 2);
        const c = b;
        assert(a.payload.count == 3);
        b.slice[1] = 1;
    }
    assert(a.slice[0] == 0);
    assert(a.slice[1] == 1);
    assert(a.payload.count == 1);
}

/// const array test
pure @system @nogc
unittest
{
    const a = RCArray!int.create(10);
    assert(a.payload.count == 1);
    {
        const b = a;
        assert(a.payload.count == 2);
        const c = b;
        assert(a.payload.count == 3);
    }
    assert(a.payload.count == 1);
}

/// mutable array with custom type test
@system @nogc
unittest
{
    const n = A.dtor;
    {
        import std.algorithm : move;
        auto a = RCArray!A.create(10);
        assert(a.payload.count == 1);
        {
            auto b = a;
            assert(a.payload.count == 2);
            const c = b;
            assert(a.payload.count == 3);
            b.slice[1].i = 1;
        }
        assert(a.slice[0].i == 123);
        assert(a.slice[1].i == 1);
        assert(a.payload.count == 1);
    }
    assert(A.dtor == n + 10);
}
