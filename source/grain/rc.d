/// reference count
// deprecated("use mir.rc.ptr instead")
module grain.rc;

import grain.allocator : CPUAllocator;
import grain.buffer : Buffer;

/// Reference-counted pointer
struct RC(T)
{
    import core.atomic : atomicOp;
    import mir.rc.context;

    /// RC implementation
    mir_rc_context* _context;

    /// real data
    static if (is(T == class) || is(T == interface))
    {
        T _payload;
    }
    else
    {
        T* _payload;
    }

    alias _payload this;

    /// static ctor
    static create(Args ...)(auto ref Args args)
    {
        import std.experimental.allocator : make;
        import mir.conv : emplace;
        import mir.functional : forward;
        import mir.type_info : mir_get_type_info;

        RC ret = void;
        ret._payload = CPUAllocator.instance.make!T(forward!args);
        static if (is(T == class) || is(T == interface))
        {
            auto p = typeid(T).initializer.ptr;
        }
        else
        {
            auto p = ret._payload;
        }
        ret._context = mir_rc_create(mir_get_type_info!T, 1, p);
        return ret;
    }

    /// check the number of RC
    auto _counter() const { return _context ? _context.counter : 0; };

    /// manually increase RC (warning)
    void _incRef()
    {
        if (this)
        {
            mir_rc_increase_counter(*_context);
        }
    }

    /// copy ctor
    this(this) scope @trusted pure nothrow @nogc
    {
        this._incRef();
    }

    /// manually decrease RC (warning)
    void _decRef()
    {
        if (this)
        {
            mir_rc_decrease_counter(*_context);
        }
    }

    /// dtor
    nothrow ~this()
    {
        this._decRef();
    }

    /// polymorphism cast
    RC!Super castTo(Super)()
    {
        static assert(is(T : Super));
        typeof(return) ret;
        static if (is(T == class) || is(T == interface))
        {
            ret._payload = this._payload;
        }
        else
        {
            // `alias super this` can be only extracted by ref
            ret._payload = ((ref Super x) => &x)(*this._payload);
        }
        ret._context = this._context;
        this._incRef();
        return ret;
    }
}

/// OOP support
@nogc nothrow @system
unittest
{
    import mir.rc.ptr;

    static interface I { ref double bar() @safe pure nothrow @nogc; }
    static abstract class D { int index; }
    static class C : D, I
    {
        @nogc nothrow:
        double value;
        ref double bar() @safe pure nothrow @nogc { return value; }
        this(double d) { value = d; }
        static dtor = 0;
        ~this() {  ++dtor; }
    }

    {
        auto a = RC!C.create(10);
        assert(a._counter == 1);
        {
            auto b = a;
            assert(a._counter == 2);
            assert(b.value == 10);
            b.value = 100; // access via alias this syntax
            assert(a.value == 100);
            assert(a._counter == 2);

            auto d = a.castTo!D; //RCPtr!D
            assert(d._counter == 3);
            d.index = 234;
            assert(a.index == 234);
            auto i = a.castTo!I; //RCPtr!I
            assert(i.bar == 100);
            assert(i._counter == 4);
        }
        assert(a._counter == 1);
    }
    assert(C.dtor == 1);
}


/// 'Alias This' support
@nogc nothrow @system
unittest
{
    struct S
    {
        double e;
    }
    struct C
    {
        int i;
        S s;
        // 'alias' should be accesable by reference
        // or a class/interface
        alias s this;
    }

    auto a = RC!C.create(10, S(3));
    auto s = a.castTo!S; // RCPtr!S
    assert(s._counter == 2);
    assert(s.e == 3);
}

/// const test
@system @nogc
unittest
{
    struct A
    {
        nothrow @nogc:

        int i = 123;
        static size_t dtor = 0;

        this(int i) { this.i = i; }
        ~this() { ++dtor; }
    }

    const n = A.dtor;
    {
        const a = RC!A.create(3);
        {
            const b = a;
            assert(a._context.counter == 2);
            const c = b;
            assert(a._context.counter == 3);
        }
        assert(a.i == 3);
        assert(a._context.counter == 1);
        assert(A.dtor == n);
    }
    assert(A.dtor == n + 1);
}


/// alias for reference-countered buffer
alias RCBuffer(opts...) = RC!(Buffer!(opts));

/// mutable buffer test
@system @nogc
unittest
{
    auto a = RCBuffer!().create(10);
    a.bytes[] = 0;
    assert(a._context.counter == 1);
    {
        auto b = a;
        assert(a._context.counter == 2);
        const c = b;
        assert(a._context.counter == 3);
        b.bytes[1] = 1;
    }
    assert(a.bytes[0] == 0);
    assert(a.bytes[1] == 1);
    assert(a._context.counter == 1);
}

/// const buffer test
@system @nogc
unittest
{
    const a = RCBuffer!().create(10);
    assert(a._context.counter == 1);
    {
        const b = a;
        assert(a._context.counter == 2);
        const c = b;
        assert(a._context.counter == 3);
    }
    assert(a._context.counter == 1);
}
