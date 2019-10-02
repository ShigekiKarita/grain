/// Reference-counted array
module grain.array;

import std.experimental.allocator.mallocator : Mallocator;


/// element/allocator-generic array. Allocator should satisfy `isAllocator`
struct Array(T, Allocator = Mallocator)
{
    import grain.traits : isAllocator, hasDestructor;

    static assert(isAllocator!Allocator,
                  "Allocator does not satisfy isAllocator concept");

    /// pointer to allocated buffer
    T* buffer;
    /// length of allocated buffer
    size_t length;

    /// allocate uninitalized T values of `length`
    this(size_t length)
    {
        if (length == 0) return;

        this.length = length;
        this.buffer = cast(T*) Allocator.instance.allocate(T.sizeof * length).ptr;
    }

    /// call ctor with std.conv.emplace!T(chunk, args)
    void emplace(Args ...)(Args args)
    {
        import std.conv : emplace;
        import std.functional : forward;

        foreach (ref chunk; this.slice)
        {
            emplace!T(&chunk, forward!args);
        }
    }

    /// free allocated buffer
    ~this()
    {
        if (length == 0) return;

        static if (hasDestructor!T)
        {
            foreach (ref x; this.slice)
            {
                object.destroy(x);
            }
        }

        auto bufferPtr = cast(void*) this.ptr;
        auto n = T.sizeof * this.length;
        Allocator.instance.deallocate(bufferPtr[0 .. n]);
        this.buffer = null;
        this.length = 0;
    }

    /// typed slice
    inout(T)[] slice() inout scope return
    {
        return buffer[0 .. length];
    }

    alias slice this;
}
