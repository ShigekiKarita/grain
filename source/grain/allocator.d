module grain.allocator;

struct CPUAllocator
{
    import std.experimental.allocator.mallocator : Mallocator;

    static shared CPUAllocator instance;

    static this() { instance = typeof(this)(Mallocator.instance); }

    Mallocator base;
    alias base this;
}
