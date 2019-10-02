module grain.allocator;

struct CPUAllocator
{
    import std.experimental.allocator.mallocator : Mallocator;

    alias instance = Mallocator.instance;

    Mallocator base;
    alias base this;
}
