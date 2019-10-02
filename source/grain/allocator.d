module grain.allocator;


/// CPU malloc based heap allocator
struct CPUAllocator
{
    import std.experimental.allocator.mallocator : Mallocator;
    import grain.dlpack.header : DLContext, kDLCPU;

    static shared CPUAllocator instance;
    enum DLContext context = {device_type: kDLCPU, device_id: -1};
    Mallocator base;
    alias base this;

    static this() { instance = typeof(this)(Mallocator.instance); }
}
