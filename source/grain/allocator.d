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

/// check allocator uses CPU memory
enum isCPU(Allocator) =
{
    import grain.dlpack.header : kDLCPU, kDLCPUPinned;
    auto device = Allocator.context.device_type;
    return device == kDLCPU || device == kDLCPUPinned;
}();
