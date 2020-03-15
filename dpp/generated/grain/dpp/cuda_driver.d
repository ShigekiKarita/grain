module grain.dpp.cuda_driver;


        import core.stdc.config;
        import core.stdc.stdarg: va_list;
        static import core.simd;
        static import std.conv;

        struct Int128 { long lower; long upper; }
        struct UInt128 { ulong lower; ulong upper; }

        struct __locale_data { int dummy; }



alias _Bool = bool;
struct dpp {
    static struct Opaque(int N) {
        void[N] bytes;
    }

    static bool isEmpty(T)() {
        return T.tupleof.length == 0;
    }
    static struct Move(T) {
        T* ptr;
    }


    static auto move(T)(ref T value) {
        return Move!T(&value);
    }
    mixin template EnumD(string name, T, string prefix) if(is(T == enum)) {
        private static string _memberMixinStr(string member) {
            import std.conv: text;
            import std.array: replace;
            return text(` `, member.replace(prefix, ""), ` = `, T.stringof, `.`, member, `,`);
        }
        private static string _enumMixinStr() {
            import std.array: join;
            string[] ret;
            ret ~= "enum " ~ name ~ "{";
            static foreach(member; __traits(allMembers, T)) {
                ret ~= _memberMixinStr(member);
            }
            ret ~= "}";
            return ret.join("\n");
        }
        mixin(_enumMixinStr());
    }
}

extern(C)
{
    cudaError_enum cuGetExportTable(const(void)**, const(CUuuid_st)*) @nogc nothrow;
    cudaError_enum cuGraphicsUnmapResources(uint, CUgraphicsResource_st**, CUstream_st*) @nogc nothrow;
    cudaError_enum cuGraphicsMapResources(uint, CUgraphicsResource_st**, CUstream_st*) @nogc nothrow;
    cudaError_enum cuGraphicsResourceSetMapFlags_v2(CUgraphicsResource_st*, uint) @nogc nothrow;
    cudaError_enum cuGraphicsResourceGetMappedPointer_v2(ulong*, c_ulong*, CUgraphicsResource_st*) @nogc nothrow;
    cudaError_enum cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray_st**, CUgraphicsResource_st*) @nogc nothrow;
    cudaError_enum cuGraphicsSubResourceGetMappedArray(CUarray_st**, CUgraphicsResource_st*, uint, uint) @nogc nothrow;
    cudaError_enum cuGraphicsUnregisterResource(CUgraphicsResource_st*) @nogc nothrow;
    cudaError_enum cuDeviceGetP2PAttribute(int*, CUdevice_P2PAttribute_enum, int, int) @nogc nothrow;
    cudaError_enum cuCtxDisablePeerAccess(CUctx_st*) @nogc nothrow;
    cudaError_enum cuCtxEnablePeerAccess(CUctx_st*, uint) @nogc nothrow;
    cudaError_enum cuDeviceCanAccessPeer(int*, int, int) @nogc nothrow;
    cudaError_enum cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC_st*, ulong) @nogc nothrow;
    cudaError_enum cuSurfObjectDestroy(ulong) @nogc nothrow;
    cudaError_enum cuSurfObjectCreate(ulong*, const(CUDA_RESOURCE_DESC_st)*) @nogc nothrow;
    cudaError_enum cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC_st*, ulong) @nogc nothrow;
    cudaError_enum cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC_st*, ulong) @nogc nothrow;
    cudaError_enum cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC_st*, ulong) @nogc nothrow;
    cudaError_enum cuTexObjectDestroy(ulong) @nogc nothrow;
    cudaError_enum cuTexObjectCreate(ulong*, const(CUDA_RESOURCE_DESC_st)*, const(CUDA_TEXTURE_DESC_st)*, const(CUDA_RESOURCE_VIEW_DESC_st)*) @nogc nothrow;
    cudaError_enum cuSurfRefGetArray(CUarray_st**, CUsurfref_st*) @nogc nothrow;
    cudaError_enum cuSurfRefSetArray(CUsurfref_st*, CUarray_st*, uint) @nogc nothrow;
    cudaError_enum cuTexRefDestroy(CUtexref_st*) @nogc nothrow;
    cudaError_enum cuTexRefCreate(CUtexref_st**) @nogc nothrow;
    cudaError_enum cuTexRefGetFlags(uint*, CUtexref_st*) @nogc nothrow;
    cudaError_enum cuTexRefGetBorderColor(float*, CUtexref_st*) @nogc nothrow;
    cudaError_enum cuTexRefGetMaxAnisotropy(int*, CUtexref_st*) @nogc nothrow;
    cudaError_enum cuTexRefGetMipmapLevelClamp(float*, float*, CUtexref_st*) @nogc nothrow;
    cudaError_enum cuTexRefGetMipmapLevelBias(float*, CUtexref_st*) @nogc nothrow;
    cudaError_enum cuTexRefGetMipmapFilterMode(CUfilter_mode_enum*, CUtexref_st*) @nogc nothrow;
    cudaError_enum cuTexRefGetFormat(CUarray_format_enum*, int*, CUtexref_st*) @nogc nothrow;
    cudaError_enum cuTexRefGetFilterMode(CUfilter_mode_enum*, CUtexref_st*) @nogc nothrow;
    cudaError_enum cuTexRefGetAddressMode(CUaddress_mode_enum*, CUtexref_st*, int) @nogc nothrow;
    cudaError_enum cuTexRefGetMipmappedArray(CUmipmappedArray_st**, CUtexref_st*) @nogc nothrow;
    cudaError_enum cuTexRefGetArray(CUarray_st**, CUtexref_st*) @nogc nothrow;
    cudaError_enum cuTexRefGetAddress_v2(ulong*, CUtexref_st*) @nogc nothrow;
    cudaError_enum cuTexRefSetFlags(CUtexref_st*, uint) @nogc nothrow;
    cudaError_enum cuTexRefSetBorderColor(CUtexref_st*, float*) @nogc nothrow;
    cudaError_enum cuTexRefSetMaxAnisotropy(CUtexref_st*, uint) @nogc nothrow;
    cudaError_enum cuTexRefSetMipmapLevelClamp(CUtexref_st*, float, float) @nogc nothrow;
    cudaError_enum cuTexRefSetMipmapLevelBias(CUtexref_st*, float) @nogc nothrow;
    cudaError_enum cuTexRefSetMipmapFilterMode(CUtexref_st*, CUfilter_mode_enum) @nogc nothrow;
    cudaError_enum cuTexRefSetFilterMode(CUtexref_st*, CUfilter_mode_enum) @nogc nothrow;
    cudaError_enum cuTexRefSetAddressMode(CUtexref_st*, int, CUaddress_mode_enum) @nogc nothrow;
    cudaError_enum cuTexRefSetFormat(CUtexref_st*, CUarray_format_enum, int) @nogc nothrow;
    cudaError_enum cuTexRefSetAddress2D_v3(CUtexref_st*, const(CUDA_ARRAY_DESCRIPTOR_st)*, ulong, c_ulong) @nogc nothrow;
    cudaError_enum cuTexRefSetAddress_v2(c_ulong*, CUtexref_st*, ulong, c_ulong) @nogc nothrow;
    cudaError_enum cuTexRefSetMipmappedArray(CUtexref_st*, CUmipmappedArray_st*, uint) @nogc nothrow;
    cudaError_enum cuTexRefSetArray(CUtexref_st*, CUarray_st*, uint) @nogc nothrow;
    cudaError_enum cuOccupancyMaxPotentialBlockSizeWithFlags(int*, int*, CUfunc_st*, c_ulong function(int), c_ulong, int, uint) @nogc nothrow;
    cudaError_enum cuOccupancyMaxPotentialBlockSize(int*, int*, CUfunc_st*, c_ulong function(int), c_ulong, int) @nogc nothrow;
    cudaError_enum cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int*, CUfunc_st*, int, c_ulong, uint) @nogc nothrow;
    cudaError_enum cuOccupancyMaxActiveBlocksPerMultiprocessor(int*, CUfunc_st*, int, c_ulong) @nogc nothrow;
    cudaError_enum cuGraphExecUpdate(CUgraphExec_st*, CUgraph_st*, CUgraphNode_st**, CUgraphExecUpdateResult_enum*) @nogc nothrow;
    cudaError_enum cuGraphDestroy(CUgraph_st*) @nogc nothrow;
    cudaError_enum cuGraphExecDestroy(CUgraphExec_st*) @nogc nothrow;
    cudaError_enum cuGraphLaunch(CUgraphExec_st*, CUstream_st*) @nogc nothrow;
    cudaError_enum cuGraphExecHostNodeSetParams(CUgraphExec_st*, CUgraphNode_st*, const(CUDA_HOST_NODE_PARAMS_st)*) @nogc nothrow;
    cudaError_enum cuGraphExecMemsetNodeSetParams(CUgraphExec_st*, CUgraphNode_st*, const(CUDA_MEMSET_NODE_PARAMS_st)*, CUctx_st*) @nogc nothrow;
    cudaError_enum cuGraphExecMemcpyNodeSetParams(CUgraphExec_st*, CUgraphNode_st*, const(CUDA_MEMCPY3D_st)*, CUctx_st*) @nogc nothrow;
    cudaError_enum cuGraphExecKernelNodeSetParams(CUgraphExec_st*, CUgraphNode_st*, const(CUDA_KERNEL_NODE_PARAMS_st)*) @nogc nothrow;
    cudaError_enum cuGraphInstantiate(CUgraphExec_st**, CUgraph_st*, CUgraphNode_st**, char*, c_ulong) @nogc nothrow;
    cudaError_enum cuGraphDestroyNode(CUgraphNode_st*) @nogc nothrow;
    cudaError_enum cuGraphRemoveDependencies(CUgraph_st*, const(CUgraphNode_st*)*, const(CUgraphNode_st*)*, c_ulong) @nogc nothrow;
    cudaError_enum cuGraphAddDependencies(CUgraph_st*, const(CUgraphNode_st*)*, const(CUgraphNode_st*)*, c_ulong) @nogc nothrow;
    cudaError_enum cuGraphNodeGetDependentNodes(CUgraphNode_st*, CUgraphNode_st**, c_ulong*) @nogc nothrow;
    cudaError_enum cuGraphNodeGetDependencies(CUgraphNode_st*, CUgraphNode_st**, c_ulong*) @nogc nothrow;
    cudaError_enum cuGraphGetEdges(CUgraph_st*, CUgraphNode_st**, CUgraphNode_st**, c_ulong*) @nogc nothrow;
    cudaError_enum cuGraphGetRootNodes(CUgraph_st*, CUgraphNode_st**, c_ulong*) @nogc nothrow;
    cudaError_enum cuGraphGetNodes(CUgraph_st*, CUgraphNode_st**, c_ulong*) @nogc nothrow;
    cudaError_enum cuGraphNodeGetType(CUgraphNode_st*, CUgraphNodeType_enum*) @nogc nothrow;
    cudaError_enum cuGraphNodeFindInClone(CUgraphNode_st**, CUgraphNode_st*, CUgraph_st*) @nogc nothrow;
    cudaError_enum cuGraphClone(CUgraph_st**, CUgraph_st*) @nogc nothrow;
    cudaError_enum cuGraphAddEmptyNode(CUgraphNode_st**, CUgraph_st*, const(CUgraphNode_st*)*, c_ulong) @nogc nothrow;
    cudaError_enum cuGraphChildGraphNodeGetGraph(CUgraphNode_st*, CUgraph_st**) @nogc nothrow;
    cudaError_enum cuGraphAddChildGraphNode(CUgraphNode_st**, CUgraph_st*, const(CUgraphNode_st*)*, c_ulong, CUgraph_st*) @nogc nothrow;
    cudaError_enum cuGraphHostNodeSetParams(CUgraphNode_st*, const(CUDA_HOST_NODE_PARAMS_st)*) @nogc nothrow;
    cudaError_enum cuGraphHostNodeGetParams(CUgraphNode_st*, CUDA_HOST_NODE_PARAMS_st*) @nogc nothrow;
    cudaError_enum cuGraphAddHostNode(CUgraphNode_st**, CUgraph_st*, const(CUgraphNode_st*)*, c_ulong, const(CUDA_HOST_NODE_PARAMS_st)*) @nogc nothrow;
    cudaError_enum cuGraphMemsetNodeSetParams(CUgraphNode_st*, const(CUDA_MEMSET_NODE_PARAMS_st)*) @nogc nothrow;
    cudaError_enum cuGraphMemsetNodeGetParams(CUgraphNode_st*, CUDA_MEMSET_NODE_PARAMS_st*) @nogc nothrow;
    cudaError_enum cuGraphAddMemsetNode(CUgraphNode_st**, CUgraph_st*, const(CUgraphNode_st*)*, c_ulong, const(CUDA_MEMSET_NODE_PARAMS_st)*, CUctx_st*) @nogc nothrow;
    cudaError_enum cuGraphMemcpyNodeSetParams(CUgraphNode_st*, const(CUDA_MEMCPY3D_st)*) @nogc nothrow;
    cudaError_enum cuGraphMemcpyNodeGetParams(CUgraphNode_st*, CUDA_MEMCPY3D_st*) @nogc nothrow;
    cudaError_enum cuGraphAddMemcpyNode(CUgraphNode_st**, CUgraph_st*, const(CUgraphNode_st*)*, c_ulong, const(CUDA_MEMCPY3D_st)*, CUctx_st*) @nogc nothrow;
    cudaError_enum cuGraphKernelNodeSetParams(CUgraphNode_st*, const(CUDA_KERNEL_NODE_PARAMS_st)*) @nogc nothrow;
    cudaError_enum cuGraphKernelNodeGetParams(CUgraphNode_st*, CUDA_KERNEL_NODE_PARAMS_st*) @nogc nothrow;
    cudaError_enum cuGraphAddKernelNode(CUgraphNode_st**, CUgraph_st*, const(CUgraphNode_st*)*, c_ulong, const(CUDA_KERNEL_NODE_PARAMS_st)*) @nogc nothrow;
    cudaError_enum cuGraphCreate(CUgraph_st**, uint) @nogc nothrow;
    cudaError_enum cuParamSetTexRef(CUfunc_st*, int, CUtexref_st*) @nogc nothrow;
    cudaError_enum cuLaunchGridAsync(CUfunc_st*, int, int, CUstream_st*) @nogc nothrow;
    cudaError_enum cuLaunchGrid(CUfunc_st*, int, int) @nogc nothrow;
    cudaError_enum cuLaunch(CUfunc_st*) @nogc nothrow;
    cudaError_enum cuParamSetv(CUfunc_st*, int, void*, uint) @nogc nothrow;
    cudaError_enum cuParamSetf(CUfunc_st*, int, float) @nogc nothrow;
    cudaError_enum cuParamSeti(CUfunc_st*, int, uint) @nogc nothrow;
    cudaError_enum cuParamSetSize(CUfunc_st*, uint) @nogc nothrow;
    cudaError_enum cuFuncSetSharedSize(CUfunc_st*, uint) @nogc nothrow;
    cudaError_enum cuFuncSetBlockShape(CUfunc_st*, int, int, int) @nogc nothrow;
    cudaError_enum cuLaunchHostFunc(CUstream_st*, void function(void*), void*) @nogc nothrow;
    cudaError_enum cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS_st*, uint, uint) @nogc nothrow;
    cudaError_enum cuLaunchCooperativeKernel(CUfunc_st*, uint, uint, uint, uint, uint, uint, uint, CUstream_st*, void**) @nogc nothrow;
    cudaError_enum cuLaunchKernel(CUfunc_st*, uint, uint, uint, uint, uint, uint, uint, CUstream_st*, void**, void**) @nogc nothrow;
    cudaError_enum cuFuncSetSharedMemConfig(CUfunc_st*, CUsharedconfig_enum) @nogc nothrow;
    cudaError_enum cuFuncSetCacheConfig(CUfunc_st*, CUfunc_cache_enum) @nogc nothrow;
    cudaError_enum cuFuncSetAttribute(CUfunc_st*, CUfunction_attribute_enum, int) @nogc nothrow;
    cudaError_enum cuFuncGetAttribute(int*, CUfunction_attribute_enum, CUfunc_st*) @nogc nothrow;
    cudaError_enum cuStreamBatchMemOp(CUstream_st*, uint, CUstreamBatchMemOpParams_union*, uint) @nogc nothrow;
    cudaError_enum cuStreamWriteValue64(CUstream_st*, ulong, c_ulong, uint) @nogc nothrow;
    cudaError_enum cuStreamWriteValue32(CUstream_st*, ulong, uint, uint) @nogc nothrow;
    cudaError_enum cuStreamWaitValue64(CUstream_st*, ulong, c_ulong, uint) @nogc nothrow;
    cudaError_enum cuStreamWaitValue32(CUstream_st*, ulong, uint, uint) @nogc nothrow;
    cudaError_enum cuDestroyExternalSemaphore(CUextSemaphore_st*) @nogc nothrow;
    cudaError_enum cuWaitExternalSemaphoresAsync(const(CUextSemaphore_st*)*, const(CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st)*, uint, CUstream_st*) @nogc nothrow;
    cudaError_enum cuSignalExternalSemaphoresAsync(const(CUextSemaphore_st*)*, const(CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st)*, uint, CUstream_st*) @nogc nothrow;
    cudaError_enum cuImportExternalSemaphore(CUextSemaphore_st**, const(CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st)*) @nogc nothrow;
    cudaError_enum cuDestroyExternalMemory(CUextMemory_st*) @nogc nothrow;
    cudaError_enum cuExternalMemoryGetMappedMipmappedArray(CUmipmappedArray_st**, CUextMemory_st*, const(CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st)*) @nogc nothrow;
    cudaError_enum cuExternalMemoryGetMappedBuffer(ulong*, CUextMemory_st*, const(CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st)*) @nogc nothrow;
    cudaError_enum cuImportExternalMemory(CUextMemory_st**, const(CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st)*) @nogc nothrow;
    cudaError_enum cuEventElapsedTime(float*, CUevent_st*, CUevent_st*) @nogc nothrow;
    cudaError_enum cuEventDestroy_v2(CUevent_st*) @nogc nothrow;
    cudaError_enum cuEventSynchronize(CUevent_st*) @nogc nothrow;
    cudaError_enum cuEventQuery(CUevent_st*) @nogc nothrow;
    cudaError_enum cuEventRecord(CUevent_st*, CUstream_st*) @nogc nothrow;
    cudaError_enum cuEventCreate(CUevent_st**, uint) @nogc nothrow;
    cudaError_enum cuStreamDestroy_v2(CUstream_st*) @nogc nothrow;
    cudaError_enum cuStreamSynchronize(CUstream_st*) @nogc nothrow;
    cudaError_enum cuStreamQuery(CUstream_st*) @nogc nothrow;
    cudaError_enum cuStreamAttachMemAsync(CUstream_st*, ulong, c_ulong, uint) @nogc nothrow;
    cudaError_enum cuStreamGetCaptureInfo(CUstream_st*, CUstreamCaptureStatus_enum*, c_ulong*) @nogc nothrow;
    cudaError_enum cuStreamIsCapturing(CUstream_st*, CUstreamCaptureStatus_enum*) @nogc nothrow;
    cudaError_enum cuStreamEndCapture(CUstream_st*, CUgraph_st**) @nogc nothrow;
    cudaError_enum cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode_enum*) @nogc nothrow;
    cudaError_enum cuStreamBeginCapture_v2(CUstream_st*, CUstreamCaptureMode_enum) @nogc nothrow;
    cudaError_enum cuStreamAddCallback(CUstream_st*, void function(CUstream_st*, cudaError_enum, void*), void*, uint) @nogc nothrow;
    cudaError_enum cuStreamWaitEvent(CUstream_st*, CUevent_st*, uint) @nogc nothrow;
    cudaError_enum cuStreamGetCtx(CUstream_st*, CUctx_st**) @nogc nothrow;
    cudaError_enum cuStreamGetFlags(CUstream_st*, uint*) @nogc nothrow;
    cudaError_enum cuStreamGetPriority(CUstream_st*, int*) @nogc nothrow;
    cudaError_enum cuStreamCreateWithPriority(CUstream_st**, uint, int) @nogc nothrow;
    cudaError_enum cuStreamCreate(CUstream_st**, uint) @nogc nothrow;
    cudaError_enum cuPointerGetAttributes(uint, CUpointer_attribute_enum*, void**, ulong) @nogc nothrow;
    cudaError_enum cuPointerSetAttribute(const(void)*, CUpointer_attribute_enum, ulong) @nogc nothrow;
    cudaError_enum cuMemRangeGetAttributes(void**, c_ulong*, CUmem_range_attribute_enum*, c_ulong, ulong, c_ulong) @nogc nothrow;
    cudaError_enum cuMemRangeGetAttribute(void*, c_ulong, CUmem_range_attribute_enum, ulong, c_ulong) @nogc nothrow;
    cudaError_enum cuMemAdvise(ulong, c_ulong, CUmem_advise_enum, int) @nogc nothrow;
    pragma(mangle, "alloca") void* alloca_(c_ulong) @nogc nothrow;
    cudaError_enum cuMemPrefetchAsync(ulong, c_ulong, int, CUstream_st*) @nogc nothrow;
    alias size_t = c_ulong;
    alias wchar_t = int;
    cudaError_enum cuPointerGetAttribute(void*, CUpointer_attribute_enum, ulong) @nogc nothrow;
    cudaError_enum cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp_st*, ulong) @nogc nothrow;
    cudaError_enum cuMemGetAllocationGranularity(c_ulong*, const(CUmemAllocationProp_st)*, CUmemAllocationGranularity_flags_enum) @nogc nothrow;
    cudaError_enum cuMemImportFromShareableHandle(ulong*, void*, CUmemAllocationHandleType_enum) @nogc nothrow;
    cudaError_enum cuMemExportToShareableHandle(void*, ulong, CUmemAllocationHandleType_enum, ulong) @nogc nothrow;
    cudaError_enum cuMemGetAccess(ulong*, const(CUmemLocation_st)*, ulong) @nogc nothrow;
    cudaError_enum cuMemSetAccess(ulong, c_ulong, const(CUmemAccessDesc_st)*, c_ulong) @nogc nothrow;
    cudaError_enum cuMemUnmap(ulong, c_ulong) @nogc nothrow;
    cudaError_enum cuMemMap(ulong, c_ulong, c_ulong, ulong, ulong) @nogc nothrow;
    cudaError_enum cuMemRelease(ulong) @nogc nothrow;
    cudaError_enum cuMemCreate(ulong*, c_ulong, const(CUmemAllocationProp_st)*, ulong) @nogc nothrow;
    cudaError_enum cuMemAddressFree(ulong, c_ulong) @nogc nothrow;
    cudaError_enum cuMemAddressReserve(ulong*, c_ulong, c_ulong, ulong, ulong) @nogc nothrow;
    cudaError_enum cuMipmappedArrayDestroy(CUmipmappedArray_st*) @nogc nothrow;
    cudaError_enum cuMipmappedArrayGetLevel(CUarray_st**, CUmipmappedArray_st*, uint) @nogc nothrow;
    cudaError_enum cuMipmappedArrayCreate(CUmipmappedArray_st**, const(CUDA_ARRAY3D_DESCRIPTOR_st)*, uint) @nogc nothrow;
    cudaError_enum cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR_st*, CUarray_st*) @nogc nothrow;
    cudaError_enum cuArray3DCreate_v2(CUarray_st**, const(CUDA_ARRAY3D_DESCRIPTOR_st)*) @nogc nothrow;
    cudaError_enum cuArrayDestroy(CUarray_st*) @nogc nothrow;
    cudaError_enum cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR_st*, CUarray_st*) @nogc nothrow;
    cudaError_enum cuArrayCreate_v2(CUarray_st**, const(CUDA_ARRAY_DESCRIPTOR_st)*) @nogc nothrow;
    cudaError_enum cuMemsetD2D32Async(ulong, c_ulong, uint, c_ulong, c_ulong, CUstream_st*) @nogc nothrow;
    cudaError_enum cuMemsetD2D16Async(ulong, c_ulong, ushort, c_ulong, c_ulong, CUstream_st*) @nogc nothrow;
    cudaError_enum cuMemsetD2D8Async(ulong, c_ulong, ubyte, c_ulong, c_ulong, CUstream_st*) @nogc nothrow;
    cudaError_enum cuMemsetD32Async(ulong, uint, c_ulong, CUstream_st*) @nogc nothrow;
    alias int_least8_t = byte;
    alias int_least16_t = short;
    alias int_least32_t = int;
    alias int_least64_t = c_long;
    alias uint_least8_t = ubyte;
    alias uint_least16_t = ushort;
    alias uint_least32_t = uint;
    alias uint_least64_t = c_ulong;
    alias int_fast8_t = byte;
    alias int_fast16_t = c_long;
    alias int_fast32_t = c_long;
    alias int_fast64_t = c_long;
    alias uint_fast8_t = ubyte;
    alias uint_fast16_t = c_ulong;
    alias uint_fast32_t = c_ulong;
    alias uint_fast64_t = c_ulong;
    alias intptr_t = c_long;
    cudaError_enum cuMemsetD16Async(ulong, ushort, c_ulong, CUstream_st*) @nogc nothrow;
    alias uintptr_t = c_ulong;
    alias intmax_t = c_long;
    alias uintmax_t = c_ulong;
    cudaError_enum cuMemsetD8Async(ulong, ubyte, c_ulong, CUstream_st*) @nogc nothrow;
    cudaError_enum cuMemsetD2D32_v2(ulong, c_ulong, uint, c_ulong, c_ulong) @nogc nothrow;
    cudaError_enum cuMemsetD2D16_v2(ulong, c_ulong, ushort, c_ulong, c_ulong) @nogc nothrow;
    cudaError_enum cuMemsetD2D8_v2(ulong, c_ulong, ubyte, c_ulong, c_ulong) @nogc nothrow;
    cudaError_enum cuMemsetD32_v2(ulong, uint, c_ulong) @nogc nothrow;
    cudaError_enum cuMemsetD16_v2(ulong, ushort, c_ulong) @nogc nothrow;
    cudaError_enum cuMemsetD8_v2(ulong, ubyte, c_ulong) @nogc nothrow;
    cudaError_enum cuMemcpy3DPeerAsync(const(CUDA_MEMCPY3D_PEER_st)*, CUstream_st*) @nogc nothrow;
    cudaError_enum cuMemcpy3DAsync_v2(const(CUDA_MEMCPY3D_st)*, CUstream_st*) @nogc nothrow;
    cudaError_enum cuMemcpy2DAsync_v2(const(CUDA_MEMCPY2D_st)*, CUstream_st*) @nogc nothrow;
    cudaError_enum cuMemcpyAtoHAsync_v2(void*, CUarray_st*, c_ulong, c_ulong, CUstream_st*) @nogc nothrow;
    cudaError_enum cuMemcpyHtoAAsync_v2(CUarray_st*, c_ulong, const(void)*, c_ulong, CUstream_st*) @nogc nothrow;
    cudaError_enum cuMemcpyDtoDAsync_v2(ulong, ulong, c_ulong, CUstream_st*) @nogc nothrow;
    cudaError_enum cuMemcpyDtoHAsync_v2(void*, ulong, c_ulong, CUstream_st*) @nogc nothrow;
    cudaError_enum cuMemcpyHtoDAsync_v2(ulong, const(void)*, c_ulong, CUstream_st*) @nogc nothrow;
    cudaError_enum cuMemcpyPeerAsync(ulong, CUctx_st*, ulong, CUctx_st*, c_ulong, CUstream_st*) @nogc nothrow;
    cudaError_enum cuMemcpyAsync(ulong, ulong, c_ulong, CUstream_st*) @nogc nothrow;
    cudaError_enum cuMemcpy3DPeer(const(CUDA_MEMCPY3D_PEER_st)*) @nogc nothrow;
    cudaError_enum cuMemcpy3D_v2(const(CUDA_MEMCPY3D_st)*) @nogc nothrow;
    cudaError_enum cuMemcpy2DUnaligned_v2(const(CUDA_MEMCPY2D_st)*) @nogc nothrow;
    cudaError_enum cuMemcpy2D_v2(const(CUDA_MEMCPY2D_st)*) @nogc nothrow;
    cudaError_enum cuMemcpyAtoA_v2(CUarray_st*, c_ulong, CUarray_st*, c_ulong, c_ulong) @nogc nothrow;
    cudaError_enum cuMemcpyAtoH_v2(void*, CUarray_st*, c_ulong, c_ulong) @nogc nothrow;
    cudaError_enum cuMemcpyHtoA_v2(CUarray_st*, c_ulong, const(void)*, c_ulong) @nogc nothrow;
    cudaError_enum cuMemcpyAtoD_v2(ulong, CUarray_st*, c_ulong, c_ulong) @nogc nothrow;
    struct div_t
    {
        int quot;
        int rem;
    }
    struct ldiv_t
    {
        c_long quot;
        c_long rem;
    }
    struct lldiv_t
    {
        long quot;
        long rem;
    }
    cudaError_enum cuMemcpyDtoA_v2(CUarray_st*, c_ulong, ulong, c_ulong) @nogc nothrow;
    cudaError_enum cuMemcpyDtoD_v2(ulong, ulong, c_ulong) @nogc nothrow;
    c_ulong __ctype_get_mb_cur_max() @nogc nothrow;
    double atof(const(char)*) @nogc nothrow;
    int atoi(const(char)*) @nogc nothrow;
    c_long atol(const(char)*) @nogc nothrow;
    long atoll(const(char)*) @nogc nothrow;
    double strtod(const(char)*, char**) @nogc nothrow;
    float strtof(const(char)*, char**) @nogc nothrow;
    real strtold(const(char)*, char**) @nogc nothrow;
    c_long strtol(const(char)*, char**, int) @nogc nothrow;
    c_ulong strtoul(const(char)*, char**, int) @nogc nothrow;
    long strtoq(const(char)*, char**, int) @nogc nothrow;
    ulong strtouq(const(char)*, char**, int) @nogc nothrow;
    long strtoll(const(char)*, char**, int) @nogc nothrow;
    ulong strtoull(const(char)*, char**, int) @nogc nothrow;
    char* l64a(c_long) @nogc nothrow;
    c_long a64l(const(char)*) @nogc nothrow;
    c_long random() @nogc nothrow;
    void srandom(uint) @nogc nothrow;
    char* initstate(uint, char*, c_ulong) @nogc nothrow;
    char* setstate(char*) @nogc nothrow;
    struct random_data
    {
        int* fptr;
        int* rptr;
        int* state;
        int rand_type;
        int rand_deg;
        int rand_sep;
        int* end_ptr;
    }
    int random_r(random_data*, int*) @nogc nothrow;
    int srandom_r(uint, random_data*) @nogc nothrow;
    int initstate_r(uint, char*, c_ulong, random_data*) @nogc nothrow;
    int setstate_r(char*, random_data*) @nogc nothrow;
    int rand() @nogc nothrow;
    void srand(uint) @nogc nothrow;
    int rand_r(uint*) @nogc nothrow;
    double drand48() @nogc nothrow;
    double erand48(ushort*) @nogc nothrow;
    c_long lrand48() @nogc nothrow;
    c_long nrand48(ushort*) @nogc nothrow;
    c_long mrand48() @nogc nothrow;
    c_long jrand48(ushort*) @nogc nothrow;
    void srand48(c_long) @nogc nothrow;
    ushort* seed48(ushort*) @nogc nothrow;
    void lcong48(ushort*) @nogc nothrow;
    struct drand48_data
    {
        ushort[3] __x;
        ushort[3] __old_x;
        ushort __c;
        ushort __init;
        ulong __a;
    }
    int drand48_r(drand48_data*, double*) @nogc nothrow;
    int erand48_r(ushort*, drand48_data*, double*) @nogc nothrow;
    int lrand48_r(drand48_data*, c_long*) @nogc nothrow;
    int nrand48_r(ushort*, drand48_data*, c_long*) @nogc nothrow;
    int mrand48_r(drand48_data*, c_long*) @nogc nothrow;
    int jrand48_r(ushort*, drand48_data*, c_long*) @nogc nothrow;
    int srand48_r(c_long, drand48_data*) @nogc nothrow;
    int seed48_r(ushort*, drand48_data*) @nogc nothrow;
    int lcong48_r(ushort*, drand48_data*) @nogc nothrow;
    void* malloc(c_ulong) @nogc nothrow;
    void* calloc(c_ulong, c_ulong) @nogc nothrow;
    void* realloc(void*, c_ulong) @nogc nothrow;
    void free(void*) @nogc nothrow;
    void* valloc(c_ulong) @nogc nothrow;
    int posix_memalign(void**, c_ulong, c_ulong) @nogc nothrow;
    void* aligned_alloc(c_ulong, c_ulong) @nogc nothrow;
    void abort() @nogc nothrow;
    int atexit(void function()) @nogc nothrow;
    int at_quick_exit(void function()) @nogc nothrow;
    int on_exit(void function(int, void*), void*) @nogc nothrow;
    void exit(int) @nogc nothrow;
    void quick_exit(int) @nogc nothrow;
    void _Exit(int) @nogc nothrow;
    char* getenv(const(char)*) @nogc nothrow;
    int putenv(char*) @nogc nothrow;
    int setenv(const(char)*, const(char)*, int) @nogc nothrow;
    int unsetenv(const(char)*) @nogc nothrow;
    int clearenv() @nogc nothrow;
    char* mktemp(char*) @nogc nothrow;
    int mkstemp(char*) @nogc nothrow;
    int mkstemps(char*, int) @nogc nothrow;
    char* mkdtemp(char*) @nogc nothrow;
    int system(const(char)*) @nogc nothrow;
    char* realpath(const(char)*, char*) @nogc nothrow;
    alias __compar_fn_t = int function(const(void)*, const(void)*);
    void* bsearch(const(void)*, const(void)*, c_ulong, c_ulong, int function(const(void)*, const(void)*)) @nogc nothrow;
    void qsort(void*, c_ulong, c_ulong, int function(const(void)*, const(void)*)) @nogc nothrow;
    int abs(int) @nogc nothrow;
    c_long labs(c_long) @nogc nothrow;
    long llabs(long) @nogc nothrow;
    div_t div(int, int) @nogc nothrow;
    ldiv_t ldiv(c_long, c_long) @nogc nothrow;
    lldiv_t lldiv(long, long) @nogc nothrow;
    char* ecvt(double, int, int*, int*) @nogc nothrow;
    char* fcvt(double, int, int*, int*) @nogc nothrow;
    char* gcvt(double, int, char*) @nogc nothrow;
    char* qecvt(real, int, int*, int*) @nogc nothrow;
    char* qfcvt(real, int, int*, int*) @nogc nothrow;
    char* qgcvt(real, int, char*) @nogc nothrow;
    int ecvt_r(double, int, int*, int*, char*, c_ulong) @nogc nothrow;
    int fcvt_r(double, int, int*, int*, char*, c_ulong) @nogc nothrow;
    int qecvt_r(real, int, int*, int*, char*, c_ulong) @nogc nothrow;
    int qfcvt_r(real, int, int*, int*, char*, c_ulong) @nogc nothrow;
    int mblen(const(char)*, c_ulong) @nogc nothrow;
    int mbtowc(int*, const(char)*, c_ulong) @nogc nothrow;
    int wctomb(char*, int) @nogc nothrow;
    c_ulong mbstowcs(int*, const(char)*, c_ulong) @nogc nothrow;
    c_ulong wcstombs(char*, const(int)*, c_ulong) @nogc nothrow;
    int rpmatch(const(char)*) @nogc nothrow;
    int getsubopt(char**, char**, char**) @nogc nothrow;
    int getloadavg(double*, int) @nogc nothrow;
    cudaError_enum cuMemcpyDtoH_v2(void*, ulong, c_ulong) @nogc nothrow;
    cudaError_enum cuMemcpyHtoD_v2(ulong, const(void)*, c_ulong) @nogc nothrow;
    cudaError_enum cuMemcpyPeer(ulong, CUctx_st*, ulong, CUctx_st*, c_ulong) @nogc nothrow;
    cudaError_enum cuMemcpy(ulong, ulong, c_ulong) @nogc nothrow;
    cudaError_enum cuMemHostUnregister(void*) @nogc nothrow;
    cudaError_enum cuMemHostRegister_v2(void*, c_ulong, uint) @nogc nothrow;
    cudaError_enum cuIpcCloseMemHandle(ulong) @nogc nothrow;
    cudaError_enum cuIpcOpenMemHandle(ulong*, CUipcMemHandle_st, uint) @nogc nothrow;
    cudaError_enum cuIpcGetMemHandle(CUipcMemHandle_st*, ulong) @nogc nothrow;
    cudaError_enum cuIpcOpenEventHandle(CUevent_st**, CUipcEventHandle_st) @nogc nothrow;
    cudaError_enum cuIpcGetEventHandle(CUipcEventHandle_st*, CUevent_st*) @nogc nothrow;
    cudaError_enum cuDeviceGetPCIBusId(char*, int, int) @nogc nothrow;
    cudaError_enum cuDeviceGetByPCIBusId(int*, const(char)*) @nogc nothrow;
    alias _Float32 = float;
    cudaError_enum cuMemAllocManaged(ulong*, c_ulong, uint) @nogc nothrow;
    alias _Float64 = double;
    cudaError_enum cuMemHostGetFlags(uint*, void*) @nogc nothrow;
    alias _Float32x = double;
    cudaError_enum cuMemHostGetDevicePointer_v2(ulong*, void*, uint) @nogc nothrow;
    cudaError_enum cuMemHostAlloc(void**, c_ulong, uint) @nogc nothrow;
    alias _Float64x = real;
    cudaError_enum cuMemFreeHost(void*) @nogc nothrow;
    cudaError_enum cuMemAllocHost_v2(void**, c_ulong) @nogc nothrow;
    cudaError_enum cuMemGetAddressRange_v2(ulong*, c_ulong*, ulong) @nogc nothrow;
    cudaError_enum cuMemFree_v2(ulong) @nogc nothrow;
    cudaError_enum cuMemAllocPitch_v2(ulong*, c_ulong*, c_ulong, c_ulong, uint) @nogc nothrow;
    cudaError_enum cuMemAlloc_v2(ulong*, c_ulong) @nogc nothrow;
    cudaError_enum cuMemGetInfo_v2(c_ulong*, c_ulong*) @nogc nothrow;
    cudaError_enum cuLinkDestroy(CUlinkState_st*) @nogc nothrow;
    cudaError_enum cuLinkComplete(CUlinkState_st*, void**, c_ulong*) @nogc nothrow;
    cudaError_enum cuLinkAddFile_v2(CUlinkState_st*, CUjitInputType_enum, const(char)*, uint, CUjit_option_enum*, void**) @nogc nothrow;
    cudaError_enum cuLinkAddData_v2(CUlinkState_st*, CUjitInputType_enum, void*, c_ulong, const(char)*, uint, CUjit_option_enum*, void**) @nogc nothrow;
    cudaError_enum cuLinkCreate_v2(uint, CUjit_option_enum*, void**, CUlinkState_st**) @nogc nothrow;
    struct __pthread_rwlock_arch_t
    {
        uint __readers;
        uint __writers;
        uint __wrphase_futex;
        uint __writers_futex;
        uint __pad3;
        uint __pad4;
        int __cur_writer;
        int __shared;
        byte __rwelision;
        ubyte[7] __pad1;
        c_ulong __pad2;
        uint __flags;
    }
    cudaError_enum cuModuleGetSurfRef(CUsurfref_st**, CUmod_st*, const(char)*) @nogc nothrow;
    alias pthread_t = c_ulong;
    union pthread_mutexattr_t
    {
        char[4] __size;
        int __align;
    }
    union pthread_condattr_t
    {
        char[4] __size;
        int __align;
    }
    alias pthread_key_t = uint;
    alias pthread_once_t = int;
    union pthread_attr_t
    {
        char[56] __size;
        c_long __align;
    }
    cudaError_enum cuModuleGetTexRef(CUtexref_st**, CUmod_st*, const(char)*) @nogc nothrow;
    union pthread_mutex_t
    {
        __pthread_mutex_s __data;
        char[40] __size;
        c_long __align;
    }
    union pthread_cond_t
    {
        __pthread_cond_s __data;
        char[48] __size;
        long __align;
    }
    union pthread_rwlock_t
    {
        __pthread_rwlock_arch_t __data;
        char[56] __size;
        c_long __align;
    }
    union pthread_rwlockattr_t
    {
        char[8] __size;
        c_long __align;
    }
    alias pthread_spinlock_t = int;
    union pthread_barrier_t
    {
        char[32] __size;
        c_long __align;
    }
    union pthread_barrierattr_t
    {
        char[4] __size;
        int __align;
    }
    cudaError_enum cuModuleGetGlobal_v2(ulong*, c_ulong*, CUmod_st*, const(char)*) @nogc nothrow;
    cudaError_enum cuModuleGetFunction(CUfunc_st**, CUmod_st*, const(char)*) @nogc nothrow;
    alias int8_t = byte;
    alias int16_t = short;
    alias int32_t = int;
    alias int64_t = c_long;
    alias uint8_t = ubyte;
    alias uint16_t = ushort;
    alias uint32_t = uint;
    alias uint64_t = ulong;
    cudaError_enum cuModuleUnload(CUmod_st*) @nogc nothrow;
    cudaError_enum cuModuleLoadFatBinary(CUmod_st**, const(void)*) @nogc nothrow;
    cudaError_enum cuModuleLoadDataEx(CUmod_st**, const(void)*, uint, CUjit_option_enum*, void**) @nogc nothrow;
    cudaError_enum cuModuleLoadData(CUmod_st**, const(void)*) @nogc nothrow;
    alias __pthread_list_t = __pthread_internal_list;
    struct __pthread_internal_list
    {
        __pthread_internal_list* __prev;
        __pthread_internal_list* __next;
    }
    cudaError_enum cuModuleLoad(CUmod_st**, const(char)*) @nogc nothrow;
    struct __pthread_mutex_s
    {
        int __lock;
        uint __count;
        int __owner;
        uint __nusers;
        int __kind;
        short __spins;
        short __elision;
        __pthread_internal_list __list;
    }
    cudaError_enum cuCtxDetach(CUctx_st*) @nogc nothrow;
    struct __pthread_cond_s
    {
        static union _Anonymous_0
        {
            ulong __wseq;
            static struct _Anonymous_1
            {
                uint __low;
                uint __high;
            }
            _Anonymous_1 __wseq32;
        }
        _Anonymous_0 _anonymous_2;
        auto __wseq() @property @nogc pure nothrow { return _anonymous_2.__wseq; }
        void __wseq(_T_)(auto ref _T_ val) @property @nogc pure nothrow { _anonymous_2.__wseq = val; }
        auto __wseq32() @property @nogc pure nothrow { return _anonymous_2.__wseq32; }
        void __wseq32(_T_)(auto ref _T_ val) @property @nogc pure nothrow { _anonymous_2.__wseq32 = val; }
        static union _Anonymous_3
        {
            ulong __g1_start;
            static struct _Anonymous_4
            {
                uint __low;
                uint __high;
            }
            _Anonymous_4 __g1_start32;
        }
        _Anonymous_3 _anonymous_5;
        auto __g1_start() @property @nogc pure nothrow { return _anonymous_5.__g1_start; }
        void __g1_start(_T_)(auto ref _T_ val) @property @nogc pure nothrow { _anonymous_5.__g1_start = val; }
        auto __g1_start32() @property @nogc pure nothrow { return _anonymous_5.__g1_start32; }
        void __g1_start32(_T_)(auto ref _T_ val) @property @nogc pure nothrow { _anonymous_5.__g1_start32 = val; }
        uint[2] __g_refs;
        uint[2] __g_size;
        uint __g1_orig_size;
        uint __wrefs;
        uint[2] __g_signals;
    }
    alias __u_char = ubyte;
    alias __u_short = ushort;
    alias __u_int = uint;
    alias __u_long = c_ulong;
    alias __int8_t = byte;
    alias __uint8_t = ubyte;
    alias __int16_t = short;
    alias __uint16_t = ushort;
    alias __int32_t = int;
    alias __uint32_t = uint;
    alias __int64_t = c_long;
    alias __uint64_t = c_ulong;
    alias __quad_t = c_long;
    alias __u_quad_t = c_ulong;
    alias __intmax_t = c_long;
    alias __uintmax_t = c_ulong;
    cudaError_enum cuCtxAttach(CUctx_st**, uint) @nogc nothrow;
    cudaError_enum cuCtxGetStreamPriorityRange(int*, int*) @nogc nothrow;
    cudaError_enum cuCtxGetApiVersion(CUctx_st*, uint*) @nogc nothrow;
    cudaError_enum cuCtxSetSharedMemConfig(CUsharedconfig_enum) @nogc nothrow;
    cudaError_enum cuCtxGetSharedMemConfig(CUsharedconfig_enum*) @nogc nothrow;
    cudaError_enum cuCtxSetCacheConfig(CUfunc_cache_enum) @nogc nothrow;
    alias __dev_t = c_ulong;
    alias __uid_t = uint;
    alias __gid_t = uint;
    alias __ino_t = c_ulong;
    alias __ino64_t = c_ulong;
    alias __mode_t = uint;
    alias __nlink_t = c_ulong;
    alias __off_t = c_long;
    alias __off64_t = c_long;
    alias __pid_t = int;
    struct __fsid_t
    {
        int[2] __val;
    }
    alias __clock_t = c_long;
    alias __rlim_t = c_ulong;
    alias __rlim64_t = c_ulong;
    alias __id_t = uint;
    alias __time_t = c_long;
    alias __useconds_t = uint;
    alias __suseconds_t = c_long;
    alias __daddr_t = int;
    alias __key_t = int;
    alias __clockid_t = int;
    alias __timer_t = void*;
    alias __blksize_t = c_long;
    alias __blkcnt_t = c_long;
    alias __blkcnt64_t = c_long;
    alias __fsblkcnt_t = c_ulong;
    alias __fsblkcnt64_t = c_ulong;
    alias __fsfilcnt_t = c_ulong;
    alias __fsfilcnt64_t = c_ulong;
    alias __fsword_t = c_long;
    alias __ssize_t = c_long;
    alias __syscall_slong_t = c_long;
    alias __syscall_ulong_t = c_ulong;
    alias __loff_t = c_long;
    alias __caddr_t = char*;
    alias __intptr_t = c_long;
    alias __socklen_t = uint;
    alias __sig_atomic_t = int;
    cudaError_enum cuCtxGetCacheConfig(CUfunc_cache_enum*) @nogc nothrow;
    struct __sigset_t
    {
        c_ulong[16] __val;
    }
    cudaError_enum cuCtxGetLimit(c_ulong*, CUlimit_enum) @nogc nothrow;
    alias clock_t = c_long;
    alias clockid_t = int;
    cudaError_enum cuCtxSetLimit(CUlimit_enum, c_ulong) @nogc nothrow;
    alias sigset_t = __sigset_t;
    struct timespec
    {
        c_long tv_sec;
        c_long tv_nsec;
    }
    cudaError_enum cuCtxSynchronize() @nogc nothrow;
    struct timeval
    {
        c_long tv_sec;
        c_long tv_usec;
    }
    alias time_t = c_long;
    cudaError_enum cuCtxGetFlags(uint*) @nogc nothrow;
    alias timer_t = void*;
    cudaError_enum cuCtxGetDevice(int*) @nogc nothrow;
    cudaError_enum cuCtxGetCurrent(CUctx_st**) @nogc nothrow;
    cudaError_enum cuCtxSetCurrent(CUctx_st*) @nogc nothrow;
    cudaError_enum cuCtxPopCurrent_v2(CUctx_st**) @nogc nothrow;
    cudaError_enum cuCtxPushCurrent_v2(CUctx_st*) @nogc nothrow;
    cudaError_enum cuCtxDestroy_v2(CUctx_st*) @nogc nothrow;
    cudaError_enum cuCtxCreate_v2(CUctx_st**, uint, int) @nogc nothrow;
    cudaError_enum cuDevicePrimaryCtxReset(int) @nogc nothrow;
    cudaError_enum cuDevicePrimaryCtxGetState(int, uint*, int*) @nogc nothrow;
    cudaError_enum cuDevicePrimaryCtxSetFlags(int, uint) @nogc nothrow;
    cudaError_enum cuDevicePrimaryCtxRelease(int) @nogc nothrow;
    cudaError_enum cuDevicePrimaryCtxRetain(CUctx_st**, int) @nogc nothrow;
    cudaError_enum cuDeviceComputeCapability(int*, int*, int) @nogc nothrow;
    cudaError_enum cuDeviceGetProperties(CUdevprop_st*, int) @nogc nothrow;
    static ushort __uint16_identity(ushort) @nogc nothrow;
    static uint __uint32_identity(uint) @nogc nothrow;
    static c_ulong __uint64_identity(c_ulong) @nogc nothrow;
    cudaError_enum cuDeviceGetNvSciSyncAttributes(void*, int, int) @nogc nothrow;
    cudaError_enum cuDeviceGetAttribute(int*, CUdevice_attribute_enum, int) @nogc nothrow;
    cudaError_enum cuDeviceTotalMem_v2(c_ulong*, int) @nogc nothrow;
    cudaError_enum cuDeviceGetUuid(CUuuid_st*, int) @nogc nothrow;
    alias idtype_t = _Anonymous_6;
    enum _Anonymous_6
    {
        P_ALL = 0,
        P_PID = 1,
        P_PGID = 2,
    }
    enum P_ALL = _Anonymous_6.P_ALL;
    enum P_PID = _Anonymous_6.P_PID;
    enum P_PGID = _Anonymous_6.P_PGID;
    cudaError_enum cuDeviceGetName(char*, int, int) @nogc nothrow;
    cudaError_enum cuDeviceGetCount(int*) @nogc nothrow;
    cudaError_enum cuDeviceGet(int*, int) @nogc nothrow;
    cudaError_enum cuDriverGetVersion(int*) @nogc nothrow;
    cudaError_enum cuInit(uint) @nogc nothrow;
    cudaError_enum cuGetErrorName(cudaError_enum, const(char)**) @nogc nothrow;
    cudaError_enum cuGetErrorString(cudaError_enum, const(char)**) @nogc nothrow;
    enum CUgraphExecUpdateResult_enum
    {
        CU_GRAPH_EXEC_UPDATE_SUCCESS = 0,
        CU_GRAPH_EXEC_UPDATE_ERROR = 1,
        CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED = 2,
        CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED = 3,
        CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED = 4,
        CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED = 5,
        CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED = 6,
    }
    enum CU_GRAPH_EXEC_UPDATE_SUCCESS = CUgraphExecUpdateResult_enum.CU_GRAPH_EXEC_UPDATE_SUCCESS;
    enum CU_GRAPH_EXEC_UPDATE_ERROR = CUgraphExecUpdateResult_enum.CU_GRAPH_EXEC_UPDATE_ERROR;
    enum CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED = CUgraphExecUpdateResult_enum.CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED;
    enum CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED = CUgraphExecUpdateResult_enum.CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED;
    enum CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED = CUgraphExecUpdateResult_enum.CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED;
    enum CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED = CUgraphExecUpdateResult_enum.CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED;
    enum CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED = CUgraphExecUpdateResult_enum.CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED;
    alias CUgraphExecUpdateResult = CUgraphExecUpdateResult_enum;
    struct CUmemAccessDesc_st
    {
        CUmemLocation_st location;
        CUmemAccess_flags_enum flags;
    }
    alias CUmemAccessDesc = CUmemAccessDesc_st;
    struct CUmemAllocationProp_st
    {
        CUmemAllocationType_enum type;
        CUmemAllocationHandleType_enum requestedHandleTypes;
        CUmemLocation_st location;
        void* win32HandleMetaData;
        ulong reserved;
    }
    alias CUmemAllocationProp = CUmemAllocationProp_st;
    struct CUmemLocation_st
    {
        CUmemLocationType_enum type;
        int id;
    }
    alias CUmemLocation = CUmemLocation_st;
    enum CUmemAllocationGranularity_flags_enum
    {
        CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0,
        CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 1,
    }
    enum CU_MEM_ALLOC_GRANULARITY_MINIMUM = CUmemAllocationGranularity_flags_enum.CU_MEM_ALLOC_GRANULARITY_MINIMUM;
    enum CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = CUmemAllocationGranularity_flags_enum.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED;
    alias CUmemAllocationGranularity_flags = CUmemAllocationGranularity_flags_enum;
    enum CUmemAllocationType_enum
    {
        CU_MEM_ALLOCATION_TYPE_INVALID = 0,
        CU_MEM_ALLOCATION_TYPE_PINNED = 1,
        CU_MEM_ALLOCATION_TYPE_MAX = -1,
    }
    enum CU_MEM_ALLOCATION_TYPE_INVALID = CUmemAllocationType_enum.CU_MEM_ALLOCATION_TYPE_INVALID;
    enum CU_MEM_ALLOCATION_TYPE_PINNED = CUmemAllocationType_enum.CU_MEM_ALLOCATION_TYPE_PINNED;
    enum CU_MEM_ALLOCATION_TYPE_MAX = CUmemAllocationType_enum.CU_MEM_ALLOCATION_TYPE_MAX;
    alias CUmemAllocationType = CUmemAllocationType_enum;
    enum CUmemLocationType_enum
    {
        CU_MEM_LOCATION_TYPE_INVALID = 0,
        CU_MEM_LOCATION_TYPE_DEVICE = 1,
        CU_MEM_LOCATION_TYPE_MAX = -1,
    }
    enum CU_MEM_LOCATION_TYPE_INVALID = CUmemLocationType_enum.CU_MEM_LOCATION_TYPE_INVALID;
    enum CU_MEM_LOCATION_TYPE_DEVICE = CUmemLocationType_enum.CU_MEM_LOCATION_TYPE_DEVICE;
    enum CU_MEM_LOCATION_TYPE_MAX = CUmemLocationType_enum.CU_MEM_LOCATION_TYPE_MAX;
    alias CUmemLocationType = CUmemLocationType_enum;
    enum CUmemAccess_flags_enum
    {
        CU_MEM_ACCESS_FLAGS_PROT_NONE = 0,
        CU_MEM_ACCESS_FLAGS_PROT_READ = 1,
        CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3,
        CU_MEM_ACCESS_FLAGS_PROT_MAX = -1,
    }
    enum CU_MEM_ACCESS_FLAGS_PROT_NONE = CUmemAccess_flags_enum.CU_MEM_ACCESS_FLAGS_PROT_NONE;
    enum CU_MEM_ACCESS_FLAGS_PROT_READ = CUmemAccess_flags_enum.CU_MEM_ACCESS_FLAGS_PROT_READ;
    enum CU_MEM_ACCESS_FLAGS_PROT_READWRITE = CUmemAccess_flags_enum.CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    enum CU_MEM_ACCESS_FLAGS_PROT_MAX = CUmemAccess_flags_enum.CU_MEM_ACCESS_FLAGS_PROT_MAX;
    alias CUmemAccess_flags = CUmemAccess_flags_enum;
    enum CUmemAllocationHandleType_enum
    {
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1,
        CU_MEM_HANDLE_TYPE_WIN32 = 2,
        CU_MEM_HANDLE_TYPE_WIN32_KMT = 4,
        CU_MEM_HANDLE_TYPE_MAX = -1,
    }
    enum CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = CUmemAllocationHandleType_enum.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    enum CU_MEM_HANDLE_TYPE_WIN32 = CUmemAllocationHandleType_enum.CU_MEM_HANDLE_TYPE_WIN32;
    enum CU_MEM_HANDLE_TYPE_WIN32_KMT = CUmemAllocationHandleType_enum.CU_MEM_HANDLE_TYPE_WIN32_KMT;
    enum CU_MEM_HANDLE_TYPE_MAX = CUmemAllocationHandleType_enum.CU_MEM_HANDLE_TYPE_MAX;
    alias CUmemAllocationHandleType = CUmemAllocationHandleType_enum;
    alias CUmemGenericAllocationHandle = ulong;
    struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st
    {
        static struct _Anonymous_7
        {
            static struct _Anonymous_8
            {
                ulong value;
            }
            _Anonymous_8 fence;
            static union _Anonymous_9
            {
                void* fence;
                ulong reserved;
            }
            _Anonymous_9 nvSciSync;
            static struct _Anonymous_10
            {
                ulong key;
                uint timeoutMs;
            }
            _Anonymous_10 keyedMutex;
            uint[10] reserved;
        }
        _Anonymous_7 params;
        uint flags;
        uint[16] reserved;
    }
    alias CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS = CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st;
    struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st
    {
        static struct _Anonymous_11
        {
            static struct _Anonymous_12
            {
                ulong value;
            }
            _Anonymous_12 fence;
            static union _Anonymous_13
            {
                void* fence;
                ulong reserved;
            }
            _Anonymous_13 nvSciSync;
            static struct _Anonymous_14
            {
                ulong key;
            }
            _Anonymous_14 keyedMutex;
            uint[12] reserved;
        }
        _Anonymous_11 params;
        uint flags;
        uint[16] reserved;
    }
    alias CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS = CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st;
    struct CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st
    {
        CUexternalSemaphoreHandleType_enum type;
        static union _Anonymous_15
        {
            int fd;
            static struct _Anonymous_16
            {
                void* handle;
                const(void)* name;
            }
            _Anonymous_16 win32;
            const(void)* nvSciSyncObj;
        }
        _Anonymous_15 handle;
        uint flags;
        uint[16] reserved;
    }
    alias CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC = CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st;
    enum CUexternalSemaphoreHandleType_enum
    {
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD = 1,
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32 = 2,
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3,
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE = 4,
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE = 5,
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC = 6,
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX = 7,
        CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT = 8,
    }
    enum CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD = CUexternalSemaphoreHandleType_enum.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD;
    enum CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32 = CUexternalSemaphoreHandleType_enum.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32;
    enum CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT = CUexternalSemaphoreHandleType_enum.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT;
    enum CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE = CUexternalSemaphoreHandleType_enum.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE;
    enum CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE = CUexternalSemaphoreHandleType_enum.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE;
    enum CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC = CUexternalSemaphoreHandleType_enum.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC;
    enum CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX = CUexternalSemaphoreHandleType_enum.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX;
    enum CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT = CUexternalSemaphoreHandleType_enum.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT;
    alias CUexternalSemaphoreHandleType = CUexternalSemaphoreHandleType_enum;
    struct CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st
    {
        ulong offset;
        CUDA_ARRAY3D_DESCRIPTOR_st arrayDesc;
        uint numLevels;
        uint[16] reserved;
    }
    alias CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC = CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st;
    struct CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st
    {
        ulong offset;
        ulong size;
        uint flags;
        uint[16] reserved;
    }
    alias CUDA_EXTERNAL_MEMORY_BUFFER_DESC = CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st;
    struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st
    {
        CUexternalMemoryHandleType_enum type;
        static union _Anonymous_17
        {
            int fd;
            static struct _Anonymous_18
            {
                void* handle;
                const(void)* name;
            }
            _Anonymous_18 win32;
            const(void)* nvSciBufObject;
        }
        _Anonymous_17 handle;
        ulong size;
        uint flags;
        uint[16] reserved;
    }
    alias CUDA_EXTERNAL_MEMORY_HANDLE_DESC = CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st;
    enum CUexternalMemoryHandleType_enum
    {
        CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = 1,
        CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 = 2,
        CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3,
        CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP = 4,
        CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE = 5,
        CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE = 6,
        CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT = 7,
        CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF = 8,
    }
    enum CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = CUexternalMemoryHandleType_enum.CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
    enum CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 = CUexternalMemoryHandleType_enum.CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32;
    enum CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT = CUexternalMemoryHandleType_enum.CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT;
    enum CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP = CUexternalMemoryHandleType_enum.CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP;
    enum CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE = CUexternalMemoryHandleType_enum.CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
    enum CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE = CUexternalMemoryHandleType_enum.CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE;
    enum CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT = CUexternalMemoryHandleType_enum.CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT;
    enum CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF = CUexternalMemoryHandleType_enum.CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF;
    alias CUexternalMemoryHandleType = CUexternalMemoryHandleType_enum;
    struct CUDA_LAUNCH_PARAMS_st
    {
        CUfunc_st* function_;
        uint gridDimX;
        uint gridDimY;
        uint gridDimZ;
        uint blockDimX;
        uint blockDimY;
        uint blockDimZ;
        uint sharedMemBytes;
        CUstream_st* hStream;
        void** kernelParams;
    }
    alias CUDA_LAUNCH_PARAMS = CUDA_LAUNCH_PARAMS_st;
    struct CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st
    {
        ulong p2pToken;
        uint vaSpaceToken;
    }
    alias CUDA_POINTER_ATTRIBUTE_P2P_TOKENS = CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st;
    struct CUDA_RESOURCE_VIEW_DESC_st
    {
        CUresourceViewFormat_enum format;
        c_ulong width;
        c_ulong height;
        c_ulong depth;
        uint firstMipmapLevel;
        uint lastMipmapLevel;
        uint firstLayer;
        uint lastLayer;
        uint[16] reserved;
    }
    alias CUDA_RESOURCE_VIEW_DESC = CUDA_RESOURCE_VIEW_DESC_st;
    enum CUresourceViewFormat_enum
    {
        CU_RES_VIEW_FORMAT_NONE = 0,
        CU_RES_VIEW_FORMAT_UINT_1X8 = 1,
        CU_RES_VIEW_FORMAT_UINT_2X8 = 2,
        CU_RES_VIEW_FORMAT_UINT_4X8 = 3,
        CU_RES_VIEW_FORMAT_SINT_1X8 = 4,
        CU_RES_VIEW_FORMAT_SINT_2X8 = 5,
        CU_RES_VIEW_FORMAT_SINT_4X8 = 6,
        CU_RES_VIEW_FORMAT_UINT_1X16 = 7,
        CU_RES_VIEW_FORMAT_UINT_2X16 = 8,
        CU_RES_VIEW_FORMAT_UINT_4X16 = 9,
        CU_RES_VIEW_FORMAT_SINT_1X16 = 10,
        CU_RES_VIEW_FORMAT_SINT_2X16 = 11,
        CU_RES_VIEW_FORMAT_SINT_4X16 = 12,
        CU_RES_VIEW_FORMAT_UINT_1X32 = 13,
        CU_RES_VIEW_FORMAT_UINT_2X32 = 14,
        CU_RES_VIEW_FORMAT_UINT_4X32 = 15,
        CU_RES_VIEW_FORMAT_SINT_1X32 = 16,
        CU_RES_VIEW_FORMAT_SINT_2X32 = 17,
        CU_RES_VIEW_FORMAT_SINT_4X32 = 18,
        CU_RES_VIEW_FORMAT_FLOAT_1X16 = 19,
        CU_RES_VIEW_FORMAT_FLOAT_2X16 = 20,
        CU_RES_VIEW_FORMAT_FLOAT_4X16 = 21,
        CU_RES_VIEW_FORMAT_FLOAT_1X32 = 22,
        CU_RES_VIEW_FORMAT_FLOAT_2X32 = 23,
        CU_RES_VIEW_FORMAT_FLOAT_4X32 = 24,
        CU_RES_VIEW_FORMAT_UNSIGNED_BC1 = 25,
        CU_RES_VIEW_FORMAT_UNSIGNED_BC2 = 26,
        CU_RES_VIEW_FORMAT_UNSIGNED_BC3 = 27,
        CU_RES_VIEW_FORMAT_UNSIGNED_BC4 = 28,
        CU_RES_VIEW_FORMAT_SIGNED_BC4 = 29,
        CU_RES_VIEW_FORMAT_UNSIGNED_BC5 = 30,
        CU_RES_VIEW_FORMAT_SIGNED_BC5 = 31,
        CU_RES_VIEW_FORMAT_UNSIGNED_BC6H = 32,
        CU_RES_VIEW_FORMAT_SIGNED_BC6H = 33,
        CU_RES_VIEW_FORMAT_UNSIGNED_BC7 = 34,
    }
    enum CU_RES_VIEW_FORMAT_NONE = CUresourceViewFormat_enum.CU_RES_VIEW_FORMAT_NONE;
    enum CU_RES_VIEW_FORMAT_UINT_1X8 = CUresourceViewFormat_enum.CU_RES_VIEW_FORMAT_UINT_1X8;
    enum CU_RES_VIEW_FORMAT_UINT_2X8 = CUresourceViewFormat_enum.CU_RES_VIEW_FORMAT_UINT_2X8;
    enum CU_RES_VIEW_FORMAT_UINT_4X8 = CUresourceViewFormat_enum.CU_RES_VIEW_FORMAT_UINT_4X8;
    enum CU_RES_VIEW_FORMAT_SINT_1X8 = CUresourceViewFormat_enum.CU_RES_VIEW_FORMAT_SINT_1X8;
    enum CU_RES_VIEW_FORMAT_SINT_2X8 = CUresourceViewFormat_enum.CU_RES_VIEW_FORMAT_SINT_2X8;
    enum CU_RES_VIEW_FORMAT_SINT_4X8 = CUresourceViewFormat_enum.CU_RES_VIEW_FORMAT_SINT_4X8;
    enum CU_RES_VIEW_FORMAT_UINT_1X16 = CUresourceViewFormat_enum.CU_RES_VIEW_FORMAT_UINT_1X16;
    enum CU_RES_VIEW_FORMAT_UINT_2X16 = CUresourceViewFormat_enum.CU_RES_VIEW_FORMAT_UINT_2X16;
    enum CU_RES_VIEW_FORMAT_UINT_4X16 = CUresourceViewFormat_enum.CU_RES_VIEW_FORMAT_UINT_4X16;
    enum CU_RES_VIEW_FORMAT_SINT_1X16 = CUresourceViewFormat_enum.CU_RES_VIEW_FORMAT_SINT_1X16;
    enum CU_RES_VIEW_FORMAT_SINT_2X16 = CUresourceViewFormat_enum.CU_RES_VIEW_FORMAT_SINT_2X16;
    enum CU_RES_VIEW_FORMAT_SINT_4X16 = CUresourceViewFormat_enum.CU_RES_VIEW_FORMAT_SINT_4X16;
    enum CU_RES_VIEW_FORMAT_UINT_1X32 = CUresourceViewFormat_enum.CU_RES_VIEW_FORMAT_UINT_1X32;
    enum CU_RES_VIEW_FORMAT_UINT_2X32 = CUresourceViewFormat_enum.CU_RES_VIEW_FORMAT_UINT_2X32;
    enum CU_RES_VIEW_FORMAT_UINT_4X32 = CUresourceViewFormat_enum.CU_RES_VIEW_FORMAT_UINT_4X32;
    enum CU_RES_VIEW_FORMAT_SINT_1X32 = CUresourceViewFormat_enum.CU_RES_VIEW_FORMAT_SINT_1X32;
    enum CU_RES_VIEW_FORMAT_SINT_2X32 = CUresourceViewFormat_enum.CU_RES_VIEW_FORMAT_SINT_2X32;
    enum CU_RES_VIEW_FORMAT_SINT_4X32 = CUresourceViewFormat_enum.CU_RES_VIEW_FORMAT_SINT_4X32;
    enum CU_RES_VIEW_FORMAT_FLOAT_1X16 = CUresourceViewFormat_enum.CU_RES_VIEW_FORMAT_FLOAT_1X16;
    enum CU_RES_VIEW_FORMAT_FLOAT_2X16 = CUresourceViewFormat_enum.CU_RES_VIEW_FORMAT_FLOAT_2X16;
    enum CU_RES_VIEW_FORMAT_FLOAT_4X16 = CUresourceViewFormat_enum.CU_RES_VIEW_FORMAT_FLOAT_4X16;
    enum CU_RES_VIEW_FORMAT_FLOAT_1X32 = CUresourceViewFormat_enum.CU_RES_VIEW_FORMAT_FLOAT_1X32;
    enum CU_RES_VIEW_FORMAT_FLOAT_2X32 = CUresourceViewFormat_enum.CU_RES_VIEW_FORMAT_FLOAT_2X32;
    enum CU_RES_VIEW_FORMAT_FLOAT_4X32 = CUresourceViewFormat_enum.CU_RES_VIEW_FORMAT_FLOAT_4X32;
    enum CU_RES_VIEW_FORMAT_UNSIGNED_BC1 = CUresourceViewFormat_enum.CU_RES_VIEW_FORMAT_UNSIGNED_BC1;
    enum CU_RES_VIEW_FORMAT_UNSIGNED_BC2 = CUresourceViewFormat_enum.CU_RES_VIEW_FORMAT_UNSIGNED_BC2;
    enum CU_RES_VIEW_FORMAT_UNSIGNED_BC3 = CUresourceViewFormat_enum.CU_RES_VIEW_FORMAT_UNSIGNED_BC3;
    enum CU_RES_VIEW_FORMAT_UNSIGNED_BC4 = CUresourceViewFormat_enum.CU_RES_VIEW_FORMAT_UNSIGNED_BC4;
    enum CU_RES_VIEW_FORMAT_SIGNED_BC4 = CUresourceViewFormat_enum.CU_RES_VIEW_FORMAT_SIGNED_BC4;
    enum CU_RES_VIEW_FORMAT_UNSIGNED_BC5 = CUresourceViewFormat_enum.CU_RES_VIEW_FORMAT_UNSIGNED_BC5;
    enum CU_RES_VIEW_FORMAT_SIGNED_BC5 = CUresourceViewFormat_enum.CU_RES_VIEW_FORMAT_SIGNED_BC5;
    enum CU_RES_VIEW_FORMAT_UNSIGNED_BC6H = CUresourceViewFormat_enum.CU_RES_VIEW_FORMAT_UNSIGNED_BC6H;
    enum CU_RES_VIEW_FORMAT_SIGNED_BC6H = CUresourceViewFormat_enum.CU_RES_VIEW_FORMAT_SIGNED_BC6H;
    enum CU_RES_VIEW_FORMAT_UNSIGNED_BC7 = CUresourceViewFormat_enum.CU_RES_VIEW_FORMAT_UNSIGNED_BC7;
    alias CUresourceViewFormat = CUresourceViewFormat_enum;
    struct CUDA_TEXTURE_DESC_st
    {
        CUaddress_mode_enum[3] addressMode;
        CUfilter_mode_enum filterMode;
        uint flags;
        uint maxAnisotropy;
        CUfilter_mode_enum mipmapFilterMode;
        float mipmapLevelBias;
        float minMipmapLevelClamp;
        float maxMipmapLevelClamp;
        float[4] borderColor;
        int[12] reserved;
    }
    alias CUDA_TEXTURE_DESC = CUDA_TEXTURE_DESC_st;
    struct CUDA_RESOURCE_DESC_st
    {
        CUresourcetype_enum resType;
        static union _Anonymous_19
        {
            static struct _Anonymous_20
            {
                CUarray_st* hArray;
            }
            _Anonymous_20 array;
            static struct _Anonymous_21
            {
                CUmipmappedArray_st* hMipmappedArray;
            }
            _Anonymous_21 mipmap;
            static struct _Anonymous_22
            {
                ulong devPtr;
                CUarray_format_enum format;
                uint numChannels;
                c_ulong sizeInBytes;
            }
            _Anonymous_22 linear;
            static struct _Anonymous_23
            {
                ulong devPtr;
                CUarray_format_enum format;
                uint numChannels;
                c_ulong width;
                c_ulong height;
                c_ulong pitchInBytes;
            }
            _Anonymous_23 pitch2D;
            static struct _Anonymous_24
            {
                int[32] reserved;
            }
            _Anonymous_24 reserved;
        }
        _Anonymous_19 res;
        uint flags;
    }
    alias CUDA_RESOURCE_DESC = CUDA_RESOURCE_DESC_st;
    struct CUDA_ARRAY3D_DESCRIPTOR_st
    {
        c_ulong Width;
        c_ulong Height;
        c_ulong Depth;
        CUarray_format_enum Format;
        uint NumChannels;
        uint Flags;
    }
    alias CUDA_ARRAY3D_DESCRIPTOR = CUDA_ARRAY3D_DESCRIPTOR_st;
    struct CUDA_ARRAY_DESCRIPTOR_st
    {
        c_ulong Width;
        c_ulong Height;
        CUarray_format_enum Format;
        uint NumChannels;
    }
    alias CUDA_ARRAY_DESCRIPTOR = CUDA_ARRAY_DESCRIPTOR_st;
    struct CUDA_MEMCPY3D_PEER_st
    {
        c_ulong srcXInBytes;
        c_ulong srcY;
        c_ulong srcZ;
        c_ulong srcLOD;
        CUmemorytype_enum srcMemoryType;
        const(void)* srcHost;
        ulong srcDevice;
        CUarray_st* srcArray;
        CUctx_st* srcContext;
        c_ulong srcPitch;
        c_ulong srcHeight;
        c_ulong dstXInBytes;
        c_ulong dstY;
        c_ulong dstZ;
        c_ulong dstLOD;
        CUmemorytype_enum dstMemoryType;
        void* dstHost;
        ulong dstDevice;
        CUarray_st* dstArray;
        CUctx_st* dstContext;
        c_ulong dstPitch;
        c_ulong dstHeight;
        c_ulong WidthInBytes;
        c_ulong Height;
        c_ulong Depth;
    }
    alias CUDA_MEMCPY3D_PEER = CUDA_MEMCPY3D_PEER_st;
    struct CUDA_MEMCPY3D_st
    {
        c_ulong srcXInBytes;
        c_ulong srcY;
        c_ulong srcZ;
        c_ulong srcLOD;
        CUmemorytype_enum srcMemoryType;
        const(void)* srcHost;
        ulong srcDevice;
        CUarray_st* srcArray;
        void* reserved0;
        c_ulong srcPitch;
        c_ulong srcHeight;
        c_ulong dstXInBytes;
        c_ulong dstY;
        c_ulong dstZ;
        c_ulong dstLOD;
        CUmemorytype_enum dstMemoryType;
        void* dstHost;
        ulong dstDevice;
        CUarray_st* dstArray;
        void* reserved1;
        c_ulong dstPitch;
        c_ulong dstHeight;
        c_ulong WidthInBytes;
        c_ulong Height;
        c_ulong Depth;
    }
    alias CUDA_MEMCPY3D = CUDA_MEMCPY3D_st;
    struct CUDA_MEMCPY2D_st
    {
        c_ulong srcXInBytes;
        c_ulong srcY;
        CUmemorytype_enum srcMemoryType;
        const(void)* srcHost;
        ulong srcDevice;
        CUarray_st* srcArray;
        c_ulong srcPitch;
        c_ulong dstXInBytes;
        c_ulong dstY;
        CUmemorytype_enum dstMemoryType;
        void* dstHost;
        ulong dstDevice;
        CUarray_st* dstArray;
        c_ulong dstPitch;
        c_ulong WidthInBytes;
        c_ulong Height;
    }
    alias CUDA_MEMCPY2D = CUDA_MEMCPY2D_st;
    alias CUoccupancyB2DSize = c_ulong function(int);
    alias CUstreamCallback = void function(CUstream_st*, cudaError_enum, void*);
    enum CUdevice_P2PAttribute_enum
    {
        CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = 1,
        CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = 2,
        CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = 3,
        CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED = 4,
        CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED = 4,
    }
    enum CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = CUdevice_P2PAttribute_enum.CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK;
    enum CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = CUdevice_P2PAttribute_enum.CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED;
    enum CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = CUdevice_P2PAttribute_enum.CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED;
    enum CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED = CUdevice_P2PAttribute_enum.CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED;
    enum CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED = CUdevice_P2PAttribute_enum.CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED;
    alias CUdevice_P2PAttribute = CUdevice_P2PAttribute_enum;
    enum cudaError_enum
    {
        CUDA_SUCCESS = 0,
        CUDA_ERROR_INVALID_VALUE = 1,
        CUDA_ERROR_OUT_OF_MEMORY = 2,
        CUDA_ERROR_NOT_INITIALIZED = 3,
        CUDA_ERROR_DEINITIALIZED = 4,
        CUDA_ERROR_PROFILER_DISABLED = 5,
        CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6,
        CUDA_ERROR_PROFILER_ALREADY_STARTED = 7,
        CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8,
        CUDA_ERROR_NO_DEVICE = 100,
        CUDA_ERROR_INVALID_DEVICE = 101,
        CUDA_ERROR_INVALID_IMAGE = 200,
        CUDA_ERROR_INVALID_CONTEXT = 201,
        CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
        CUDA_ERROR_MAP_FAILED = 205,
        CUDA_ERROR_UNMAP_FAILED = 206,
        CUDA_ERROR_ARRAY_IS_MAPPED = 207,
        CUDA_ERROR_ALREADY_MAPPED = 208,
        CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
        CUDA_ERROR_ALREADY_ACQUIRED = 210,
        CUDA_ERROR_NOT_MAPPED = 211,
        CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,
        CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,
        CUDA_ERROR_ECC_UNCORRECTABLE = 214,
        CUDA_ERROR_UNSUPPORTED_LIMIT = 215,
        CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,
        CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,
        CUDA_ERROR_INVALID_PTX = 218,
        CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,
        CUDA_ERROR_NVLINK_UNCORRECTABLE = 220,
        CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221,
        CUDA_ERROR_INVALID_SOURCE = 300,
        CUDA_ERROR_FILE_NOT_FOUND = 301,
        CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
        CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
        CUDA_ERROR_OPERATING_SYSTEM = 304,
        CUDA_ERROR_INVALID_HANDLE = 400,
        CUDA_ERROR_ILLEGAL_STATE = 401,
        CUDA_ERROR_NOT_FOUND = 500,
        CUDA_ERROR_NOT_READY = 600,
        CUDA_ERROR_ILLEGAL_ADDRESS = 700,
        CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
        CUDA_ERROR_LAUNCH_TIMEOUT = 702,
        CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
        CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
        CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,
        CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
        CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
        CUDA_ERROR_ASSERT = 710,
        CUDA_ERROR_TOO_MANY_PEERS = 711,
        CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
        CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
        CUDA_ERROR_HARDWARE_STACK_ERROR = 714,
        CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,
        CUDA_ERROR_MISALIGNED_ADDRESS = 716,
        CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,
        CUDA_ERROR_INVALID_PC = 718,
        CUDA_ERROR_LAUNCH_FAILED = 719,
        CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720,
        CUDA_ERROR_NOT_PERMITTED = 800,
        CUDA_ERROR_NOT_SUPPORTED = 801,
        CUDA_ERROR_SYSTEM_NOT_READY = 802,
        CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803,
        CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,
        CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900,
        CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901,
        CUDA_ERROR_STREAM_CAPTURE_MERGE = 902,
        CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903,
        CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904,
        CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905,
        CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906,
        CUDA_ERROR_CAPTURED_EVENT = 907,
        CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908,
        CUDA_ERROR_TIMEOUT = 909,
        CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910,
        CUDA_ERROR_UNKNOWN = 999,
    }
    enum CUDA_SUCCESS = cudaError_enum.CUDA_SUCCESS;
    enum CUDA_ERROR_INVALID_VALUE = cudaError_enum.CUDA_ERROR_INVALID_VALUE;
    enum CUDA_ERROR_OUT_OF_MEMORY = cudaError_enum.CUDA_ERROR_OUT_OF_MEMORY;
    enum CUDA_ERROR_NOT_INITIALIZED = cudaError_enum.CUDA_ERROR_NOT_INITIALIZED;
    enum CUDA_ERROR_DEINITIALIZED = cudaError_enum.CUDA_ERROR_DEINITIALIZED;
    enum CUDA_ERROR_PROFILER_DISABLED = cudaError_enum.CUDA_ERROR_PROFILER_DISABLED;
    enum CUDA_ERROR_PROFILER_NOT_INITIALIZED = cudaError_enum.CUDA_ERROR_PROFILER_NOT_INITIALIZED;
    enum CUDA_ERROR_PROFILER_ALREADY_STARTED = cudaError_enum.CUDA_ERROR_PROFILER_ALREADY_STARTED;
    enum CUDA_ERROR_PROFILER_ALREADY_STOPPED = cudaError_enum.CUDA_ERROR_PROFILER_ALREADY_STOPPED;
    enum CUDA_ERROR_NO_DEVICE = cudaError_enum.CUDA_ERROR_NO_DEVICE;
    enum CUDA_ERROR_INVALID_DEVICE = cudaError_enum.CUDA_ERROR_INVALID_DEVICE;
    enum CUDA_ERROR_INVALID_IMAGE = cudaError_enum.CUDA_ERROR_INVALID_IMAGE;
    enum CUDA_ERROR_INVALID_CONTEXT = cudaError_enum.CUDA_ERROR_INVALID_CONTEXT;
    enum CUDA_ERROR_CONTEXT_ALREADY_CURRENT = cudaError_enum.CUDA_ERROR_CONTEXT_ALREADY_CURRENT;
    enum CUDA_ERROR_MAP_FAILED = cudaError_enum.CUDA_ERROR_MAP_FAILED;
    enum CUDA_ERROR_UNMAP_FAILED = cudaError_enum.CUDA_ERROR_UNMAP_FAILED;
    enum CUDA_ERROR_ARRAY_IS_MAPPED = cudaError_enum.CUDA_ERROR_ARRAY_IS_MAPPED;
    enum CUDA_ERROR_ALREADY_MAPPED = cudaError_enum.CUDA_ERROR_ALREADY_MAPPED;
    enum CUDA_ERROR_NO_BINARY_FOR_GPU = cudaError_enum.CUDA_ERROR_NO_BINARY_FOR_GPU;
    enum CUDA_ERROR_ALREADY_ACQUIRED = cudaError_enum.CUDA_ERROR_ALREADY_ACQUIRED;
    enum CUDA_ERROR_NOT_MAPPED = cudaError_enum.CUDA_ERROR_NOT_MAPPED;
    enum CUDA_ERROR_NOT_MAPPED_AS_ARRAY = cudaError_enum.CUDA_ERROR_NOT_MAPPED_AS_ARRAY;
    enum CUDA_ERROR_NOT_MAPPED_AS_POINTER = cudaError_enum.CUDA_ERROR_NOT_MAPPED_AS_POINTER;
    enum CUDA_ERROR_ECC_UNCORRECTABLE = cudaError_enum.CUDA_ERROR_ECC_UNCORRECTABLE;
    enum CUDA_ERROR_UNSUPPORTED_LIMIT = cudaError_enum.CUDA_ERROR_UNSUPPORTED_LIMIT;
    enum CUDA_ERROR_CONTEXT_ALREADY_IN_USE = cudaError_enum.CUDA_ERROR_CONTEXT_ALREADY_IN_USE;
    enum CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = cudaError_enum.CUDA_ERROR_PEER_ACCESS_UNSUPPORTED;
    enum CUDA_ERROR_INVALID_PTX = cudaError_enum.CUDA_ERROR_INVALID_PTX;
    enum CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = cudaError_enum.CUDA_ERROR_INVALID_GRAPHICS_CONTEXT;
    enum CUDA_ERROR_NVLINK_UNCORRECTABLE = cudaError_enum.CUDA_ERROR_NVLINK_UNCORRECTABLE;
    enum CUDA_ERROR_JIT_COMPILER_NOT_FOUND = cudaError_enum.CUDA_ERROR_JIT_COMPILER_NOT_FOUND;
    enum CUDA_ERROR_INVALID_SOURCE = cudaError_enum.CUDA_ERROR_INVALID_SOURCE;
    enum CUDA_ERROR_FILE_NOT_FOUND = cudaError_enum.CUDA_ERROR_FILE_NOT_FOUND;
    enum CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = cudaError_enum.CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
    enum CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = cudaError_enum.CUDA_ERROR_SHARED_OBJECT_INIT_FAILED;
    enum CUDA_ERROR_OPERATING_SYSTEM = cudaError_enum.CUDA_ERROR_OPERATING_SYSTEM;
    enum CUDA_ERROR_INVALID_HANDLE = cudaError_enum.CUDA_ERROR_INVALID_HANDLE;
    enum CUDA_ERROR_ILLEGAL_STATE = cudaError_enum.CUDA_ERROR_ILLEGAL_STATE;
    enum CUDA_ERROR_NOT_FOUND = cudaError_enum.CUDA_ERROR_NOT_FOUND;
    enum CUDA_ERROR_NOT_READY = cudaError_enum.CUDA_ERROR_NOT_READY;
    enum CUDA_ERROR_ILLEGAL_ADDRESS = cudaError_enum.CUDA_ERROR_ILLEGAL_ADDRESS;
    enum CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = cudaError_enum.CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
    enum CUDA_ERROR_LAUNCH_TIMEOUT = cudaError_enum.CUDA_ERROR_LAUNCH_TIMEOUT;
    enum CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = cudaError_enum.CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING;
    enum CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = cudaError_enum.CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED;
    enum CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = cudaError_enum.CUDA_ERROR_PEER_ACCESS_NOT_ENABLED;
    enum CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = cudaError_enum.CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE;
    enum CUDA_ERROR_CONTEXT_IS_DESTROYED = cudaError_enum.CUDA_ERROR_CONTEXT_IS_DESTROYED;
    enum CUDA_ERROR_ASSERT = cudaError_enum.CUDA_ERROR_ASSERT;
    enum CUDA_ERROR_TOO_MANY_PEERS = cudaError_enum.CUDA_ERROR_TOO_MANY_PEERS;
    enum CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = cudaError_enum.CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED;
    enum CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = cudaError_enum.CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED;
    enum CUDA_ERROR_HARDWARE_STACK_ERROR = cudaError_enum.CUDA_ERROR_HARDWARE_STACK_ERROR;
    enum CUDA_ERROR_ILLEGAL_INSTRUCTION = cudaError_enum.CUDA_ERROR_ILLEGAL_INSTRUCTION;
    enum CUDA_ERROR_MISALIGNED_ADDRESS = cudaError_enum.CUDA_ERROR_MISALIGNED_ADDRESS;
    enum CUDA_ERROR_INVALID_ADDRESS_SPACE = cudaError_enum.CUDA_ERROR_INVALID_ADDRESS_SPACE;
    enum CUDA_ERROR_INVALID_PC = cudaError_enum.CUDA_ERROR_INVALID_PC;
    enum CUDA_ERROR_LAUNCH_FAILED = cudaError_enum.CUDA_ERROR_LAUNCH_FAILED;
    enum CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = cudaError_enum.CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE;
    enum CUDA_ERROR_NOT_PERMITTED = cudaError_enum.CUDA_ERROR_NOT_PERMITTED;
    enum CUDA_ERROR_NOT_SUPPORTED = cudaError_enum.CUDA_ERROR_NOT_SUPPORTED;
    enum CUDA_ERROR_SYSTEM_NOT_READY = cudaError_enum.CUDA_ERROR_SYSTEM_NOT_READY;
    enum CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = cudaError_enum.CUDA_ERROR_SYSTEM_DRIVER_MISMATCH;
    enum CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = cudaError_enum.CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE;
    enum CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = cudaError_enum.CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
    enum CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = cudaError_enum.CUDA_ERROR_STREAM_CAPTURE_INVALIDATED;
    enum CUDA_ERROR_STREAM_CAPTURE_MERGE = cudaError_enum.CUDA_ERROR_STREAM_CAPTURE_MERGE;
    enum CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = cudaError_enum.CUDA_ERROR_STREAM_CAPTURE_UNMATCHED;
    enum CUDA_ERROR_STREAM_CAPTURE_UNJOINED = cudaError_enum.CUDA_ERROR_STREAM_CAPTURE_UNJOINED;
    enum CUDA_ERROR_STREAM_CAPTURE_ISOLATION = cudaError_enum.CUDA_ERROR_STREAM_CAPTURE_ISOLATION;
    enum CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = cudaError_enum.CUDA_ERROR_STREAM_CAPTURE_IMPLICIT;
    enum CUDA_ERROR_CAPTURED_EVENT = cudaError_enum.CUDA_ERROR_CAPTURED_EVENT;
    enum CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = cudaError_enum.CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD;
    enum CUDA_ERROR_TIMEOUT = cudaError_enum.CUDA_ERROR_TIMEOUT;
    enum CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = cudaError_enum.CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE;
    enum CUDA_ERROR_UNKNOWN = cudaError_enum.CUDA_ERROR_UNKNOWN;
    alias CUresult = cudaError_enum;
    enum CUstreamCaptureMode_enum
    {
        CU_STREAM_CAPTURE_MODE_GLOBAL = 0,
        CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1,
        CU_STREAM_CAPTURE_MODE_RELAXED = 2,
    }
    enum CU_STREAM_CAPTURE_MODE_GLOBAL = CUstreamCaptureMode_enum.CU_STREAM_CAPTURE_MODE_GLOBAL;
    enum CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = CUstreamCaptureMode_enum.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;
    enum CU_STREAM_CAPTURE_MODE_RELAXED = CUstreamCaptureMode_enum.CU_STREAM_CAPTURE_MODE_RELAXED;
    alias CUstreamCaptureMode = CUstreamCaptureMode_enum;
    enum CUstreamCaptureStatus_enum
    {
        CU_STREAM_CAPTURE_STATUS_NONE = 0,
        CU_STREAM_CAPTURE_STATUS_ACTIVE = 1,
        CU_STREAM_CAPTURE_STATUS_INVALIDATED = 2,
    }
    enum CU_STREAM_CAPTURE_STATUS_NONE = CUstreamCaptureStatus_enum.CU_STREAM_CAPTURE_STATUS_NONE;
    enum CU_STREAM_CAPTURE_STATUS_ACTIVE = CUstreamCaptureStatus_enum.CU_STREAM_CAPTURE_STATUS_ACTIVE;
    enum CU_STREAM_CAPTURE_STATUS_INVALIDATED = CUstreamCaptureStatus_enum.CU_STREAM_CAPTURE_STATUS_INVALIDATED;
    alias CUstreamCaptureStatus = CUstreamCaptureStatus_enum;
    enum CUgraphNodeType_enum
    {
        CU_GRAPH_NODE_TYPE_KERNEL = 0,
        CU_GRAPH_NODE_TYPE_MEMCPY = 1,
        CU_GRAPH_NODE_TYPE_MEMSET = 2,
        CU_GRAPH_NODE_TYPE_HOST = 3,
        CU_GRAPH_NODE_TYPE_GRAPH = 4,
        CU_GRAPH_NODE_TYPE_EMPTY = 5,
        CU_GRAPH_NODE_TYPE_COUNT = 6,
    }
    enum CU_GRAPH_NODE_TYPE_KERNEL = CUgraphNodeType_enum.CU_GRAPH_NODE_TYPE_KERNEL;
    enum CU_GRAPH_NODE_TYPE_MEMCPY = CUgraphNodeType_enum.CU_GRAPH_NODE_TYPE_MEMCPY;
    enum CU_GRAPH_NODE_TYPE_MEMSET = CUgraphNodeType_enum.CU_GRAPH_NODE_TYPE_MEMSET;
    enum CU_GRAPH_NODE_TYPE_HOST = CUgraphNodeType_enum.CU_GRAPH_NODE_TYPE_HOST;
    enum CU_GRAPH_NODE_TYPE_GRAPH = CUgraphNodeType_enum.CU_GRAPH_NODE_TYPE_GRAPH;
    enum CU_GRAPH_NODE_TYPE_EMPTY = CUgraphNodeType_enum.CU_GRAPH_NODE_TYPE_EMPTY;
    enum CU_GRAPH_NODE_TYPE_COUNT = CUgraphNodeType_enum.CU_GRAPH_NODE_TYPE_COUNT;
    alias CUgraphNodeType = CUgraphNodeType_enum;
    struct CUDA_HOST_NODE_PARAMS_st
    {
        void function(void*) fn;
        void* userData;
    }
    alias CUDA_HOST_NODE_PARAMS = CUDA_HOST_NODE_PARAMS_st;
    struct CUDA_MEMSET_NODE_PARAMS_st
    {
        ulong dst;
        c_ulong pitch;
        uint value;
        uint elementSize;
        c_ulong width;
        c_ulong height;
    }
    alias CUDA_MEMSET_NODE_PARAMS = CUDA_MEMSET_NODE_PARAMS_st;
    struct CUDA_KERNEL_NODE_PARAMS_st
    {
        CUfunc_st* func;
        uint gridDimX;
        uint gridDimY;
        uint gridDimZ;
        uint blockDimX;
        uint blockDimY;
        uint blockDimZ;
        uint sharedMemBytes;
        void** kernelParams;
        void** extra;
    }
    alias CUDA_KERNEL_NODE_PARAMS = CUDA_KERNEL_NODE_PARAMS_st;
    alias CUhostFn = void function(void*);
    enum CUresourcetype_enum
    {
        CU_RESOURCE_TYPE_ARRAY = 0,
        CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 1,
        CU_RESOURCE_TYPE_LINEAR = 2,
        CU_RESOURCE_TYPE_PITCH2D = 3,
    }
    enum CU_RESOURCE_TYPE_ARRAY = CUresourcetype_enum.CU_RESOURCE_TYPE_ARRAY;
    enum CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = CUresourcetype_enum.CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
    enum CU_RESOURCE_TYPE_LINEAR = CUresourcetype_enum.CU_RESOURCE_TYPE_LINEAR;
    enum CU_RESOURCE_TYPE_PITCH2D = CUresourcetype_enum.CU_RESOURCE_TYPE_PITCH2D;
    alias CUresourcetype = CUresourcetype_enum;
    enum CUlimit_enum
    {
        CU_LIMIT_STACK_SIZE = 0,
        CU_LIMIT_PRINTF_FIFO_SIZE = 1,
        CU_LIMIT_MALLOC_HEAP_SIZE = 2,
        CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = 3,
        CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 4,
        CU_LIMIT_MAX_L2_FETCH_GRANULARITY = 5,
        CU_LIMIT_MAX = 6,
    }
    enum CU_LIMIT_STACK_SIZE = CUlimit_enum.CU_LIMIT_STACK_SIZE;
    enum CU_LIMIT_PRINTF_FIFO_SIZE = CUlimit_enum.CU_LIMIT_PRINTF_FIFO_SIZE;
    enum CU_LIMIT_MALLOC_HEAP_SIZE = CUlimit_enum.CU_LIMIT_MALLOC_HEAP_SIZE;
    enum CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = CUlimit_enum.CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH;
    enum CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = CUlimit_enum.CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT;
    enum CU_LIMIT_MAX_L2_FETCH_GRANULARITY = CUlimit_enum.CU_LIMIT_MAX_L2_FETCH_GRANULARITY;
    enum CU_LIMIT_MAX = CUlimit_enum.CU_LIMIT_MAX;
    alias CUlimit = CUlimit_enum;
    enum CUarray_cubemap_face_enum
    {
        CU_CUBEMAP_FACE_POSITIVE_X = 0,
        CU_CUBEMAP_FACE_NEGATIVE_X = 1,
        CU_CUBEMAP_FACE_POSITIVE_Y = 2,
        CU_CUBEMAP_FACE_NEGATIVE_Y = 3,
        CU_CUBEMAP_FACE_POSITIVE_Z = 4,
        CU_CUBEMAP_FACE_NEGATIVE_Z = 5,
    }
    enum CU_CUBEMAP_FACE_POSITIVE_X = CUarray_cubemap_face_enum.CU_CUBEMAP_FACE_POSITIVE_X;
    enum CU_CUBEMAP_FACE_NEGATIVE_X = CUarray_cubemap_face_enum.CU_CUBEMAP_FACE_NEGATIVE_X;
    enum CU_CUBEMAP_FACE_POSITIVE_Y = CUarray_cubemap_face_enum.CU_CUBEMAP_FACE_POSITIVE_Y;
    enum CU_CUBEMAP_FACE_NEGATIVE_Y = CUarray_cubemap_face_enum.CU_CUBEMAP_FACE_NEGATIVE_Y;
    enum CU_CUBEMAP_FACE_POSITIVE_Z = CUarray_cubemap_face_enum.CU_CUBEMAP_FACE_POSITIVE_Z;
    enum CU_CUBEMAP_FACE_NEGATIVE_Z = CUarray_cubemap_face_enum.CU_CUBEMAP_FACE_NEGATIVE_Z;
    alias CUarray_cubemap_face = CUarray_cubemap_face_enum;
    alias suseconds_t = c_long;
    enum CUgraphicsMapResourceFlags_enum
    {
        CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE = 0,
        CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY = 1,
        CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD = 2,
    }
    enum CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE = CUgraphicsMapResourceFlags_enum.CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE;
    enum CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY = CUgraphicsMapResourceFlags_enum.CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY;
    enum CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD = CUgraphicsMapResourceFlags_enum.CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD;
    alias __fd_mask = c_long;
    alias CUgraphicsMapResourceFlags = CUgraphicsMapResourceFlags_enum;
    enum CUgraphicsRegisterFlags_enum
    {
        CU_GRAPHICS_REGISTER_FLAGS_NONE = 0,
        CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY = 1,
        CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 2,
        CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST = 4,
        CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = 8,
    }
    enum CU_GRAPHICS_REGISTER_FLAGS_NONE = CUgraphicsRegisterFlags_enum.CU_GRAPHICS_REGISTER_FLAGS_NONE;
    enum CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY = CUgraphicsRegisterFlags_enum.CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY;
    enum CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = CUgraphicsRegisterFlags_enum.CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD;
    enum CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST = CUgraphicsRegisterFlags_enum.CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST;
    enum CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = CUgraphicsRegisterFlags_enum.CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER;
    alias CUgraphicsRegisterFlags = CUgraphicsRegisterFlags_enum;
    struct fd_set
    {
        c_long[16] __fds_bits;
    }
    struct CUlinkState_st;
    alias CUlinkState = CUlinkState_st*;
    alias fd_mask = c_long;
    enum CUjitInputType_enum
    {
        CU_JIT_INPUT_CUBIN = 0,
        CU_JIT_INPUT_PTX = 1,
        CU_JIT_INPUT_FATBINARY = 2,
        CU_JIT_INPUT_OBJECT = 3,
        CU_JIT_INPUT_LIBRARY = 4,
        CU_JIT_NUM_INPUT_TYPES = 5,
    }
    enum CU_JIT_INPUT_CUBIN = CUjitInputType_enum.CU_JIT_INPUT_CUBIN;
    enum CU_JIT_INPUT_PTX = CUjitInputType_enum.CU_JIT_INPUT_PTX;
    enum CU_JIT_INPUT_FATBINARY = CUjitInputType_enum.CU_JIT_INPUT_FATBINARY;
    enum CU_JIT_INPUT_OBJECT = CUjitInputType_enum.CU_JIT_INPUT_OBJECT;
    enum CU_JIT_INPUT_LIBRARY = CUjitInputType_enum.CU_JIT_INPUT_LIBRARY;
    enum CU_JIT_NUM_INPUT_TYPES = CUjitInputType_enum.CU_JIT_NUM_INPUT_TYPES;
    alias CUjitInputType = CUjitInputType_enum;
    enum CUjit_cacheMode_enum
    {
        CU_JIT_CACHE_OPTION_NONE = 0,
        CU_JIT_CACHE_OPTION_CG = 1,
        CU_JIT_CACHE_OPTION_CA = 2,
    }
    enum CU_JIT_CACHE_OPTION_NONE = CUjit_cacheMode_enum.CU_JIT_CACHE_OPTION_NONE;
    enum CU_JIT_CACHE_OPTION_CG = CUjit_cacheMode_enum.CU_JIT_CACHE_OPTION_CG;
    enum CU_JIT_CACHE_OPTION_CA = CUjit_cacheMode_enum.CU_JIT_CACHE_OPTION_CA;
    alias CUjit_cacheMode = CUjit_cacheMode_enum;
    int select(int, fd_set*, fd_set*, fd_set*, timeval*) @nogc nothrow;
    int pselect(int, fd_set*, fd_set*, fd_set*, const(timespec)*, const(__sigset_t)*) @nogc nothrow;
    enum CUjit_fallback_enum
    {
        CU_PREFER_PTX = 0,
        CU_PREFER_BINARY = 1,
    }
    enum CU_PREFER_PTX = CUjit_fallback_enum.CU_PREFER_PTX;
    enum CU_PREFER_BINARY = CUjit_fallback_enum.CU_PREFER_BINARY;
    alias CUjit_fallback = CUjit_fallback_enum;
    enum CUjit_target_enum
    {
        CU_TARGET_COMPUTE_20 = 20,
        CU_TARGET_COMPUTE_21 = 21,
        CU_TARGET_COMPUTE_30 = 30,
        CU_TARGET_COMPUTE_32 = 32,
        CU_TARGET_COMPUTE_35 = 35,
        CU_TARGET_COMPUTE_37 = 37,
        CU_TARGET_COMPUTE_50 = 50,
        CU_TARGET_COMPUTE_52 = 52,
        CU_TARGET_COMPUTE_53 = 53,
        CU_TARGET_COMPUTE_60 = 60,
        CU_TARGET_COMPUTE_61 = 61,
        CU_TARGET_COMPUTE_62 = 62,
        CU_TARGET_COMPUTE_70 = 70,
        CU_TARGET_COMPUTE_72 = 72,
        CU_TARGET_COMPUTE_75 = 75,
    }
    enum CU_TARGET_COMPUTE_20 = CUjit_target_enum.CU_TARGET_COMPUTE_20;
    enum CU_TARGET_COMPUTE_21 = CUjit_target_enum.CU_TARGET_COMPUTE_21;
    enum CU_TARGET_COMPUTE_30 = CUjit_target_enum.CU_TARGET_COMPUTE_30;
    enum CU_TARGET_COMPUTE_32 = CUjit_target_enum.CU_TARGET_COMPUTE_32;
    enum CU_TARGET_COMPUTE_35 = CUjit_target_enum.CU_TARGET_COMPUTE_35;
    enum CU_TARGET_COMPUTE_37 = CUjit_target_enum.CU_TARGET_COMPUTE_37;
    enum CU_TARGET_COMPUTE_50 = CUjit_target_enum.CU_TARGET_COMPUTE_50;
    enum CU_TARGET_COMPUTE_52 = CUjit_target_enum.CU_TARGET_COMPUTE_52;
    enum CU_TARGET_COMPUTE_53 = CUjit_target_enum.CU_TARGET_COMPUTE_53;
    enum CU_TARGET_COMPUTE_60 = CUjit_target_enum.CU_TARGET_COMPUTE_60;
    enum CU_TARGET_COMPUTE_61 = CUjit_target_enum.CU_TARGET_COMPUTE_61;
    enum CU_TARGET_COMPUTE_62 = CUjit_target_enum.CU_TARGET_COMPUTE_62;
    enum CU_TARGET_COMPUTE_70 = CUjit_target_enum.CU_TARGET_COMPUTE_70;
    enum CU_TARGET_COMPUTE_72 = CUjit_target_enum.CU_TARGET_COMPUTE_72;
    enum CU_TARGET_COMPUTE_75 = CUjit_target_enum.CU_TARGET_COMPUTE_75;
    alias CUjit_target = CUjit_target_enum;
    enum CUjit_option_enum
    {
        CU_JIT_MAX_REGISTERS = 0,
        CU_JIT_THREADS_PER_BLOCK = 1,
        CU_JIT_WALL_TIME = 2,
        CU_JIT_INFO_LOG_BUFFER = 3,
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
        CU_JIT_ERROR_LOG_BUFFER = 5,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
        CU_JIT_OPTIMIZATION_LEVEL = 7,
        CU_JIT_TARGET_FROM_CUCONTEXT = 8,
        CU_JIT_TARGET = 9,
        CU_JIT_FALLBACK_STRATEGY = 10,
        CU_JIT_GENERATE_DEBUG_INFO = 11,
        CU_JIT_LOG_VERBOSE = 12,
        CU_JIT_GENERATE_LINE_INFO = 13,
        CU_JIT_CACHE_MODE = 14,
        CU_JIT_NEW_SM3X_OPT = 15,
        CU_JIT_FAST_COMPILE = 16,
        CU_JIT_GLOBAL_SYMBOL_NAMES = 17,
        CU_JIT_GLOBAL_SYMBOL_ADDRESSES = 18,
        CU_JIT_GLOBAL_SYMBOL_COUNT = 19,
        CU_JIT_NUM_OPTIONS = 20,
    }
    enum CU_JIT_MAX_REGISTERS = CUjit_option_enum.CU_JIT_MAX_REGISTERS;
    enum CU_JIT_THREADS_PER_BLOCK = CUjit_option_enum.CU_JIT_THREADS_PER_BLOCK;
    enum CU_JIT_WALL_TIME = CUjit_option_enum.CU_JIT_WALL_TIME;
    enum CU_JIT_INFO_LOG_BUFFER = CUjit_option_enum.CU_JIT_INFO_LOG_BUFFER;
    enum CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = CUjit_option_enum.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
    enum CU_JIT_ERROR_LOG_BUFFER = CUjit_option_enum.CU_JIT_ERROR_LOG_BUFFER;
    enum CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = CUjit_option_enum.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
    enum CU_JIT_OPTIMIZATION_LEVEL = CUjit_option_enum.CU_JIT_OPTIMIZATION_LEVEL;
    enum CU_JIT_TARGET_FROM_CUCONTEXT = CUjit_option_enum.CU_JIT_TARGET_FROM_CUCONTEXT;
    enum CU_JIT_TARGET = CUjit_option_enum.CU_JIT_TARGET;
    enum CU_JIT_FALLBACK_STRATEGY = CUjit_option_enum.CU_JIT_FALLBACK_STRATEGY;
    enum CU_JIT_GENERATE_DEBUG_INFO = CUjit_option_enum.CU_JIT_GENERATE_DEBUG_INFO;
    enum CU_JIT_LOG_VERBOSE = CUjit_option_enum.CU_JIT_LOG_VERBOSE;
    enum CU_JIT_GENERATE_LINE_INFO = CUjit_option_enum.CU_JIT_GENERATE_LINE_INFO;
    enum CU_JIT_CACHE_MODE = CUjit_option_enum.CU_JIT_CACHE_MODE;
    enum CU_JIT_NEW_SM3X_OPT = CUjit_option_enum.CU_JIT_NEW_SM3X_OPT;
    enum CU_JIT_FAST_COMPILE = CUjit_option_enum.CU_JIT_FAST_COMPILE;
    enum CU_JIT_GLOBAL_SYMBOL_NAMES = CUjit_option_enum.CU_JIT_GLOBAL_SYMBOL_NAMES;
    enum CU_JIT_GLOBAL_SYMBOL_ADDRESSES = CUjit_option_enum.CU_JIT_GLOBAL_SYMBOL_ADDRESSES;
    enum CU_JIT_GLOBAL_SYMBOL_COUNT = CUjit_option_enum.CU_JIT_GLOBAL_SYMBOL_COUNT;
    enum CU_JIT_NUM_OPTIONS = CUjit_option_enum.CU_JIT_NUM_OPTIONS;
    uint gnu_dev_major(c_ulong) @nogc nothrow;
    uint gnu_dev_minor(c_ulong) @nogc nothrow;
    c_ulong gnu_dev_makedev(uint, uint) @nogc nothrow;
    alias CUjit_option = CUjit_option_enum;
    enum CUmem_range_attribute_enum
    {
        CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY = 1,
        CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION = 2,
        CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY = 3,
        CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION = 4,
    }
    enum CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY = CUmem_range_attribute_enum.CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY;
    enum CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION = CUmem_range_attribute_enum.CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION;
    enum CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY = CUmem_range_attribute_enum.CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY;
    enum CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION = CUmem_range_attribute_enum.CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION;
    alias CUmem_range_attribute = CUmem_range_attribute_enum;
    enum CUmem_advise_enum
    {
        CU_MEM_ADVISE_SET_READ_MOSTLY = 1,
        CU_MEM_ADVISE_UNSET_READ_MOSTLY = 2,
        CU_MEM_ADVISE_SET_PREFERRED_LOCATION = 3,
        CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION = 4,
        CU_MEM_ADVISE_SET_ACCESSED_BY = 5,
        CU_MEM_ADVISE_UNSET_ACCESSED_BY = 6,
    }
    enum CU_MEM_ADVISE_SET_READ_MOSTLY = CUmem_advise_enum.CU_MEM_ADVISE_SET_READ_MOSTLY;
    enum CU_MEM_ADVISE_UNSET_READ_MOSTLY = CUmem_advise_enum.CU_MEM_ADVISE_UNSET_READ_MOSTLY;
    enum CU_MEM_ADVISE_SET_PREFERRED_LOCATION = CUmem_advise_enum.CU_MEM_ADVISE_SET_PREFERRED_LOCATION;
    enum CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION = CUmem_advise_enum.CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION;
    enum CU_MEM_ADVISE_SET_ACCESSED_BY = CUmem_advise_enum.CU_MEM_ADVISE_SET_ACCESSED_BY;
    enum CU_MEM_ADVISE_UNSET_ACCESSED_BY = CUmem_advise_enum.CU_MEM_ADVISE_UNSET_ACCESSED_BY;
    alias u_char = ubyte;
    alias u_short = ushort;
    alias u_int = uint;
    alias u_long = c_ulong;
    alias quad_t = c_long;
    alias u_quad_t = c_ulong;
    alias fsid_t = __fsid_t;
    alias CUmem_advise = CUmem_advise_enum;
    alias loff_t = c_long;
    alias ino_t = c_ulong;
    enum CUcomputemode_enum
    {
        CU_COMPUTEMODE_DEFAULT = 0,
        CU_COMPUTEMODE_PROHIBITED = 2,
        CU_COMPUTEMODE_EXCLUSIVE_PROCESS = 3,
    }
    enum CU_COMPUTEMODE_DEFAULT = CUcomputemode_enum.CU_COMPUTEMODE_DEFAULT;
    enum CU_COMPUTEMODE_PROHIBITED = CUcomputemode_enum.CU_COMPUTEMODE_PROHIBITED;
    enum CU_COMPUTEMODE_EXCLUSIVE_PROCESS = CUcomputemode_enum.CU_COMPUTEMODE_EXCLUSIVE_PROCESS;
    alias dev_t = c_ulong;
    alias CUcomputemode = CUcomputemode_enum;
    alias gid_t = uint;
    enum CUmemorytype_enum
    {
        CU_MEMORYTYPE_HOST = 1,
        CU_MEMORYTYPE_DEVICE = 2,
        CU_MEMORYTYPE_ARRAY = 3,
        CU_MEMORYTYPE_UNIFIED = 4,
    }
    enum CU_MEMORYTYPE_HOST = CUmemorytype_enum.CU_MEMORYTYPE_HOST;
    enum CU_MEMORYTYPE_DEVICE = CUmemorytype_enum.CU_MEMORYTYPE_DEVICE;
    enum CU_MEMORYTYPE_ARRAY = CUmemorytype_enum.CU_MEMORYTYPE_ARRAY;
    enum CU_MEMORYTYPE_UNIFIED = CUmemorytype_enum.CU_MEMORYTYPE_UNIFIED;
    alias mode_t = uint;
    alias CUmemorytype = CUmemorytype_enum;
    alias nlink_t = c_ulong;
    enum CUshared_carveout_enum
    {
        CU_SHAREDMEM_CARVEOUT_DEFAULT = -1,
        CU_SHAREDMEM_CARVEOUT_MAX_SHARED = 100,
        CU_SHAREDMEM_CARVEOUT_MAX_L1 = 0,
    }
    enum CU_SHAREDMEM_CARVEOUT_DEFAULT = CUshared_carveout_enum.CU_SHAREDMEM_CARVEOUT_DEFAULT;
    enum CU_SHAREDMEM_CARVEOUT_MAX_SHARED = CUshared_carveout_enum.CU_SHAREDMEM_CARVEOUT_MAX_SHARED;
    enum CU_SHAREDMEM_CARVEOUT_MAX_L1 = CUshared_carveout_enum.CU_SHAREDMEM_CARVEOUT_MAX_L1;
    alias uid_t = uint;
    alias CUshared_carveout = CUshared_carveout_enum;
    alias off_t = c_long;
    enum CUsharedconfig_enum
    {
        CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = 0,
        CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = 1,
        CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = 2,
    }
    enum CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = CUsharedconfig_enum.CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE;
    enum CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = CUsharedconfig_enum.CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE;
    enum CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = CUsharedconfig_enum.CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE;
    alias pid_t = int;
    alias CUsharedconfig = CUsharedconfig_enum;
    alias id_t = uint;
    enum CUfunc_cache_enum
    {
        CU_FUNC_CACHE_PREFER_NONE = 0,
        CU_FUNC_CACHE_PREFER_SHARED = 1,
        CU_FUNC_CACHE_PREFER_L1 = 2,
        CU_FUNC_CACHE_PREFER_EQUAL = 3,
    }
    enum CU_FUNC_CACHE_PREFER_NONE = CUfunc_cache_enum.CU_FUNC_CACHE_PREFER_NONE;
    enum CU_FUNC_CACHE_PREFER_SHARED = CUfunc_cache_enum.CU_FUNC_CACHE_PREFER_SHARED;
    enum CU_FUNC_CACHE_PREFER_L1 = CUfunc_cache_enum.CU_FUNC_CACHE_PREFER_L1;
    enum CU_FUNC_CACHE_PREFER_EQUAL = CUfunc_cache_enum.CU_FUNC_CACHE_PREFER_EQUAL;
    alias ssize_t = c_long;
    alias CUfunc_cache = CUfunc_cache_enum;
    alias daddr_t = int;
    alias caddr_t = char*;
    enum CUfunction_attribute_enum
    {
        CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
        CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
        CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,
        CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,
        CU_FUNC_ATTRIBUTE_NUM_REGS = 4,
        CU_FUNC_ATTRIBUTE_PTX_VERSION = 5,
        CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6,
        CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
        CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9,
        CU_FUNC_ATTRIBUTE_MAX = 10,
    }
    enum CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = CUfunction_attribute_enum.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK;
    enum CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = CUfunction_attribute_enum.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES;
    enum CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = CUfunction_attribute_enum.CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES;
    enum CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = CUfunction_attribute_enum.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES;
    enum CU_FUNC_ATTRIBUTE_NUM_REGS = CUfunction_attribute_enum.CU_FUNC_ATTRIBUTE_NUM_REGS;
    enum CU_FUNC_ATTRIBUTE_PTX_VERSION = CUfunction_attribute_enum.CU_FUNC_ATTRIBUTE_PTX_VERSION;
    enum CU_FUNC_ATTRIBUTE_BINARY_VERSION = CUfunction_attribute_enum.CU_FUNC_ATTRIBUTE_BINARY_VERSION;
    enum CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = CUfunction_attribute_enum.CU_FUNC_ATTRIBUTE_CACHE_MODE_CA;
    enum CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = CUfunction_attribute_enum.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
    enum CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = CUfunction_attribute_enum.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT;
    enum CU_FUNC_ATTRIBUTE_MAX = CUfunction_attribute_enum.CU_FUNC_ATTRIBUTE_MAX;
    alias key_t = int;
    alias CUfunction_attribute = CUfunction_attribute_enum;
    enum CUpointer_attribute_enum
    {
        CU_POINTER_ATTRIBUTE_CONTEXT = 1,
        CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,
        CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3,
        CU_POINTER_ATTRIBUTE_HOST_POINTER = 4,
        CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5,
        CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6,
        CU_POINTER_ATTRIBUTE_BUFFER_ID = 7,
        CU_POINTER_ATTRIBUTE_IS_MANAGED = 8,
        CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9,
        CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE = 10,
        CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11,
        CU_POINTER_ATTRIBUTE_RANGE_SIZE = 12,
        CU_POINTER_ATTRIBUTE_MAPPED = 13,
        CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = 14,
    }
    enum CU_POINTER_ATTRIBUTE_CONTEXT = CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_CONTEXT;
    enum CU_POINTER_ATTRIBUTE_MEMORY_TYPE = CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_MEMORY_TYPE;
    enum CU_POINTER_ATTRIBUTE_DEVICE_POINTER = CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_DEVICE_POINTER;
    enum CU_POINTER_ATTRIBUTE_HOST_POINTER = CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_HOST_POINTER;
    enum CU_POINTER_ATTRIBUTE_P2P_TOKENS = CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_P2P_TOKENS;
    enum CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_SYNC_MEMOPS;
    enum CU_POINTER_ATTRIBUTE_BUFFER_ID = CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_BUFFER_ID;
    enum CU_POINTER_ATTRIBUTE_IS_MANAGED = CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_IS_MANAGED;
    enum CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL;
    enum CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE = CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE;
    enum CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_RANGE_START_ADDR;
    enum CU_POINTER_ATTRIBUTE_RANGE_SIZE = CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_RANGE_SIZE;
    enum CU_POINTER_ATTRIBUTE_MAPPED = CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_MAPPED;
    enum CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = CUpointer_attribute_enum.CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES;
    alias CUpointer_attribute = CUpointer_attribute_enum;
    alias u_int8_t = ubyte;
    alias u_int16_t = ushort;
    alias u_int32_t = uint;
    alias u_int64_t = c_ulong;
    alias register_t = c_long;
    struct CUdevprop_st
    {
        int maxThreadsPerBlock;
        int[3] maxThreadsDim;
        int[3] maxGridSize;
        int sharedMemPerBlock;
        int totalConstantMemory;
        int SIMDWidth;
        int memPitch;
        int regsPerBlock;
        int clockRate;
        int textureAlign;
    }
    alias CUdevprop = CUdevprop_st;
    alias blksize_t = c_long;
    enum CUdevice_attribute_enum
    {
        CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
        CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
        CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
        CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
        CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
        CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
        CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
        CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8,
        CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
        CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
        CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
        CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
        CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12,
        CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
        CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
        CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
        CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
        CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
        CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
        CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
        CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = 27,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = 28,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = 29,
        CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
        CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
        CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
        CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
        CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
        CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
        CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
        CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
        CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
        CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
        CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
        CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
        CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
        CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
        CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
        CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
        CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
        CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
        CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
        CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
        CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
        CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
        CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
        CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
        CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
        CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
        CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
        CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
        CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
        CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS = 92,
        CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 93,
        CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 94,
        CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
        CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
        CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
        CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
        CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
        CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
        CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,
        CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
        CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
        CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
        CU_DEVICE_ATTRIBUTE_MAX = 106,
    }
    enum CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK;
    enum CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X;
    enum CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y;
    enum CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z;
    enum CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X;
    enum CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y;
    enum CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z;
    enum CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK;
    enum CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK;
    enum CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY;
    enum CU_DEVICE_ATTRIBUTE_WARP_SIZE = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_WARP_SIZE;
    enum CU_DEVICE_ATTRIBUTE_MAX_PITCH = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAX_PITCH;
    enum CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK;
    enum CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK;
    enum CU_DEVICE_ATTRIBUTE_CLOCK_RATE = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_CLOCK_RATE;
    enum CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT;
    enum CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_GPU_OVERLAP;
    enum CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
    enum CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT;
    enum CU_DEVICE_ATTRIBUTE_INTEGRATED = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_INTEGRATED;
    enum CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY;
    enum CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_COMPUTE_MODE;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES;
    enum CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT;
    enum CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS;
    enum CU_DEVICE_ATTRIBUTE_ECC_ENABLED = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_ECC_ENABLED;
    enum CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_PCI_BUS_ID;
    enum CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID;
    enum CU_DEVICE_ATTRIBUTE_TCC_DRIVER = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_TCC_DRIVER;
    enum CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE;
    enum CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH;
    enum CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE;
    enum CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR;
    enum CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT;
    enum CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS;
    enum CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE;
    enum CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID;
    enum CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT;
    enum CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR;
    enum CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR;
    enum CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH;
    enum CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED;
    enum CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED;
    enum CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED;
    enum CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR;
    enum CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR;
    enum CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY;
    enum CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD;
    enum CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID;
    enum CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED;
    enum CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO;
    enum CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS;
    enum CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS;
    enum CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED;
    enum CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM;
    enum CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS;
    enum CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS;
    enum CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR;
    enum CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH;
    enum CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH;
    enum CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN;
    enum CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES;
    enum CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED;
    enum CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES;
    enum CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST;
    enum CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED;
    enum CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED;
    enum CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED;
    enum CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED;
    enum CU_DEVICE_ATTRIBUTE_MAX = CUdevice_attribute_enum.CU_DEVICE_ATTRIBUTE_MAX;
    alias blkcnt_t = c_long;
    alias CUdevice_attribute = CUdevice_attribute_enum;
    alias fsblkcnt_t = c_ulong;
    enum CUfilter_mode_enum
    {
        CU_TR_FILTER_MODE_POINT = 0,
        CU_TR_FILTER_MODE_LINEAR = 1,
    }
    enum CU_TR_FILTER_MODE_POINT = CUfilter_mode_enum.CU_TR_FILTER_MODE_POINT;
    enum CU_TR_FILTER_MODE_LINEAR = CUfilter_mode_enum.CU_TR_FILTER_MODE_LINEAR;
    alias fsfilcnt_t = c_ulong;
    alias CUfilter_mode = CUfilter_mode_enum;
    enum CUaddress_mode_enum
    {
        CU_TR_ADDRESS_MODE_WRAP = 0,
        CU_TR_ADDRESS_MODE_CLAMP = 1,
        CU_TR_ADDRESS_MODE_MIRROR = 2,
        CU_TR_ADDRESS_MODE_BORDER = 3,
    }
    enum CU_TR_ADDRESS_MODE_WRAP = CUaddress_mode_enum.CU_TR_ADDRESS_MODE_WRAP;
    enum CU_TR_ADDRESS_MODE_CLAMP = CUaddress_mode_enum.CU_TR_ADDRESS_MODE_CLAMP;
    enum CU_TR_ADDRESS_MODE_MIRROR = CUaddress_mode_enum.CU_TR_ADDRESS_MODE_MIRROR;
    enum CU_TR_ADDRESS_MODE_BORDER = CUaddress_mode_enum.CU_TR_ADDRESS_MODE_BORDER;
    alias cuuint32_t = uint;
    alias cuuint64_t = c_ulong;
    alias CUaddress_mode = CUaddress_mode_enum;
    enum CUarray_format_enum
    {
        CU_AD_FORMAT_UNSIGNED_INT8 = 1,
        CU_AD_FORMAT_UNSIGNED_INT16 = 2,
        CU_AD_FORMAT_UNSIGNED_INT32 = 3,
        CU_AD_FORMAT_SIGNED_INT8 = 8,
        CU_AD_FORMAT_SIGNED_INT16 = 9,
        CU_AD_FORMAT_SIGNED_INT32 = 10,
        CU_AD_FORMAT_HALF = 16,
        CU_AD_FORMAT_FLOAT = 32,
    }
    enum CU_AD_FORMAT_UNSIGNED_INT8 = CUarray_format_enum.CU_AD_FORMAT_UNSIGNED_INT8;
    enum CU_AD_FORMAT_UNSIGNED_INT16 = CUarray_format_enum.CU_AD_FORMAT_UNSIGNED_INT16;
    enum CU_AD_FORMAT_UNSIGNED_INT32 = CUarray_format_enum.CU_AD_FORMAT_UNSIGNED_INT32;
    enum CU_AD_FORMAT_SIGNED_INT8 = CUarray_format_enum.CU_AD_FORMAT_SIGNED_INT8;
    enum CU_AD_FORMAT_SIGNED_INT16 = CUarray_format_enum.CU_AD_FORMAT_SIGNED_INT16;
    enum CU_AD_FORMAT_SIGNED_INT32 = CUarray_format_enum.CU_AD_FORMAT_SIGNED_INT32;
    enum CU_AD_FORMAT_HALF = CUarray_format_enum.CU_AD_FORMAT_HALF;
    enum CU_AD_FORMAT_FLOAT = CUarray_format_enum.CU_AD_FORMAT_FLOAT;
    alias CUarray_format = CUarray_format_enum;
    enum CUoccupancy_flags_enum
    {
        CU_OCCUPANCY_DEFAULT = 0,
        CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE = 1,
    }
    enum CU_OCCUPANCY_DEFAULT = CUoccupancy_flags_enum.CU_OCCUPANCY_DEFAULT;
    enum CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE = CUoccupancy_flags_enum.CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE;
    alias CUoccupancy_flags = CUoccupancy_flags_enum;
    union CUstreamBatchMemOpParams_union
    {
        CUstreamBatchMemOpType_enum operation;
        struct CUstreamMemOpWaitValueParams_st
        {
            CUstreamBatchMemOpType_enum operation;
            ulong address;
            static union _Anonymous_25
            {
                uint value;
                c_ulong value64;
            }
            _Anonymous_25 _anonymous_26;
            auto value() @property @nogc pure nothrow { return _anonymous_26.value; }
            void value(_T_)(auto ref _T_ val) @property @nogc pure nothrow { _anonymous_26.value = val; }
            auto value64() @property @nogc pure nothrow { return _anonymous_26.value64; }
            void value64(_T_)(auto ref _T_ val) @property @nogc pure nothrow { _anonymous_26.value64 = val; }
            uint flags;
            ulong alias_;
        }
        CUstreamBatchMemOpParams_union.CUstreamMemOpWaitValueParams_st waitValue;
        struct CUstreamMemOpWriteValueParams_st
        {
            CUstreamBatchMemOpType_enum operation;
            ulong address;
            static union _Anonymous_27
            {
                uint value;
                c_ulong value64;
            }
            _Anonymous_27 _anonymous_28;
            auto value() @property @nogc pure nothrow { return _anonymous_28.value; }
            void value(_T_)(auto ref _T_ val) @property @nogc pure nothrow { _anonymous_28.value = val; }
            auto value64() @property @nogc pure nothrow { return _anonymous_28.value64; }
            void value64(_T_)(auto ref _T_ val) @property @nogc pure nothrow { _anonymous_28.value64 = val; }
            uint flags;
            ulong alias_;
        }
        CUstreamBatchMemOpParams_union.CUstreamMemOpWriteValueParams_st writeValue;
        struct CUstreamMemOpFlushRemoteWritesParams_st
        {
            CUstreamBatchMemOpType_enum operation;
            uint flags;
        }
        CUstreamBatchMemOpParams_union.CUstreamMemOpFlushRemoteWritesParams_st flushRemoteWrites;
        c_ulong[6] pad;
    }
    alias CUstreamBatchMemOpParams = CUstreamBatchMemOpParams_union;
    enum CUstreamBatchMemOpType_enum
    {
        CU_STREAM_MEM_OP_WAIT_VALUE_32 = 1,
        CU_STREAM_MEM_OP_WRITE_VALUE_32 = 2,
        CU_STREAM_MEM_OP_WAIT_VALUE_64 = 4,
        CU_STREAM_MEM_OP_WRITE_VALUE_64 = 5,
        CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES = 3,
    }
    enum CU_STREAM_MEM_OP_WAIT_VALUE_32 = CUstreamBatchMemOpType_enum.CU_STREAM_MEM_OP_WAIT_VALUE_32;
    enum CU_STREAM_MEM_OP_WRITE_VALUE_32 = CUstreamBatchMemOpType_enum.CU_STREAM_MEM_OP_WRITE_VALUE_32;
    enum CU_STREAM_MEM_OP_WAIT_VALUE_64 = CUstreamBatchMemOpType_enum.CU_STREAM_MEM_OP_WAIT_VALUE_64;
    enum CU_STREAM_MEM_OP_WRITE_VALUE_64 = CUstreamBatchMemOpType_enum.CU_STREAM_MEM_OP_WRITE_VALUE_64;
    enum CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES = CUstreamBatchMemOpType_enum.CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES;
    alias CUstreamBatchMemOpType = CUstreamBatchMemOpType_enum;
    enum CUstreamWriteValue_flags_enum
    {
        CU_STREAM_WRITE_VALUE_DEFAULT = 0,
        CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER = 1,
    }
    enum CU_STREAM_WRITE_VALUE_DEFAULT = CUstreamWriteValue_flags_enum.CU_STREAM_WRITE_VALUE_DEFAULT;
    enum CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER = CUstreamWriteValue_flags_enum.CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER;
    alias CUstreamWriteValue_flags = CUstreamWriteValue_flags_enum;
    enum CUstreamWaitValue_flags_enum
    {
        CU_STREAM_WAIT_VALUE_GEQ = 0,
        CU_STREAM_WAIT_VALUE_EQ = 1,
        CU_STREAM_WAIT_VALUE_AND = 2,
        CU_STREAM_WAIT_VALUE_NOR = 3,
        CU_STREAM_WAIT_VALUE_FLUSH = 1073741824,
    }
    enum CU_STREAM_WAIT_VALUE_GEQ = CUstreamWaitValue_flags_enum.CU_STREAM_WAIT_VALUE_GEQ;
    enum CU_STREAM_WAIT_VALUE_EQ = CUstreamWaitValue_flags_enum.CU_STREAM_WAIT_VALUE_EQ;
    enum CU_STREAM_WAIT_VALUE_AND = CUstreamWaitValue_flags_enum.CU_STREAM_WAIT_VALUE_AND;
    enum CU_STREAM_WAIT_VALUE_NOR = CUstreamWaitValue_flags_enum.CU_STREAM_WAIT_VALUE_NOR;
    enum CU_STREAM_WAIT_VALUE_FLUSH = CUstreamWaitValue_flags_enum.CU_STREAM_WAIT_VALUE_FLUSH;
    alias CUstreamWaitValue_flags = CUstreamWaitValue_flags_enum;
    enum CUevent_flags_enum
    {
        CU_EVENT_DEFAULT = 0,
        CU_EVENT_BLOCKING_SYNC = 1,
        CU_EVENT_DISABLE_TIMING = 2,
        CU_EVENT_INTERPROCESS = 4,
    }
    enum CU_EVENT_DEFAULT = CUevent_flags_enum.CU_EVENT_DEFAULT;
    enum CU_EVENT_BLOCKING_SYNC = CUevent_flags_enum.CU_EVENT_BLOCKING_SYNC;
    enum CU_EVENT_DISABLE_TIMING = CUevent_flags_enum.CU_EVENT_DISABLE_TIMING;
    enum CU_EVENT_INTERPROCESS = CUevent_flags_enum.CU_EVENT_INTERPROCESS;
    alias CUevent_flags = CUevent_flags_enum;
    enum CUstream_flags_enum
    {
        CU_STREAM_DEFAULT = 0,
        CU_STREAM_NON_BLOCKING = 1,
    }
    enum CU_STREAM_DEFAULT = CUstream_flags_enum.CU_STREAM_DEFAULT;
    enum CU_STREAM_NON_BLOCKING = CUstream_flags_enum.CU_STREAM_NON_BLOCKING;
    alias CUstream_flags = CUstream_flags_enum;
    enum CUctx_flags_enum
    {
        CU_CTX_SCHED_AUTO = 0,
        CU_CTX_SCHED_SPIN = 1,
        CU_CTX_SCHED_YIELD = 2,
        CU_CTX_SCHED_BLOCKING_SYNC = 4,
        CU_CTX_BLOCKING_SYNC = 4,
        CU_CTX_SCHED_MASK = 7,
        CU_CTX_MAP_HOST = 8,
        CU_CTX_LMEM_RESIZE_TO_MAX = 16,
        CU_CTX_FLAGS_MASK = 31,
    }
    enum CU_CTX_SCHED_AUTO = CUctx_flags_enum.CU_CTX_SCHED_AUTO;
    enum CU_CTX_SCHED_SPIN = CUctx_flags_enum.CU_CTX_SCHED_SPIN;
    enum CU_CTX_SCHED_YIELD = CUctx_flags_enum.CU_CTX_SCHED_YIELD;
    enum CU_CTX_SCHED_BLOCKING_SYNC = CUctx_flags_enum.CU_CTX_SCHED_BLOCKING_SYNC;
    enum CU_CTX_BLOCKING_SYNC = CUctx_flags_enum.CU_CTX_BLOCKING_SYNC;
    enum CU_CTX_SCHED_MASK = CUctx_flags_enum.CU_CTX_SCHED_MASK;
    enum CU_CTX_MAP_HOST = CUctx_flags_enum.CU_CTX_MAP_HOST;
    enum CU_CTX_LMEM_RESIZE_TO_MAX = CUctx_flags_enum.CU_CTX_LMEM_RESIZE_TO_MAX;
    enum CU_CTX_FLAGS_MASK = CUctx_flags_enum.CU_CTX_FLAGS_MASK;
    alias CUctx_flags = CUctx_flags_enum;
    enum CUmemAttach_flags_enum
    {
        CU_MEM_ATTACH_GLOBAL = 1,
        CU_MEM_ATTACH_HOST = 2,
        CU_MEM_ATTACH_SINGLE = 4,
    }
    enum CU_MEM_ATTACH_GLOBAL = CUmemAttach_flags_enum.CU_MEM_ATTACH_GLOBAL;
    enum CU_MEM_ATTACH_HOST = CUmemAttach_flags_enum.CU_MEM_ATTACH_HOST;
    enum CU_MEM_ATTACH_SINGLE = CUmemAttach_flags_enum.CU_MEM_ATTACH_SINGLE;
    alias CUmemAttach_flags = CUmemAttach_flags_enum;
    enum CUipcMem_flags_enum
    {
        CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1,
    }
    enum CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = CUipcMem_flags_enum.CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS;
    alias CUipcMem_flags = CUipcMem_flags_enum;
    struct CUipcMemHandle_st
    {
        char[64] reserved;
    }
    alias CUipcMemHandle = CUipcMemHandle_st;
    struct CUipcEventHandle_st
    {
        char[64] reserved;
    }
    alias CUipcEventHandle = CUipcEventHandle_st;
    struct CUuuid_st
    {
        char[16] bytes;
    }
    alias CUuuid = CUuuid_st;
    struct CUgraphExec_st;
    alias CUgraphExec = CUgraphExec_st*;
    struct CUgraphNode_st;
    alias CUgraphNode = CUgraphNode_st*;
    struct CUgraph_st;
    alias CUgraph = CUgraph_st*;
    struct CUextSemaphore_st;
    alias CUexternalSemaphore = CUextSemaphore_st*;
    struct CUextMemory_st;
    alias CUexternalMemory = CUextMemory_st*;
    alias CUsurfObject = ulong;
    alias CUtexObject = ulong;
    struct CUgraphicsResource_st;
    alias CUgraphicsResource = CUgraphicsResource_st*;
    struct CUstream_st;
    alias CUstream = CUstream_st*;
    struct CUevent_st;
    alias CUevent = CUevent_st*;
    struct CUsurfref_st;
    alias CUsurfref = CUsurfref_st*;
    struct CUtexref_st;
    alias CUtexref = CUtexref_st*;
    struct CUmipmappedArray_st;
    alias CUmipmappedArray = CUmipmappedArray_st*;
    struct CUarray_st;
    alias CUdeviceptr = ulong;
    alias CUdevice = int;
    alias CUcontext = CUctx_st*;
    struct CUctx_st;
    alias CUmodule = CUmod_st*;
    struct CUmod_st;
    alias CUfunction = CUfunc_st*;
    struct CUfunc_st;
    alias CUarray = CUarray_st*;



    static if(!is(typeof(CUDA_VERSION))) {
        private enum enumMixinStr_CUDA_VERSION = `enum CUDA_VERSION = 10020;`;
        static if(is(typeof({ mixin(enumMixinStr_CUDA_VERSION); }))) {
            mixin(enumMixinStr_CUDA_VERSION);
        }
    }




    static if(!is(typeof(cuStreamBeginCapture))) {
        private enum enumMixinStr_cuStreamBeginCapture = `enum cuStreamBeginCapture = __CUDA_API_PTSZ ( cuStreamBeginCapture_v2 );`;
        static if(is(typeof({ mixin(enumMixinStr_cuStreamBeginCapture); }))) {
            mixin(enumMixinStr_cuStreamBeginCapture);
        }
    }




    static if(!is(typeof(cuGraphicsResourceSetMapFlags))) {
        private enum enumMixinStr_cuGraphicsResourceSetMapFlags = `enum cuGraphicsResourceSetMapFlags = cuGraphicsResourceSetMapFlags_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cuGraphicsResourceSetMapFlags); }))) {
            mixin(enumMixinStr_cuGraphicsResourceSetMapFlags);
        }
    }




    static if(!is(typeof(cuMemHostRegister))) {
        private enum enumMixinStr_cuMemHostRegister = `enum cuMemHostRegister = cuMemHostRegister_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cuMemHostRegister); }))) {
            mixin(enumMixinStr_cuMemHostRegister);
        }
    }




    static if(!is(typeof(cuLinkAddFile))) {
        private enum enumMixinStr_cuLinkAddFile = `enum cuLinkAddFile = cuLinkAddFile_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cuLinkAddFile); }))) {
            mixin(enumMixinStr_cuLinkAddFile);
        }
    }




    static if(!is(typeof(cuLinkAddData))) {
        private enum enumMixinStr_cuLinkAddData = `enum cuLinkAddData = cuLinkAddData_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cuLinkAddData); }))) {
            mixin(enumMixinStr_cuLinkAddData);
        }
    }




    static if(!is(typeof(cuLinkCreate))) {
        private enum enumMixinStr_cuLinkCreate = `enum cuLinkCreate = cuLinkCreate_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cuLinkCreate); }))) {
            mixin(enumMixinStr_cuLinkCreate);
        }
    }




    static if(!is(typeof(cuTexRefSetAddress2D))) {
        private enum enumMixinStr_cuTexRefSetAddress2D = `enum cuTexRefSetAddress2D = cuTexRefSetAddress2D_v3;`;
        static if(is(typeof({ mixin(enumMixinStr_cuTexRefSetAddress2D); }))) {
            mixin(enumMixinStr_cuTexRefSetAddress2D);
        }
    }




    static if(!is(typeof(cuEventDestroy))) {
        private enum enumMixinStr_cuEventDestroy = `enum cuEventDestroy = cuEventDestroy_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cuEventDestroy); }))) {
            mixin(enumMixinStr_cuEventDestroy);
        }
    }




    static if(!is(typeof(cuStreamDestroy))) {
        private enum enumMixinStr_cuStreamDestroy = `enum cuStreamDestroy = cuStreamDestroy_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cuStreamDestroy); }))) {
            mixin(enumMixinStr_cuStreamDestroy);
        }
    }




    static if(!is(typeof(cuCtxPushCurrent))) {
        private enum enumMixinStr_cuCtxPushCurrent = `enum cuCtxPushCurrent = cuCtxPushCurrent_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cuCtxPushCurrent); }))) {
            mixin(enumMixinStr_cuCtxPushCurrent);
        }
    }




    static if(!is(typeof(cuCtxPopCurrent))) {
        private enum enumMixinStr_cuCtxPopCurrent = `enum cuCtxPopCurrent = cuCtxPopCurrent_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cuCtxPopCurrent); }))) {
            mixin(enumMixinStr_cuCtxPopCurrent);
        }
    }




    static if(!is(typeof(cuCtxDestroy))) {
        private enum enumMixinStr_cuCtxDestroy = `enum cuCtxDestroy = cuCtxDestroy_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cuCtxDestroy); }))) {
            mixin(enumMixinStr_cuCtxDestroy);
        }
    }




    static if(!is(typeof(cuGraphicsResourceGetMappedPointer))) {
        private enum enumMixinStr_cuGraphicsResourceGetMappedPointer = `enum cuGraphicsResourceGetMappedPointer = cuGraphicsResourceGetMappedPointer_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cuGraphicsResourceGetMappedPointer); }))) {
            mixin(enumMixinStr_cuGraphicsResourceGetMappedPointer);
        }
    }




    static if(!is(typeof(cuTexRefGetAddress))) {
        private enum enumMixinStr_cuTexRefGetAddress = `enum cuTexRefGetAddress = cuTexRefGetAddress_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cuTexRefGetAddress); }))) {
            mixin(enumMixinStr_cuTexRefGetAddress);
        }
    }




    static if(!is(typeof(cuTexRefSetAddress))) {
        private enum enumMixinStr_cuTexRefSetAddress = `enum cuTexRefSetAddress = cuTexRefSetAddress_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cuTexRefSetAddress); }))) {
            mixin(enumMixinStr_cuTexRefSetAddress);
        }
    }




    static if(!is(typeof(cuArray3DGetDescriptor))) {
        private enum enumMixinStr_cuArray3DGetDescriptor = `enum cuArray3DGetDescriptor = cuArray3DGetDescriptor_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cuArray3DGetDescriptor); }))) {
            mixin(enumMixinStr_cuArray3DGetDescriptor);
        }
    }




    static if(!is(typeof(cuArray3DCreate))) {
        private enum enumMixinStr_cuArray3DCreate = `enum cuArray3DCreate = cuArray3DCreate_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cuArray3DCreate); }))) {
            mixin(enumMixinStr_cuArray3DCreate);
        }
    }




    static if(!is(typeof(cuArrayGetDescriptor))) {
        private enum enumMixinStr_cuArrayGetDescriptor = `enum cuArrayGetDescriptor = cuArrayGetDescriptor_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cuArrayGetDescriptor); }))) {
            mixin(enumMixinStr_cuArrayGetDescriptor);
        }
    }




    static if(!is(typeof(cuArrayCreate))) {
        private enum enumMixinStr_cuArrayCreate = `enum cuArrayCreate = cuArrayCreate_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cuArrayCreate); }))) {
            mixin(enumMixinStr_cuArrayCreate);
        }
    }




    static if(!is(typeof(cuMemsetD2D32))) {
        private enum enumMixinStr_cuMemsetD2D32 = `enum cuMemsetD2D32 = __CUDA_API_PTDS ( cuMemsetD2D32_v2 );`;
        static if(is(typeof({ mixin(enumMixinStr_cuMemsetD2D32); }))) {
            mixin(enumMixinStr_cuMemsetD2D32);
        }
    }




    static if(!is(typeof(cuMemsetD2D16))) {
        private enum enumMixinStr_cuMemsetD2D16 = `enum cuMemsetD2D16 = __CUDA_API_PTDS ( cuMemsetD2D16_v2 );`;
        static if(is(typeof({ mixin(enumMixinStr_cuMemsetD2D16); }))) {
            mixin(enumMixinStr_cuMemsetD2D16);
        }
    }




    static if(!is(typeof(cuMemsetD2D8))) {
        private enum enumMixinStr_cuMemsetD2D8 = `enum cuMemsetD2D8 = __CUDA_API_PTDS ( cuMemsetD2D8_v2 );`;
        static if(is(typeof({ mixin(enumMixinStr_cuMemsetD2D8); }))) {
            mixin(enumMixinStr_cuMemsetD2D8);
        }
    }




    static if(!is(typeof(cuMemsetD32))) {
        private enum enumMixinStr_cuMemsetD32 = `enum cuMemsetD32 = __CUDA_API_PTDS ( cuMemsetD32_v2 );`;
        static if(is(typeof({ mixin(enumMixinStr_cuMemsetD32); }))) {
            mixin(enumMixinStr_cuMemsetD32);
        }
    }




    static if(!is(typeof(cuMemsetD16))) {
        private enum enumMixinStr_cuMemsetD16 = `enum cuMemsetD16 = __CUDA_API_PTDS ( cuMemsetD16_v2 );`;
        static if(is(typeof({ mixin(enumMixinStr_cuMemsetD16); }))) {
            mixin(enumMixinStr_cuMemsetD16);
        }
    }






    static if(!is(typeof(cuMemsetD8))) {
        private enum enumMixinStr_cuMemsetD8 = `enum cuMemsetD8 = __CUDA_API_PTDS ( cuMemsetD8_v2 );`;
        static if(is(typeof({ mixin(enumMixinStr_cuMemsetD8); }))) {
            mixin(enumMixinStr_cuMemsetD8);
        }
    }




    static if(!is(typeof(cuMemcpy3DAsync))) {
        private enum enumMixinStr_cuMemcpy3DAsync = `enum cuMemcpy3DAsync = __CUDA_API_PTSZ ( cuMemcpy3DAsync_v2 );`;
        static if(is(typeof({ mixin(enumMixinStr_cuMemcpy3DAsync); }))) {
            mixin(enumMixinStr_cuMemcpy3DAsync);
        }
    }




    static if(!is(typeof(cuMemcpy2DAsync))) {
        private enum enumMixinStr_cuMemcpy2DAsync = `enum cuMemcpy2DAsync = __CUDA_API_PTSZ ( cuMemcpy2DAsync_v2 );`;
        static if(is(typeof({ mixin(enumMixinStr_cuMemcpy2DAsync); }))) {
            mixin(enumMixinStr_cuMemcpy2DAsync);
        }
    }




    static if(!is(typeof(CU_IPC_HANDLE_SIZE))) {
        private enum enumMixinStr_CU_IPC_HANDLE_SIZE = `enum CU_IPC_HANDLE_SIZE = 64;`;
        static if(is(typeof({ mixin(enumMixinStr_CU_IPC_HANDLE_SIZE); }))) {
            mixin(enumMixinStr_CU_IPC_HANDLE_SIZE);
        }
    }




    static if(!is(typeof(cuMemcpyDtoDAsync))) {
        private enum enumMixinStr_cuMemcpyDtoDAsync = `enum cuMemcpyDtoDAsync = __CUDA_API_PTSZ ( cuMemcpyDtoDAsync_v2 );`;
        static if(is(typeof({ mixin(enumMixinStr_cuMemcpyDtoDAsync); }))) {
            mixin(enumMixinStr_cuMemcpyDtoDAsync);
        }
    }




    static if(!is(typeof(cuMemcpyDtoHAsync))) {
        private enum enumMixinStr_cuMemcpyDtoHAsync = `enum cuMemcpyDtoHAsync = __CUDA_API_PTSZ ( cuMemcpyDtoHAsync_v2 );`;
        static if(is(typeof({ mixin(enumMixinStr_cuMemcpyDtoHAsync); }))) {
            mixin(enumMixinStr_cuMemcpyDtoHAsync);
        }
    }




    static if(!is(typeof(cuMemcpyHtoDAsync))) {
        private enum enumMixinStr_cuMemcpyHtoDAsync = `enum cuMemcpyHtoDAsync = __CUDA_API_PTSZ ( cuMemcpyHtoDAsync_v2 );`;
        static if(is(typeof({ mixin(enumMixinStr_cuMemcpyHtoDAsync); }))) {
            mixin(enumMixinStr_cuMemcpyHtoDAsync);
        }
    }




    static if(!is(typeof(cuMemcpy3D))) {
        private enum enumMixinStr_cuMemcpy3D = `enum cuMemcpy3D = __CUDA_API_PTDS ( cuMemcpy3D_v2 );`;
        static if(is(typeof({ mixin(enumMixinStr_cuMemcpy3D); }))) {
            mixin(enumMixinStr_cuMemcpy3D);
        }
    }




    static if(!is(typeof(cuMemcpy2DUnaligned))) {
        private enum enumMixinStr_cuMemcpy2DUnaligned = `enum cuMemcpy2DUnaligned = __CUDA_API_PTDS ( cuMemcpy2DUnaligned_v2 );`;
        static if(is(typeof({ mixin(enumMixinStr_cuMemcpy2DUnaligned); }))) {
            mixin(enumMixinStr_cuMemcpy2DUnaligned);
        }
    }




    static if(!is(typeof(cuMemcpy2D))) {
        private enum enumMixinStr_cuMemcpy2D = `enum cuMemcpy2D = __CUDA_API_PTDS ( cuMemcpy2D_v2 );`;
        static if(is(typeof({ mixin(enumMixinStr_cuMemcpy2D); }))) {
            mixin(enumMixinStr_cuMemcpy2D);
        }
    }




    static if(!is(typeof(cuMemcpyAtoHAsync))) {
        private enum enumMixinStr_cuMemcpyAtoHAsync = `enum cuMemcpyAtoHAsync = __CUDA_API_PTSZ ( cuMemcpyAtoHAsync_v2 );`;
        static if(is(typeof({ mixin(enumMixinStr_cuMemcpyAtoHAsync); }))) {
            mixin(enumMixinStr_cuMemcpyAtoHAsync);
        }
    }




    static if(!is(typeof(cuMemcpyHtoAAsync))) {
        private enum enumMixinStr_cuMemcpyHtoAAsync = `enum cuMemcpyHtoAAsync = __CUDA_API_PTSZ ( cuMemcpyHtoAAsync_v2 );`;
        static if(is(typeof({ mixin(enumMixinStr_cuMemcpyHtoAAsync); }))) {
            mixin(enumMixinStr_cuMemcpyHtoAAsync);
        }
    }




    static if(!is(typeof(cuMemcpyAtoA))) {
        private enum enumMixinStr_cuMemcpyAtoA = `enum cuMemcpyAtoA = __CUDA_API_PTDS ( cuMemcpyAtoA_v2 );`;
        static if(is(typeof({ mixin(enumMixinStr_cuMemcpyAtoA); }))) {
            mixin(enumMixinStr_cuMemcpyAtoA);
        }
    }




    static if(!is(typeof(cuMemcpyAtoH))) {
        private enum enumMixinStr_cuMemcpyAtoH = `enum cuMemcpyAtoH = __CUDA_API_PTDS ( cuMemcpyAtoH_v2 );`;
        static if(is(typeof({ mixin(enumMixinStr_cuMemcpyAtoH); }))) {
            mixin(enumMixinStr_cuMemcpyAtoH);
        }
    }




    static if(!is(typeof(cuMemcpyHtoA))) {
        private enum enumMixinStr_cuMemcpyHtoA = `enum cuMemcpyHtoA = __CUDA_API_PTDS ( cuMemcpyHtoA_v2 );`;
        static if(is(typeof({ mixin(enumMixinStr_cuMemcpyHtoA); }))) {
            mixin(enumMixinStr_cuMemcpyHtoA);
        }
    }




    static if(!is(typeof(cuMemcpyAtoD))) {
        private enum enumMixinStr_cuMemcpyAtoD = `enum cuMemcpyAtoD = __CUDA_API_PTDS ( cuMemcpyAtoD_v2 );`;
        static if(is(typeof({ mixin(enumMixinStr_cuMemcpyAtoD); }))) {
            mixin(enumMixinStr_cuMemcpyAtoD);
        }
    }




    static if(!is(typeof(cuMemcpyDtoA))) {
        private enum enumMixinStr_cuMemcpyDtoA = `enum cuMemcpyDtoA = __CUDA_API_PTDS ( cuMemcpyDtoA_v2 );`;
        static if(is(typeof({ mixin(enumMixinStr_cuMemcpyDtoA); }))) {
            mixin(enumMixinStr_cuMemcpyDtoA);
        }
    }




    static if(!is(typeof(cuMemcpyDtoD))) {
        private enum enumMixinStr_cuMemcpyDtoD = `enum cuMemcpyDtoD = __CUDA_API_PTDS ( cuMemcpyDtoD_v2 );`;
        static if(is(typeof({ mixin(enumMixinStr_cuMemcpyDtoD); }))) {
            mixin(enumMixinStr_cuMemcpyDtoD);
        }
    }




    static if(!is(typeof(CU_STREAM_LEGACY))) {
        private enum enumMixinStr_CU_STREAM_LEGACY = `enum CU_STREAM_LEGACY = ( cast( CUstream ) 0x1 );`;
        static if(is(typeof({ mixin(enumMixinStr_CU_STREAM_LEGACY); }))) {
            mixin(enumMixinStr_CU_STREAM_LEGACY);
        }
    }




    static if(!is(typeof(CU_STREAM_PER_THREAD))) {
        private enum enumMixinStr_CU_STREAM_PER_THREAD = `enum CU_STREAM_PER_THREAD = ( cast( CUstream ) 0x2 );`;
        static if(is(typeof({ mixin(enumMixinStr_CU_STREAM_PER_THREAD); }))) {
            mixin(enumMixinStr_CU_STREAM_PER_THREAD);
        }
    }




    static if(!is(typeof(cuMemcpyDtoH))) {
        private enum enumMixinStr_cuMemcpyDtoH = `enum cuMemcpyDtoH = __CUDA_API_PTDS ( cuMemcpyDtoH_v2 );`;
        static if(is(typeof({ mixin(enumMixinStr_cuMemcpyDtoH); }))) {
            mixin(enumMixinStr_cuMemcpyDtoH);
        }
    }




    static if(!is(typeof(cuMemcpyHtoD))) {
        private enum enumMixinStr_cuMemcpyHtoD = `enum cuMemcpyHtoD = __CUDA_API_PTDS ( cuMemcpyHtoD_v2 );`;
        static if(is(typeof({ mixin(enumMixinStr_cuMemcpyHtoD); }))) {
            mixin(enumMixinStr_cuMemcpyHtoD);
        }
    }




    static if(!is(typeof(cuMemHostGetDevicePointer))) {
        private enum enumMixinStr_cuMemHostGetDevicePointer = `enum cuMemHostGetDevicePointer = cuMemHostGetDevicePointer_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cuMemHostGetDevicePointer); }))) {
            mixin(enumMixinStr_cuMemHostGetDevicePointer);
        }
    }




    static if(!is(typeof(cuMemAllocHost))) {
        private enum enumMixinStr_cuMemAllocHost = `enum cuMemAllocHost = cuMemAllocHost_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cuMemAllocHost); }))) {
            mixin(enumMixinStr_cuMemAllocHost);
        }
    }




    static if(!is(typeof(cuMemGetAddressRange))) {
        private enum enumMixinStr_cuMemGetAddressRange = `enum cuMemGetAddressRange = cuMemGetAddressRange_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cuMemGetAddressRange); }))) {
            mixin(enumMixinStr_cuMemGetAddressRange);
        }
    }




    static if(!is(typeof(cuMemFree))) {
        private enum enumMixinStr_cuMemFree = `enum cuMemFree = cuMemFree_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cuMemFree); }))) {
            mixin(enumMixinStr_cuMemFree);
        }
    }




    static if(!is(typeof(cuMemAllocPitch))) {
        private enum enumMixinStr_cuMemAllocPitch = `enum cuMemAllocPitch = cuMemAllocPitch_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cuMemAllocPitch); }))) {
            mixin(enumMixinStr_cuMemAllocPitch);
        }
    }




    static if(!is(typeof(cuMemAlloc))) {
        private enum enumMixinStr_cuMemAlloc = `enum cuMemAlloc = cuMemAlloc_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cuMemAlloc); }))) {
            mixin(enumMixinStr_cuMemAlloc);
        }
    }




    static if(!is(typeof(cuMemGetInfo))) {
        private enum enumMixinStr_cuMemGetInfo = `enum cuMemGetInfo = cuMemGetInfo_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cuMemGetInfo); }))) {
            mixin(enumMixinStr_cuMemGetInfo);
        }
    }




    static if(!is(typeof(cuModuleGetGlobal))) {
        private enum enumMixinStr_cuModuleGetGlobal = `enum cuModuleGetGlobal = cuModuleGetGlobal_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cuModuleGetGlobal); }))) {
            mixin(enumMixinStr_cuModuleGetGlobal);
        }
    }




    static if(!is(typeof(cuCtxCreate))) {
        private enum enumMixinStr_cuCtxCreate = `enum cuCtxCreate = cuCtxCreate_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cuCtxCreate); }))) {
            mixin(enumMixinStr_cuCtxCreate);
        }
    }




    static if(!is(typeof(cuDeviceTotalMem))) {
        private enum enumMixinStr_cuDeviceTotalMem = `enum cuDeviceTotalMem = cuDeviceTotalMem_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cuDeviceTotalMem); }))) {
            mixin(enumMixinStr_cuDeviceTotalMem);
        }
    }
    static if(!is(typeof(__CUDA_API_VERSION))) {
        private enum enumMixinStr___CUDA_API_VERSION = `enum __CUDA_API_VERSION = 10020;`;
        static if(is(typeof({ mixin(enumMixinStr___CUDA_API_VERSION); }))) {
            mixin(enumMixinStr___CUDA_API_VERSION);
        }
    }




    static if(!is(typeof(__CUDA_DEPRECATED))) {
        private enum enumMixinStr___CUDA_DEPRECATED = `enum __CUDA_DEPRECATED = __attribute__ ( ( deprecated ) );`;
        static if(is(typeof({ mixin(enumMixinStr___CUDA_DEPRECATED); }))) {
            mixin(enumMixinStr___CUDA_DEPRECATED);
        }
    }
    static if(!is(typeof(__BIT_TYPES_DEFINED__))) {
        private enum enumMixinStr___BIT_TYPES_DEFINED__ = `enum __BIT_TYPES_DEFINED__ = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___BIT_TYPES_DEFINED__); }))) {
            mixin(enumMixinStr___BIT_TYPES_DEFINED__);
        }
    }
    static if(!is(typeof(_SYS_TYPES_H))) {
        private enum enumMixinStr__SYS_TYPES_H = `enum _SYS_TYPES_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__SYS_TYPES_H); }))) {
            mixin(enumMixinStr__SYS_TYPES_H);
        }
    }
    static if(!is(typeof(__SYSMACROS_IMPL_TEMPL))) {
        private enum enumMixinStr___SYSMACROS_IMPL_TEMPL = `enum __SYSMACROS_IMPL_TEMPL = ( rtype , name , proto ) __extension__ __extern_inline __attribute_const__ rtype __NTH ( gnu_dev_ ## name proto );`;
        static if(is(typeof({ mixin(enumMixinStr___SYSMACROS_IMPL_TEMPL); }))) {
            mixin(enumMixinStr___SYSMACROS_IMPL_TEMPL);
        }
    }




    static if(!is(typeof(__SYSMACROS_DECL_TEMPL))) {
        private enum enumMixinStr___SYSMACROS_DECL_TEMPL = `enum __SYSMACROS_DECL_TEMPL = ( rtype , name , proto ) extern rtype gnu_dev_ ## name proto __THROW __attribute_const__ ;;`;
        static if(is(typeof({ mixin(enumMixinStr___SYSMACROS_DECL_TEMPL); }))) {
            mixin(enumMixinStr___SYSMACROS_DECL_TEMPL);
        }
    }
    static if(!is(typeof(_SYS_SYSMACROS_H))) {
        private enum enumMixinStr__SYS_SYSMACROS_H = `enum _SYS_SYSMACROS_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__SYS_SYSMACROS_H); }))) {
            mixin(enumMixinStr__SYS_SYSMACROS_H);
        }
    }
    static if(!is(typeof(NFDBITS))) {
        private enum enumMixinStr_NFDBITS = `enum NFDBITS = __NFDBITS;`;
        static if(is(typeof({ mixin(enumMixinStr_NFDBITS); }))) {
            mixin(enumMixinStr_NFDBITS);
        }
    }




    static if(!is(typeof(FD_SETSIZE))) {
        private enum enumMixinStr_FD_SETSIZE = `enum FD_SETSIZE = __FD_SETSIZE;`;
        static if(is(typeof({ mixin(enumMixinStr_FD_SETSIZE); }))) {
            mixin(enumMixinStr_FD_SETSIZE);
        }
    }
    static if(!is(typeof(__NFDBITS))) {
        private enum enumMixinStr___NFDBITS = `enum __NFDBITS = ( 8 * cast( int ) ( __fd_mask ) .sizeof );`;
        static if(is(typeof({ mixin(enumMixinStr___NFDBITS); }))) {
            mixin(enumMixinStr___NFDBITS);
        }
    }






    static if(!is(typeof(_SYS_SELECT_H))) {
        private enum enumMixinStr__SYS_SELECT_H = `enum _SYS_SELECT_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__SYS_SELECT_H); }))) {
            mixin(enumMixinStr__SYS_SELECT_H);
        }
    }




    static if(!is(typeof(__HAVE_GENERIC_SELECTION))) {
        private enum enumMixinStr___HAVE_GENERIC_SELECTION = `enum __HAVE_GENERIC_SELECTION = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_GENERIC_SELECTION); }))) {
            mixin(enumMixinStr___HAVE_GENERIC_SELECTION);
        }
    }
    static if(!is(typeof(__restrict_arr))) {
        private enum enumMixinStr___restrict_arr = `enum __restrict_arr = __restrict;`;
        static if(is(typeof({ mixin(enumMixinStr___restrict_arr); }))) {
            mixin(enumMixinStr___restrict_arr);
        }
    }




    static if(!is(typeof(__fortify_function))) {
        private enum enumMixinStr___fortify_function = `enum __fortify_function = __extern_always_inline __attribute_artificial__;`;
        static if(is(typeof({ mixin(enumMixinStr___fortify_function); }))) {
            mixin(enumMixinStr___fortify_function);
        }
    }




    static if(!is(typeof(__extern_always_inline))) {
        private enum enumMixinStr___extern_always_inline = `enum __extern_always_inline = extern __always_inline __attribute__ ( ( __gnu_inline__ ) );`;
        static if(is(typeof({ mixin(enumMixinStr___extern_always_inline); }))) {
            mixin(enumMixinStr___extern_always_inline);
        }
    }




    static if(!is(typeof(__extern_inline))) {
        private enum enumMixinStr___extern_inline = `enum __extern_inline = extern __inline __attribute__ ( ( __gnu_inline__ ) );`;
        static if(is(typeof({ mixin(enumMixinStr___extern_inline); }))) {
            mixin(enumMixinStr___extern_inline);
        }
    }






    static if(!is(typeof(__always_inline))) {
        private enum enumMixinStr___always_inline = `enum __always_inline = __inline __attribute__ ( ( __always_inline__ ) );`;
        static if(is(typeof({ mixin(enumMixinStr___always_inline); }))) {
            mixin(enumMixinStr___always_inline);
        }
    }






    static if(!is(typeof(__attribute_warn_unused_result__))) {
        private enum enumMixinStr___attribute_warn_unused_result__ = `enum __attribute_warn_unused_result__ = __attribute__ ( ( __warn_unused_result__ ) );`;
        static if(is(typeof({ mixin(enumMixinStr___attribute_warn_unused_result__); }))) {
            mixin(enumMixinStr___attribute_warn_unused_result__);
        }
    }
    static if(!is(typeof(__attribute_deprecated__))) {
        private enum enumMixinStr___attribute_deprecated__ = `enum __attribute_deprecated__ = __attribute__ ( ( __deprecated__ ) );`;
        static if(is(typeof({ mixin(enumMixinStr___attribute_deprecated__); }))) {
            mixin(enumMixinStr___attribute_deprecated__);
        }
    }




    static if(!is(typeof(__attribute_noinline__))) {
        private enum enumMixinStr___attribute_noinline__ = `enum __attribute_noinline__ = __attribute__ ( ( __noinline__ ) );`;
        static if(is(typeof({ mixin(enumMixinStr___attribute_noinline__); }))) {
            mixin(enumMixinStr___attribute_noinline__);
        }
    }




    static if(!is(typeof(__attribute_used__))) {
        private enum enumMixinStr___attribute_used__ = `enum __attribute_used__ = __attribute__ ( ( __used__ ) );`;
        static if(is(typeof({ mixin(enumMixinStr___attribute_used__); }))) {
            mixin(enumMixinStr___attribute_used__);
        }
    }




    static if(!is(typeof(__attribute_const__))) {
        private enum enumMixinStr___attribute_const__ = `enum __attribute_const__ = __attribute__ ( cast( __const__ ) );`;
        static if(is(typeof({ mixin(enumMixinStr___attribute_const__); }))) {
            mixin(enumMixinStr___attribute_const__);
        }
    }




    static if(!is(typeof(CU_MEMHOSTALLOC_PORTABLE))) {
        private enum enumMixinStr_CU_MEMHOSTALLOC_PORTABLE = `enum CU_MEMHOSTALLOC_PORTABLE = 0x01;`;
        static if(is(typeof({ mixin(enumMixinStr_CU_MEMHOSTALLOC_PORTABLE); }))) {
            mixin(enumMixinStr_CU_MEMHOSTALLOC_PORTABLE);
        }
    }




    static if(!is(typeof(CU_MEMHOSTALLOC_DEVICEMAP))) {
        private enum enumMixinStr_CU_MEMHOSTALLOC_DEVICEMAP = `enum CU_MEMHOSTALLOC_DEVICEMAP = 0x02;`;
        static if(is(typeof({ mixin(enumMixinStr_CU_MEMHOSTALLOC_DEVICEMAP); }))) {
            mixin(enumMixinStr_CU_MEMHOSTALLOC_DEVICEMAP);
        }
    }




    static if(!is(typeof(CU_MEMHOSTALLOC_WRITECOMBINED))) {
        private enum enumMixinStr_CU_MEMHOSTALLOC_WRITECOMBINED = `enum CU_MEMHOSTALLOC_WRITECOMBINED = 0x04;`;
        static if(is(typeof({ mixin(enumMixinStr_CU_MEMHOSTALLOC_WRITECOMBINED); }))) {
            mixin(enumMixinStr_CU_MEMHOSTALLOC_WRITECOMBINED);
        }
    }




    static if(!is(typeof(CU_MEMHOSTREGISTER_PORTABLE))) {
        private enum enumMixinStr_CU_MEMHOSTREGISTER_PORTABLE = `enum CU_MEMHOSTREGISTER_PORTABLE = 0x01;`;
        static if(is(typeof({ mixin(enumMixinStr_CU_MEMHOSTREGISTER_PORTABLE); }))) {
            mixin(enumMixinStr_CU_MEMHOSTREGISTER_PORTABLE);
        }
    }




    static if(!is(typeof(CU_MEMHOSTREGISTER_DEVICEMAP))) {
        private enum enumMixinStr_CU_MEMHOSTREGISTER_DEVICEMAP = `enum CU_MEMHOSTREGISTER_DEVICEMAP = 0x02;`;
        static if(is(typeof({ mixin(enumMixinStr_CU_MEMHOSTREGISTER_DEVICEMAP); }))) {
            mixin(enumMixinStr_CU_MEMHOSTREGISTER_DEVICEMAP);
        }
    }




    static if(!is(typeof(CU_MEMHOSTREGISTER_IOMEMORY))) {
        private enum enumMixinStr_CU_MEMHOSTREGISTER_IOMEMORY = `enum CU_MEMHOSTREGISTER_IOMEMORY = 0x04;`;
        static if(is(typeof({ mixin(enumMixinStr_CU_MEMHOSTREGISTER_IOMEMORY); }))) {
            mixin(enumMixinStr_CU_MEMHOSTREGISTER_IOMEMORY);
        }
    }




    static if(!is(typeof(__attribute_pure__))) {
        private enum enumMixinStr___attribute_pure__ = `enum __attribute_pure__ = __attribute__ ( ( __pure__ ) );`;
        static if(is(typeof({ mixin(enumMixinStr___attribute_pure__); }))) {
            mixin(enumMixinStr___attribute_pure__);
        }
    }






    static if(!is(typeof(__attribute_malloc__))) {
        private enum enumMixinStr___attribute_malloc__ = `enum __attribute_malloc__ = __attribute__ ( ( __malloc__ ) );`;
        static if(is(typeof({ mixin(enumMixinStr___attribute_malloc__); }))) {
            mixin(enumMixinStr___attribute_malloc__);
        }
    }
    static if(!is(typeof(__glibc_c99_flexarr_available))) {
        private enum enumMixinStr___glibc_c99_flexarr_available = `enum __glibc_c99_flexarr_available = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___glibc_c99_flexarr_available); }))) {
            mixin(enumMixinStr___glibc_c99_flexarr_available);
        }
    }




    static if(!is(typeof(__flexarr))) {
        private enum enumMixinStr___flexarr = `enum __flexarr = [ ];`;
        static if(is(typeof({ mixin(enumMixinStr___flexarr); }))) {
            mixin(enumMixinStr___flexarr);
        }
    }
    static if(!is(typeof(__ptr_t))) {
        private enum enumMixinStr___ptr_t = `enum __ptr_t = void *;`;
        static if(is(typeof({ mixin(enumMixinStr___ptr_t); }))) {
            mixin(enumMixinStr___ptr_t);
        }
    }
    static if(!is(typeof(__THROWNL))) {
        private enum enumMixinStr___THROWNL = `enum __THROWNL = __attribute__ ( ( __nothrow__ ) );`;
        static if(is(typeof({ mixin(enumMixinStr___THROWNL); }))) {
            mixin(enumMixinStr___THROWNL);
        }
    }




    static if(!is(typeof(__THROW))) {
        private enum enumMixinStr___THROW = `enum __THROW = __attribute__ ( ( __nothrow__ __LEAF ) );`;
        static if(is(typeof({ mixin(enumMixinStr___THROW); }))) {
            mixin(enumMixinStr___THROW);
        }
    }






    static if(!is(typeof(CUDA_EXTERNAL_MEMORY_DEDICATED))) {
        private enum enumMixinStr_CUDA_EXTERNAL_MEMORY_DEDICATED = `enum CUDA_EXTERNAL_MEMORY_DEDICATED = 0x1;`;
        static if(is(typeof({ mixin(enumMixinStr_CUDA_EXTERNAL_MEMORY_DEDICATED); }))) {
            mixin(enumMixinStr_CUDA_EXTERNAL_MEMORY_DEDICATED);
        }
    }




    static if(!is(typeof(CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC))) {
        private enum enumMixinStr_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC = `enum CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC = 0x01;`;
        static if(is(typeof({ mixin(enumMixinStr_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC); }))) {
            mixin(enumMixinStr_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC);
        }
    }




    static if(!is(typeof(CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC))) {
        private enum enumMixinStr_CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC = `enum CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC = 0x02;`;
        static if(is(typeof({ mixin(enumMixinStr_CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC); }))) {
            mixin(enumMixinStr_CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC);
        }
    }




    static if(!is(typeof(CUDA_NVSCISYNC_ATTR_SIGNAL))) {
        private enum enumMixinStr_CUDA_NVSCISYNC_ATTR_SIGNAL = `enum CUDA_NVSCISYNC_ATTR_SIGNAL = 0x1;`;
        static if(is(typeof({ mixin(enumMixinStr_CUDA_NVSCISYNC_ATTR_SIGNAL); }))) {
            mixin(enumMixinStr_CUDA_NVSCISYNC_ATTR_SIGNAL);
        }
    }




    static if(!is(typeof(CUDA_NVSCISYNC_ATTR_WAIT))) {
        private enum enumMixinStr_CUDA_NVSCISYNC_ATTR_WAIT = `enum CUDA_NVSCISYNC_ATTR_WAIT = 0x2;`;
        static if(is(typeof({ mixin(enumMixinStr_CUDA_NVSCISYNC_ATTR_WAIT); }))) {
            mixin(enumMixinStr_CUDA_NVSCISYNC_ATTR_WAIT);
        }
    }






    static if(!is(typeof(_SYS_CDEFS_H))) {
        private enum enumMixinStr__SYS_CDEFS_H = `enum _SYS_CDEFS_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__SYS_CDEFS_H); }))) {
            mixin(enumMixinStr__SYS_CDEFS_H);
        }
    }
    static if(!is(typeof(__SYSCALL_WORDSIZE))) {
        private enum enumMixinStr___SYSCALL_WORDSIZE = `enum __SYSCALL_WORDSIZE = 64;`;
        static if(is(typeof({ mixin(enumMixinStr___SYSCALL_WORDSIZE); }))) {
            mixin(enumMixinStr___SYSCALL_WORDSIZE);
        }
    }




    static if(!is(typeof(__WORDSIZE_TIME64_COMPAT32))) {
        private enum enumMixinStr___WORDSIZE_TIME64_COMPAT32 = `enum __WORDSIZE_TIME64_COMPAT32 = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___WORDSIZE_TIME64_COMPAT32); }))) {
            mixin(enumMixinStr___WORDSIZE_TIME64_COMPAT32);
        }
    }




    static if(!is(typeof(__WORDSIZE))) {
        private enum enumMixinStr___WORDSIZE = `enum __WORDSIZE = 64;`;
        static if(is(typeof({ mixin(enumMixinStr___WORDSIZE); }))) {
            mixin(enumMixinStr___WORDSIZE);
        }
    }




    static if(!is(typeof(CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC))) {
        private enum enumMixinStr_CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC = `enum CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC = 0x01;`;
        static if(is(typeof({ mixin(enumMixinStr_CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC); }))) {
            mixin(enumMixinStr_CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC);
        }
    }




    static if(!is(typeof(CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC))) {
        private enum enumMixinStr_CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC = `enum CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC = 0x02;`;
        static if(is(typeof({ mixin(enumMixinStr_CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC); }))) {
            mixin(enumMixinStr_CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC);
        }
    }




    static if(!is(typeof(CUDA_ARRAY3D_LAYERED))) {
        private enum enumMixinStr_CUDA_ARRAY3D_LAYERED = `enum CUDA_ARRAY3D_LAYERED = 0x01;`;
        static if(is(typeof({ mixin(enumMixinStr_CUDA_ARRAY3D_LAYERED); }))) {
            mixin(enumMixinStr_CUDA_ARRAY3D_LAYERED);
        }
    }




    static if(!is(typeof(CUDA_ARRAY3D_2DARRAY))) {
        private enum enumMixinStr_CUDA_ARRAY3D_2DARRAY = `enum CUDA_ARRAY3D_2DARRAY = 0x01;`;
        static if(is(typeof({ mixin(enumMixinStr_CUDA_ARRAY3D_2DARRAY); }))) {
            mixin(enumMixinStr_CUDA_ARRAY3D_2DARRAY);
        }
    }




    static if(!is(typeof(CUDA_ARRAY3D_SURFACE_LDST))) {
        private enum enumMixinStr_CUDA_ARRAY3D_SURFACE_LDST = `enum CUDA_ARRAY3D_SURFACE_LDST = 0x02;`;
        static if(is(typeof({ mixin(enumMixinStr_CUDA_ARRAY3D_SURFACE_LDST); }))) {
            mixin(enumMixinStr_CUDA_ARRAY3D_SURFACE_LDST);
        }
    }




    static if(!is(typeof(CUDA_ARRAY3D_CUBEMAP))) {
        private enum enumMixinStr_CUDA_ARRAY3D_CUBEMAP = `enum CUDA_ARRAY3D_CUBEMAP = 0x04;`;
        static if(is(typeof({ mixin(enumMixinStr_CUDA_ARRAY3D_CUBEMAP); }))) {
            mixin(enumMixinStr_CUDA_ARRAY3D_CUBEMAP);
        }
    }




    static if(!is(typeof(CUDA_ARRAY3D_TEXTURE_GATHER))) {
        private enum enumMixinStr_CUDA_ARRAY3D_TEXTURE_GATHER = `enum CUDA_ARRAY3D_TEXTURE_GATHER = 0x08;`;
        static if(is(typeof({ mixin(enumMixinStr_CUDA_ARRAY3D_TEXTURE_GATHER); }))) {
            mixin(enumMixinStr_CUDA_ARRAY3D_TEXTURE_GATHER);
        }
    }




    static if(!is(typeof(CUDA_ARRAY3D_DEPTH_TEXTURE))) {
        private enum enumMixinStr_CUDA_ARRAY3D_DEPTH_TEXTURE = `enum CUDA_ARRAY3D_DEPTH_TEXTURE = 0x10;`;
        static if(is(typeof({ mixin(enumMixinStr_CUDA_ARRAY3D_DEPTH_TEXTURE); }))) {
            mixin(enumMixinStr_CUDA_ARRAY3D_DEPTH_TEXTURE);
        }
    }




    static if(!is(typeof(CUDA_ARRAY3D_COLOR_ATTACHMENT))) {
        private enum enumMixinStr_CUDA_ARRAY3D_COLOR_ATTACHMENT = `enum CUDA_ARRAY3D_COLOR_ATTACHMENT = 0x20;`;
        static if(is(typeof({ mixin(enumMixinStr_CUDA_ARRAY3D_COLOR_ATTACHMENT); }))) {
            mixin(enumMixinStr_CUDA_ARRAY3D_COLOR_ATTACHMENT);
        }
    }




    static if(!is(typeof(CU_TRSA_OVERRIDE_FORMAT))) {
        private enum enumMixinStr_CU_TRSA_OVERRIDE_FORMAT = `enum CU_TRSA_OVERRIDE_FORMAT = 0x01;`;
        static if(is(typeof({ mixin(enumMixinStr_CU_TRSA_OVERRIDE_FORMAT); }))) {
            mixin(enumMixinStr_CU_TRSA_OVERRIDE_FORMAT);
        }
    }




    static if(!is(typeof(CU_TRSF_READ_AS_INTEGER))) {
        private enum enumMixinStr_CU_TRSF_READ_AS_INTEGER = `enum CU_TRSF_READ_AS_INTEGER = 0x01;`;
        static if(is(typeof({ mixin(enumMixinStr_CU_TRSF_READ_AS_INTEGER); }))) {
            mixin(enumMixinStr_CU_TRSF_READ_AS_INTEGER);
        }
    }




    static if(!is(typeof(CU_TRSF_NORMALIZED_COORDINATES))) {
        private enum enumMixinStr_CU_TRSF_NORMALIZED_COORDINATES = `enum CU_TRSF_NORMALIZED_COORDINATES = 0x02;`;
        static if(is(typeof({ mixin(enumMixinStr_CU_TRSF_NORMALIZED_COORDINATES); }))) {
            mixin(enumMixinStr_CU_TRSF_NORMALIZED_COORDINATES);
        }
    }




    static if(!is(typeof(CU_TRSF_SRGB))) {
        private enum enumMixinStr_CU_TRSF_SRGB = `enum CU_TRSF_SRGB = 0x10;`;
        static if(is(typeof({ mixin(enumMixinStr_CU_TRSF_SRGB); }))) {
            mixin(enumMixinStr_CU_TRSF_SRGB);
        }
    }




    static if(!is(typeof(CU_LAUNCH_PARAM_END))) {
        private enum enumMixinStr_CU_LAUNCH_PARAM_END = `enum CU_LAUNCH_PARAM_END = ( cast( void * ) 0x00 );`;
        static if(is(typeof({ mixin(enumMixinStr_CU_LAUNCH_PARAM_END); }))) {
            mixin(enumMixinStr_CU_LAUNCH_PARAM_END);
        }
    }




    static if(!is(typeof(CU_LAUNCH_PARAM_BUFFER_POINTER))) {
        private enum enumMixinStr_CU_LAUNCH_PARAM_BUFFER_POINTER = `enum CU_LAUNCH_PARAM_BUFFER_POINTER = ( cast( void * ) 0x01 );`;
        static if(is(typeof({ mixin(enumMixinStr_CU_LAUNCH_PARAM_BUFFER_POINTER); }))) {
            mixin(enumMixinStr_CU_LAUNCH_PARAM_BUFFER_POINTER);
        }
    }




    static if(!is(typeof(CU_LAUNCH_PARAM_BUFFER_SIZE))) {
        private enum enumMixinStr_CU_LAUNCH_PARAM_BUFFER_SIZE = `enum CU_LAUNCH_PARAM_BUFFER_SIZE = ( cast( void * ) 0x02 );`;
        static if(is(typeof({ mixin(enumMixinStr_CU_LAUNCH_PARAM_BUFFER_SIZE); }))) {
            mixin(enumMixinStr_CU_LAUNCH_PARAM_BUFFER_SIZE);
        }
    }




    static if(!is(typeof(CU_PARAM_TR_DEFAULT))) {
        private enum enumMixinStr_CU_PARAM_TR_DEFAULT = `enum CU_PARAM_TR_DEFAULT = - 1;`;
        static if(is(typeof({ mixin(enumMixinStr_CU_PARAM_TR_DEFAULT); }))) {
            mixin(enumMixinStr_CU_PARAM_TR_DEFAULT);
        }
    }




    static if(!is(typeof(CU_DEVICE_CPU))) {
        private enum enumMixinStr_CU_DEVICE_CPU = `enum CU_DEVICE_CPU = ( cast( CUdevice ) - 1 );`;
        static if(is(typeof({ mixin(enumMixinStr_CU_DEVICE_CPU); }))) {
            mixin(enumMixinStr_CU_DEVICE_CPU);
        }
    }




    static if(!is(typeof(CU_DEVICE_INVALID))) {
        private enum enumMixinStr_CU_DEVICE_INVALID = `enum CU_DEVICE_INVALID = ( cast( CUdevice ) - 2 );`;
        static if(is(typeof({ mixin(enumMixinStr_CU_DEVICE_INVALID); }))) {
            mixin(enumMixinStr_CU_DEVICE_INVALID);
        }
    }






    static if(!is(typeof(__WCHAR_MIN))) {
        private enum enumMixinStr___WCHAR_MIN = `enum __WCHAR_MIN = ( - __WCHAR_MAX - 1 );`;
        static if(is(typeof({ mixin(enumMixinStr___WCHAR_MIN); }))) {
            mixin(enumMixinStr___WCHAR_MIN);
        }
    }




    static if(!is(typeof(__WCHAR_MAX))) {
        private enum enumMixinStr___WCHAR_MAX = `enum __WCHAR_MAX = 0x7fffffff;`;
        static if(is(typeof({ mixin(enumMixinStr___WCHAR_MAX); }))) {
            mixin(enumMixinStr___WCHAR_MAX);
        }
    }




    static if(!is(typeof(_BITS_WCHAR_H))) {
        private enum enumMixinStr__BITS_WCHAR_H = `enum _BITS_WCHAR_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__BITS_WCHAR_H); }))) {
            mixin(enumMixinStr__BITS_WCHAR_H);
        }
    }




    static if(!is(typeof(__WCOREFLAG))) {
        private enum enumMixinStr___WCOREFLAG = `enum __WCOREFLAG = 0x80;`;
        static if(is(typeof({ mixin(enumMixinStr___WCOREFLAG); }))) {
            mixin(enumMixinStr___WCOREFLAG);
        }
    }




    static if(!is(typeof(__W_CONTINUED))) {
        private enum enumMixinStr___W_CONTINUED = `enum __W_CONTINUED = 0xffff;`;
        static if(is(typeof({ mixin(enumMixinStr___W_CONTINUED); }))) {
            mixin(enumMixinStr___W_CONTINUED);
        }
    }
    static if(!is(typeof(__ENUM_IDTYPE_T))) {
        private enum enumMixinStr___ENUM_IDTYPE_T = `enum __ENUM_IDTYPE_T = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___ENUM_IDTYPE_T); }))) {
            mixin(enumMixinStr___ENUM_IDTYPE_T);
        }
    }




    static if(!is(typeof(__WCLONE))) {
        private enum enumMixinStr___WCLONE = `enum __WCLONE = 0x80000000;`;
        static if(is(typeof({ mixin(enumMixinStr___WCLONE); }))) {
            mixin(enumMixinStr___WCLONE);
        }
    }




    static if(!is(typeof(__WALL))) {
        private enum enumMixinStr___WALL = `enum __WALL = 0x40000000;`;
        static if(is(typeof({ mixin(enumMixinStr___WALL); }))) {
            mixin(enumMixinStr___WALL);
        }
    }




    static if(!is(typeof(__WNOTHREAD))) {
        private enum enumMixinStr___WNOTHREAD = `enum __WNOTHREAD = 0x20000000;`;
        static if(is(typeof({ mixin(enumMixinStr___WNOTHREAD); }))) {
            mixin(enumMixinStr___WNOTHREAD);
        }
    }




    static if(!is(typeof(WNOWAIT))) {
        private enum enumMixinStr_WNOWAIT = `enum WNOWAIT = 0x01000000;`;
        static if(is(typeof({ mixin(enumMixinStr_WNOWAIT); }))) {
            mixin(enumMixinStr_WNOWAIT);
        }
    }




    static if(!is(typeof(WCONTINUED))) {
        private enum enumMixinStr_WCONTINUED = `enum WCONTINUED = 8;`;
        static if(is(typeof({ mixin(enumMixinStr_WCONTINUED); }))) {
            mixin(enumMixinStr_WCONTINUED);
        }
    }




    static if(!is(typeof(WEXITED))) {
        private enum enumMixinStr_WEXITED = `enum WEXITED = 4;`;
        static if(is(typeof({ mixin(enumMixinStr_WEXITED); }))) {
            mixin(enumMixinStr_WEXITED);
        }
    }




    static if(!is(typeof(WSTOPPED))) {
        private enum enumMixinStr_WSTOPPED = `enum WSTOPPED = 2;`;
        static if(is(typeof({ mixin(enumMixinStr_WSTOPPED); }))) {
            mixin(enumMixinStr_WSTOPPED);
        }
    }




    static if(!is(typeof(WUNTRACED))) {
        private enum enumMixinStr_WUNTRACED = `enum WUNTRACED = 2;`;
        static if(is(typeof({ mixin(enumMixinStr_WUNTRACED); }))) {
            mixin(enumMixinStr_WUNTRACED);
        }
    }




    static if(!is(typeof(WNOHANG))) {
        private enum enumMixinStr_WNOHANG = `enum WNOHANG = 1;`;
        static if(is(typeof({ mixin(enumMixinStr_WNOHANG); }))) {
            mixin(enumMixinStr_WNOHANG);
        }
    }




    static if(!is(typeof(_BITS_UINTN_IDENTITY_H))) {
        private enum enumMixinStr__BITS_UINTN_IDENTITY_H = `enum _BITS_UINTN_IDENTITY_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__BITS_UINTN_IDENTITY_H); }))) {
            mixin(enumMixinStr__BITS_UINTN_IDENTITY_H);
        }
    }




    static if(!is(typeof(__FD_SETSIZE))) {
        private enum enumMixinStr___FD_SETSIZE = `enum __FD_SETSIZE = 1024;`;
        static if(is(typeof({ mixin(enumMixinStr___FD_SETSIZE); }))) {
            mixin(enumMixinStr___FD_SETSIZE);
        }
    }




    static if(!is(typeof(__RLIM_T_MATCHES_RLIM64_T))) {
        private enum enumMixinStr___RLIM_T_MATCHES_RLIM64_T = `enum __RLIM_T_MATCHES_RLIM64_T = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___RLIM_T_MATCHES_RLIM64_T); }))) {
            mixin(enumMixinStr___RLIM_T_MATCHES_RLIM64_T);
        }
    }




    static if(!is(typeof(__INO_T_MATCHES_INO64_T))) {
        private enum enumMixinStr___INO_T_MATCHES_INO64_T = `enum __INO_T_MATCHES_INO64_T = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___INO_T_MATCHES_INO64_T); }))) {
            mixin(enumMixinStr___INO_T_MATCHES_INO64_T);
        }
    }




    static if(!is(typeof(__OFF_T_MATCHES_OFF64_T))) {
        private enum enumMixinStr___OFF_T_MATCHES_OFF64_T = `enum __OFF_T_MATCHES_OFF64_T = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___OFF_T_MATCHES_OFF64_T); }))) {
            mixin(enumMixinStr___OFF_T_MATCHES_OFF64_T);
        }
    }




    static if(!is(typeof(__CPU_MASK_TYPE))) {
        private enum enumMixinStr___CPU_MASK_TYPE = `enum __CPU_MASK_TYPE = __SYSCALL_ULONG_TYPE;`;
        static if(is(typeof({ mixin(enumMixinStr___CPU_MASK_TYPE); }))) {
            mixin(enumMixinStr___CPU_MASK_TYPE);
        }
    }




    static if(!is(typeof(__SSIZE_T_TYPE))) {
        private enum enumMixinStr___SSIZE_T_TYPE = `enum __SSIZE_T_TYPE = __SWORD_TYPE;`;
        static if(is(typeof({ mixin(enumMixinStr___SSIZE_T_TYPE); }))) {
            mixin(enumMixinStr___SSIZE_T_TYPE);
        }
    }




    static if(!is(typeof(__FSID_T_TYPE))) {
        private enum enumMixinStr___FSID_T_TYPE = `enum __FSID_T_TYPE = { int __val [ 2 ] ; };`;
        static if(is(typeof({ mixin(enumMixinStr___FSID_T_TYPE); }))) {
            mixin(enumMixinStr___FSID_T_TYPE);
        }
    }




    static if(!is(typeof(__BLKSIZE_T_TYPE))) {
        private enum enumMixinStr___BLKSIZE_T_TYPE = `enum __BLKSIZE_T_TYPE = __SYSCALL_SLONG_TYPE;`;
        static if(is(typeof({ mixin(enumMixinStr___BLKSIZE_T_TYPE); }))) {
            mixin(enumMixinStr___BLKSIZE_T_TYPE);
        }
    }




    static if(!is(typeof(__TIMER_T_TYPE))) {
        private enum enumMixinStr___TIMER_T_TYPE = `enum __TIMER_T_TYPE = void *;`;
        static if(is(typeof({ mixin(enumMixinStr___TIMER_T_TYPE); }))) {
            mixin(enumMixinStr___TIMER_T_TYPE);
        }
    }




    static if(!is(typeof(__CLOCKID_T_TYPE))) {
        private enum enumMixinStr___CLOCKID_T_TYPE = `enum __CLOCKID_T_TYPE = __S32_TYPE;`;
        static if(is(typeof({ mixin(enumMixinStr___CLOCKID_T_TYPE); }))) {
            mixin(enumMixinStr___CLOCKID_T_TYPE);
        }
    }




    static if(!is(typeof(__KEY_T_TYPE))) {
        private enum enumMixinStr___KEY_T_TYPE = `enum __KEY_T_TYPE = __S32_TYPE;`;
        static if(is(typeof({ mixin(enumMixinStr___KEY_T_TYPE); }))) {
            mixin(enumMixinStr___KEY_T_TYPE);
        }
    }




    static if(!is(typeof(__DADDR_T_TYPE))) {
        private enum enumMixinStr___DADDR_T_TYPE = `enum __DADDR_T_TYPE = __S32_TYPE;`;
        static if(is(typeof({ mixin(enumMixinStr___DADDR_T_TYPE); }))) {
            mixin(enumMixinStr___DADDR_T_TYPE);
        }
    }




    static if(!is(typeof(__SUSECONDS_T_TYPE))) {
        private enum enumMixinStr___SUSECONDS_T_TYPE = `enum __SUSECONDS_T_TYPE = __SYSCALL_SLONG_TYPE;`;
        static if(is(typeof({ mixin(enumMixinStr___SUSECONDS_T_TYPE); }))) {
            mixin(enumMixinStr___SUSECONDS_T_TYPE);
        }
    }




    static if(!is(typeof(__USECONDS_T_TYPE))) {
        private enum enumMixinStr___USECONDS_T_TYPE = `enum __USECONDS_T_TYPE = __U32_TYPE;`;
        static if(is(typeof({ mixin(enumMixinStr___USECONDS_T_TYPE); }))) {
            mixin(enumMixinStr___USECONDS_T_TYPE);
        }
    }




    static if(!is(typeof(__TIME_T_TYPE))) {
        private enum enumMixinStr___TIME_T_TYPE = `enum __TIME_T_TYPE = __SYSCALL_SLONG_TYPE;`;
        static if(is(typeof({ mixin(enumMixinStr___TIME_T_TYPE); }))) {
            mixin(enumMixinStr___TIME_T_TYPE);
        }
    }




    static if(!is(typeof(__CLOCK_T_TYPE))) {
        private enum enumMixinStr___CLOCK_T_TYPE = `enum __CLOCK_T_TYPE = __SYSCALL_SLONG_TYPE;`;
        static if(is(typeof({ mixin(enumMixinStr___CLOCK_T_TYPE); }))) {
            mixin(enumMixinStr___CLOCK_T_TYPE);
        }
    }




    static if(!is(typeof(__ID_T_TYPE))) {
        private enum enumMixinStr___ID_T_TYPE = `enum __ID_T_TYPE = __U32_TYPE;`;
        static if(is(typeof({ mixin(enumMixinStr___ID_T_TYPE); }))) {
            mixin(enumMixinStr___ID_T_TYPE);
        }
    }




    static if(!is(typeof(__FSFILCNT64_T_TYPE))) {
        private enum enumMixinStr___FSFILCNT64_T_TYPE = `enum __FSFILCNT64_T_TYPE = __UQUAD_TYPE;`;
        static if(is(typeof({ mixin(enumMixinStr___FSFILCNT64_T_TYPE); }))) {
            mixin(enumMixinStr___FSFILCNT64_T_TYPE);
        }
    }




    static if(!is(typeof(__FSFILCNT_T_TYPE))) {
        private enum enumMixinStr___FSFILCNT_T_TYPE = `enum __FSFILCNT_T_TYPE = __SYSCALL_ULONG_TYPE;`;
        static if(is(typeof({ mixin(enumMixinStr___FSFILCNT_T_TYPE); }))) {
            mixin(enumMixinStr___FSFILCNT_T_TYPE);
        }
    }




    static if(!is(typeof(__FSBLKCNT64_T_TYPE))) {
        private enum enumMixinStr___FSBLKCNT64_T_TYPE = `enum __FSBLKCNT64_T_TYPE = __UQUAD_TYPE;`;
        static if(is(typeof({ mixin(enumMixinStr___FSBLKCNT64_T_TYPE); }))) {
            mixin(enumMixinStr___FSBLKCNT64_T_TYPE);
        }
    }




    static if(!is(typeof(__FSBLKCNT_T_TYPE))) {
        private enum enumMixinStr___FSBLKCNT_T_TYPE = `enum __FSBLKCNT_T_TYPE = __SYSCALL_ULONG_TYPE;`;
        static if(is(typeof({ mixin(enumMixinStr___FSBLKCNT_T_TYPE); }))) {
            mixin(enumMixinStr___FSBLKCNT_T_TYPE);
        }
    }




    static if(!is(typeof(__BLKCNT64_T_TYPE))) {
        private enum enumMixinStr___BLKCNT64_T_TYPE = `enum __BLKCNT64_T_TYPE = __SQUAD_TYPE;`;
        static if(is(typeof({ mixin(enumMixinStr___BLKCNT64_T_TYPE); }))) {
            mixin(enumMixinStr___BLKCNT64_T_TYPE);
        }
    }




    static if(!is(typeof(__BLKCNT_T_TYPE))) {
        private enum enumMixinStr___BLKCNT_T_TYPE = `enum __BLKCNT_T_TYPE = __SYSCALL_SLONG_TYPE;`;
        static if(is(typeof({ mixin(enumMixinStr___BLKCNT_T_TYPE); }))) {
            mixin(enumMixinStr___BLKCNT_T_TYPE);
        }
    }




    static if(!is(typeof(__RLIM64_T_TYPE))) {
        private enum enumMixinStr___RLIM64_T_TYPE = `enum __RLIM64_T_TYPE = __UQUAD_TYPE;`;
        static if(is(typeof({ mixin(enumMixinStr___RLIM64_T_TYPE); }))) {
            mixin(enumMixinStr___RLIM64_T_TYPE);
        }
    }




    static if(!is(typeof(__RLIM_T_TYPE))) {
        private enum enumMixinStr___RLIM_T_TYPE = `enum __RLIM_T_TYPE = __SYSCALL_ULONG_TYPE;`;
        static if(is(typeof({ mixin(enumMixinStr___RLIM_T_TYPE); }))) {
            mixin(enumMixinStr___RLIM_T_TYPE);
        }
    }




    static if(!is(typeof(__PID_T_TYPE))) {
        private enum enumMixinStr___PID_T_TYPE = `enum __PID_T_TYPE = __S32_TYPE;`;
        static if(is(typeof({ mixin(enumMixinStr___PID_T_TYPE); }))) {
            mixin(enumMixinStr___PID_T_TYPE);
        }
    }




    static if(!is(typeof(__OFF64_T_TYPE))) {
        private enum enumMixinStr___OFF64_T_TYPE = `enum __OFF64_T_TYPE = __SQUAD_TYPE;`;
        static if(is(typeof({ mixin(enumMixinStr___OFF64_T_TYPE); }))) {
            mixin(enumMixinStr___OFF64_T_TYPE);
        }
    }




    static if(!is(typeof(__OFF_T_TYPE))) {
        private enum enumMixinStr___OFF_T_TYPE = `enum __OFF_T_TYPE = __SYSCALL_SLONG_TYPE;`;
        static if(is(typeof({ mixin(enumMixinStr___OFF_T_TYPE); }))) {
            mixin(enumMixinStr___OFF_T_TYPE);
        }
    }




    static if(!is(typeof(__FSWORD_T_TYPE))) {
        private enum enumMixinStr___FSWORD_T_TYPE = `enum __FSWORD_T_TYPE = __SYSCALL_SLONG_TYPE;`;
        static if(is(typeof({ mixin(enumMixinStr___FSWORD_T_TYPE); }))) {
            mixin(enumMixinStr___FSWORD_T_TYPE);
        }
    }




    static if(!is(typeof(__NLINK_T_TYPE))) {
        private enum enumMixinStr___NLINK_T_TYPE = `enum __NLINK_T_TYPE = __SYSCALL_ULONG_TYPE;`;
        static if(is(typeof({ mixin(enumMixinStr___NLINK_T_TYPE); }))) {
            mixin(enumMixinStr___NLINK_T_TYPE);
        }
    }




    static if(!is(typeof(__MODE_T_TYPE))) {
        private enum enumMixinStr___MODE_T_TYPE = `enum __MODE_T_TYPE = __U32_TYPE;`;
        static if(is(typeof({ mixin(enumMixinStr___MODE_T_TYPE); }))) {
            mixin(enumMixinStr___MODE_T_TYPE);
        }
    }




    static if(!is(typeof(__INO64_T_TYPE))) {
        private enum enumMixinStr___INO64_T_TYPE = `enum __INO64_T_TYPE = __UQUAD_TYPE;`;
        static if(is(typeof({ mixin(enumMixinStr___INO64_T_TYPE); }))) {
            mixin(enumMixinStr___INO64_T_TYPE);
        }
    }




    static if(!is(typeof(__INO_T_TYPE))) {
        private enum enumMixinStr___INO_T_TYPE = `enum __INO_T_TYPE = __SYSCALL_ULONG_TYPE;`;
        static if(is(typeof({ mixin(enumMixinStr___INO_T_TYPE); }))) {
            mixin(enumMixinStr___INO_T_TYPE);
        }
    }




    static if(!is(typeof(__GID_T_TYPE))) {
        private enum enumMixinStr___GID_T_TYPE = `enum __GID_T_TYPE = __U32_TYPE;`;
        static if(is(typeof({ mixin(enumMixinStr___GID_T_TYPE); }))) {
            mixin(enumMixinStr___GID_T_TYPE);
        }
    }




    static if(!is(typeof(__UID_T_TYPE))) {
        private enum enumMixinStr___UID_T_TYPE = `enum __UID_T_TYPE = __U32_TYPE;`;
        static if(is(typeof({ mixin(enumMixinStr___UID_T_TYPE); }))) {
            mixin(enumMixinStr___UID_T_TYPE);
        }
    }




    static if(!is(typeof(__DEV_T_TYPE))) {
        private enum enumMixinStr___DEV_T_TYPE = `enum __DEV_T_TYPE = __UQUAD_TYPE;`;
        static if(is(typeof({ mixin(enumMixinStr___DEV_T_TYPE); }))) {
            mixin(enumMixinStr___DEV_T_TYPE);
        }
    }




    static if(!is(typeof(__SYSCALL_ULONG_TYPE))) {
        private enum enumMixinStr___SYSCALL_ULONG_TYPE = `enum __SYSCALL_ULONG_TYPE = __ULONGWORD_TYPE;`;
        static if(is(typeof({ mixin(enumMixinStr___SYSCALL_ULONG_TYPE); }))) {
            mixin(enumMixinStr___SYSCALL_ULONG_TYPE);
        }
    }




    static if(!is(typeof(__SYSCALL_SLONG_TYPE))) {
        private enum enumMixinStr___SYSCALL_SLONG_TYPE = `enum __SYSCALL_SLONG_TYPE = __SLONGWORD_TYPE;`;
        static if(is(typeof({ mixin(enumMixinStr___SYSCALL_SLONG_TYPE); }))) {
            mixin(enumMixinStr___SYSCALL_SLONG_TYPE);
        }
    }




    static if(!is(typeof(_BITS_TYPESIZES_H))) {
        private enum enumMixinStr__BITS_TYPESIZES_H = `enum _BITS_TYPESIZES_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__BITS_TYPESIZES_H); }))) {
            mixin(enumMixinStr__BITS_TYPESIZES_H);
        }
    }




    static if(!is(typeof(__timer_t_defined))) {
        private enum enumMixinStr___timer_t_defined = `enum __timer_t_defined = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___timer_t_defined); }))) {
            mixin(enumMixinStr___timer_t_defined);
        }
    }




    static if(!is(typeof(__time_t_defined))) {
        private enum enumMixinStr___time_t_defined = `enum __time_t_defined = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___time_t_defined); }))) {
            mixin(enumMixinStr___time_t_defined);
        }
    }




    static if(!is(typeof(__timeval_defined))) {
        private enum enumMixinStr___timeval_defined = `enum __timeval_defined = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___timeval_defined); }))) {
            mixin(enumMixinStr___timeval_defined);
        }
    }




    static if(!is(typeof(__timespec_defined))) {
        private enum enumMixinStr___timespec_defined = `enum __timespec_defined = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___timespec_defined); }))) {
            mixin(enumMixinStr___timespec_defined);
        }
    }




    static if(!is(typeof(__sigset_t_defined))) {
        private enum enumMixinStr___sigset_t_defined = `enum __sigset_t_defined = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___sigset_t_defined); }))) {
            mixin(enumMixinStr___sigset_t_defined);
        }
    }




    static if(!is(typeof(__clockid_t_defined))) {
        private enum enumMixinStr___clockid_t_defined = `enum __clockid_t_defined = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___clockid_t_defined); }))) {
            mixin(enumMixinStr___clockid_t_defined);
        }
    }




    static if(!is(typeof(__clock_t_defined))) {
        private enum enumMixinStr___clock_t_defined = `enum __clock_t_defined = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___clock_t_defined); }))) {
            mixin(enumMixinStr___clock_t_defined);
        }
    }




    static if(!is(typeof(_SIGSET_NWORDS))) {
        private enum enumMixinStr__SIGSET_NWORDS = `enum _SIGSET_NWORDS = ( 1024 / ( 8 * ( unsigned long int ) .sizeof ) );`;
        static if(is(typeof({ mixin(enumMixinStr__SIGSET_NWORDS); }))) {
            mixin(enumMixinStr__SIGSET_NWORDS);
        }
    }






    static if(!is(typeof(__STD_TYPE))) {
        private enum enumMixinStr___STD_TYPE = `enum __STD_TYPE = typedef;`;
        static if(is(typeof({ mixin(enumMixinStr___STD_TYPE); }))) {
            mixin(enumMixinStr___STD_TYPE);
        }
    }




    static if(!is(typeof(__U64_TYPE))) {
        private enum enumMixinStr___U64_TYPE = `enum __U64_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___U64_TYPE); }))) {
            mixin(enumMixinStr___U64_TYPE);
        }
    }




    static if(!is(typeof(__S64_TYPE))) {
        private enum enumMixinStr___S64_TYPE = `enum __S64_TYPE = long int;`;
        static if(is(typeof({ mixin(enumMixinStr___S64_TYPE); }))) {
            mixin(enumMixinStr___S64_TYPE);
        }
    }




    static if(!is(typeof(__ULONG32_TYPE))) {
        private enum enumMixinStr___ULONG32_TYPE = `enum __ULONG32_TYPE = unsigned int;`;
        static if(is(typeof({ mixin(enumMixinStr___ULONG32_TYPE); }))) {
            mixin(enumMixinStr___ULONG32_TYPE);
        }
    }




    static if(!is(typeof(__SLONG32_TYPE))) {
        private enum enumMixinStr___SLONG32_TYPE = `enum __SLONG32_TYPE = int;`;
        static if(is(typeof({ mixin(enumMixinStr___SLONG32_TYPE); }))) {
            mixin(enumMixinStr___SLONG32_TYPE);
        }
    }




    static if(!is(typeof(__UWORD_TYPE))) {
        private enum enumMixinStr___UWORD_TYPE = `enum __UWORD_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___UWORD_TYPE); }))) {
            mixin(enumMixinStr___UWORD_TYPE);
        }
    }




    static if(!is(typeof(__SWORD_TYPE))) {
        private enum enumMixinStr___SWORD_TYPE = `enum __SWORD_TYPE = long int;`;
        static if(is(typeof({ mixin(enumMixinStr___SWORD_TYPE); }))) {
            mixin(enumMixinStr___SWORD_TYPE);
        }
    }




    static if(!is(typeof(__UQUAD_TYPE))) {
        private enum enumMixinStr___UQUAD_TYPE = `enum __UQUAD_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___UQUAD_TYPE); }))) {
            mixin(enumMixinStr___UQUAD_TYPE);
        }
    }




    static if(!is(typeof(__SQUAD_TYPE))) {
        private enum enumMixinStr___SQUAD_TYPE = `enum __SQUAD_TYPE = long int;`;
        static if(is(typeof({ mixin(enumMixinStr___SQUAD_TYPE); }))) {
            mixin(enumMixinStr___SQUAD_TYPE);
        }
    }




    static if(!is(typeof(__ULONGWORD_TYPE))) {
        private enum enumMixinStr___ULONGWORD_TYPE = `enum __ULONGWORD_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___ULONGWORD_TYPE); }))) {
            mixin(enumMixinStr___ULONGWORD_TYPE);
        }
    }




    static if(!is(typeof(__SLONGWORD_TYPE))) {
        private enum enumMixinStr___SLONGWORD_TYPE = `enum __SLONGWORD_TYPE = long int;`;
        static if(is(typeof({ mixin(enumMixinStr___SLONGWORD_TYPE); }))) {
            mixin(enumMixinStr___SLONGWORD_TYPE);
        }
    }




    static if(!is(typeof(__U32_TYPE))) {
        private enum enumMixinStr___U32_TYPE = `enum __U32_TYPE = unsigned int;`;
        static if(is(typeof({ mixin(enumMixinStr___U32_TYPE); }))) {
            mixin(enumMixinStr___U32_TYPE);
        }
    }




    static if(!is(typeof(__S32_TYPE))) {
        private enum enumMixinStr___S32_TYPE = `enum __S32_TYPE = int;`;
        static if(is(typeof({ mixin(enumMixinStr___S32_TYPE); }))) {
            mixin(enumMixinStr___S32_TYPE);
        }
    }




    static if(!is(typeof(__U16_TYPE))) {
        private enum enumMixinStr___U16_TYPE = `enum __U16_TYPE = unsigned short int;`;
        static if(is(typeof({ mixin(enumMixinStr___U16_TYPE); }))) {
            mixin(enumMixinStr___U16_TYPE);
        }
    }




    static if(!is(typeof(__S16_TYPE))) {
        private enum enumMixinStr___S16_TYPE = `enum __S16_TYPE = short int;`;
        static if(is(typeof({ mixin(enumMixinStr___S16_TYPE); }))) {
            mixin(enumMixinStr___S16_TYPE);
        }
    }




    static if(!is(typeof(_BITS_TYPES_H))) {
        private enum enumMixinStr__BITS_TYPES_H = `enum _BITS_TYPES_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__BITS_TYPES_H); }))) {
            mixin(enumMixinStr__BITS_TYPES_H);
        }
    }




    static if(!is(typeof(__PTHREAD_MUTEX_HAVE_PREV))) {
        private enum enumMixinStr___PTHREAD_MUTEX_HAVE_PREV = `enum __PTHREAD_MUTEX_HAVE_PREV = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___PTHREAD_MUTEX_HAVE_PREV); }))) {
            mixin(enumMixinStr___PTHREAD_MUTEX_HAVE_PREV);
        }
    }




    static if(!is(typeof(__PTHREAD_SPINS))) {
        private enum enumMixinStr___PTHREAD_SPINS = `enum __PTHREAD_SPINS = 0 , 0;`;
        static if(is(typeof({ mixin(enumMixinStr___PTHREAD_SPINS); }))) {
            mixin(enumMixinStr___PTHREAD_SPINS);
        }
    }




    static if(!is(typeof(__PTHREAD_SPINS_DATA))) {
        private enum enumMixinStr___PTHREAD_SPINS_DATA = `enum __PTHREAD_SPINS_DATA = short __spins ; short __elision;`;
        static if(is(typeof({ mixin(enumMixinStr___PTHREAD_SPINS_DATA); }))) {
            mixin(enumMixinStr___PTHREAD_SPINS_DATA);
        }
    }




    static if(!is(typeof(_THREAD_SHARED_TYPES_H))) {
        private enum enumMixinStr__THREAD_SHARED_TYPES_H = `enum _THREAD_SHARED_TYPES_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__THREAD_SHARED_TYPES_H); }))) {
            mixin(enumMixinStr__THREAD_SHARED_TYPES_H);
        }
    }




    static if(!is(typeof(__SYSMACROS_DEFINE_MAKEDEV))) {
        private enum enumMixinStr___SYSMACROS_DEFINE_MAKEDEV = `enum __SYSMACROS_DEFINE_MAKEDEV = ( DECL_TEMPL ) __SYSMACROS_DECLARE_MAKEDEV ( DECL_TEMPL ) { __dev_t __dev ; __dev = ( ( cast( __dev_t ) ( __major & 0x00000fffu ) ) << 8 ) ; __dev |= ( ( cast( __dev_t ) ( __major & 0xfffff000u ) ) << 32 ) ; __dev |= ( ( cast( __dev_t ) ( __minor & 0x000000ffu ) ) << 0 ) ; __dev |= ( ( cast( __dev_t ) ( __minor & 0xffffff00u ) ) << 12 ) ; return __dev ; };`;
        static if(is(typeof({ mixin(enumMixinStr___SYSMACROS_DEFINE_MAKEDEV); }))) {
            mixin(enumMixinStr___SYSMACROS_DEFINE_MAKEDEV);
        }
    }




    static if(!is(typeof(__SYSMACROS_DECLARE_MAKEDEV))) {
        private enum enumMixinStr___SYSMACROS_DECLARE_MAKEDEV = `enum __SYSMACROS_DECLARE_MAKEDEV = ( DECL_TEMPL ) DECL_TEMPL ( __dev_t , makedev , ( unsigned int __major , unsigned int __minor ) );`;
        static if(is(typeof({ mixin(enumMixinStr___SYSMACROS_DECLARE_MAKEDEV); }))) {
            mixin(enumMixinStr___SYSMACROS_DECLARE_MAKEDEV);
        }
    }




    static if(!is(typeof(__SYSMACROS_DEFINE_MINOR))) {
        private enum enumMixinStr___SYSMACROS_DEFINE_MINOR = `enum __SYSMACROS_DEFINE_MINOR = ( DECL_TEMPL ) __SYSMACROS_DECLARE_MINOR ( DECL_TEMPL ) { unsigned int __minor ; __minor = ( ( __dev & cast( __dev_t ) 0x00000000000000ffu ) >> 0 ) ; __minor |= ( ( __dev & cast( __dev_t ) 0x00000ffffff00000u ) >> 12 ) ; return __minor ; };`;
        static if(is(typeof({ mixin(enumMixinStr___SYSMACROS_DEFINE_MINOR); }))) {
            mixin(enumMixinStr___SYSMACROS_DEFINE_MINOR);
        }
    }




    static if(!is(typeof(__SYSMACROS_DECLARE_MINOR))) {
        private enum enumMixinStr___SYSMACROS_DECLARE_MINOR = `enum __SYSMACROS_DECLARE_MINOR = ( DECL_TEMPL ) DECL_TEMPL ( unsigned int , minor , ( __dev_t __dev ) );`;
        static if(is(typeof({ mixin(enumMixinStr___SYSMACROS_DECLARE_MINOR); }))) {
            mixin(enumMixinStr___SYSMACROS_DECLARE_MINOR);
        }
    }




    static if(!is(typeof(__SYSMACROS_DEFINE_MAJOR))) {
        private enum enumMixinStr___SYSMACROS_DEFINE_MAJOR = `enum __SYSMACROS_DEFINE_MAJOR = ( DECL_TEMPL ) __SYSMACROS_DECLARE_MAJOR ( DECL_TEMPL ) { unsigned int __major ; __major = ( ( __dev & cast( __dev_t ) 0x00000000000fff00u ) >> 8 ) ; __major |= ( ( __dev & cast( __dev_t ) 0xfffff00000000000u ) >> 32 ) ; return __major ; };`;
        static if(is(typeof({ mixin(enumMixinStr___SYSMACROS_DEFINE_MAJOR); }))) {
            mixin(enumMixinStr___SYSMACROS_DEFINE_MAJOR);
        }
    }




    static if(!is(typeof(__SYSMACROS_DECLARE_MAJOR))) {
        private enum enumMixinStr___SYSMACROS_DECLARE_MAJOR = `enum __SYSMACROS_DECLARE_MAJOR = ( DECL_TEMPL ) DECL_TEMPL ( unsigned int , major , ( __dev_t __dev ) );`;
        static if(is(typeof({ mixin(enumMixinStr___SYSMACROS_DECLARE_MAJOR); }))) {
            mixin(enumMixinStr___SYSMACROS_DECLARE_MAJOR);
        }
    }




    static if(!is(typeof(_BITS_SYSMACROS_H))) {
        private enum enumMixinStr__BITS_SYSMACROS_H = `enum _BITS_SYSMACROS_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__BITS_SYSMACROS_H); }))) {
            mixin(enumMixinStr__BITS_SYSMACROS_H);
        }
    }




    static if(!is(typeof(_BITS_STDINT_UINTN_H))) {
        private enum enumMixinStr__BITS_STDINT_UINTN_H = `enum _BITS_STDINT_UINTN_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__BITS_STDINT_UINTN_H); }))) {
            mixin(enumMixinStr__BITS_STDINT_UINTN_H);
        }
    }




    static if(!is(typeof(_BITS_STDINT_INTN_H))) {
        private enum enumMixinStr__BITS_STDINT_INTN_H = `enum _BITS_STDINT_INTN_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__BITS_STDINT_INTN_H); }))) {
            mixin(enumMixinStr__BITS_STDINT_INTN_H);
        }
    }
    static if(!is(typeof(__FD_ZERO_STOS))) {
        private enum enumMixinStr___FD_ZERO_STOS = `enum __FD_ZERO_STOS = "stosq";`;
        static if(is(typeof({ mixin(enumMixinStr___FD_ZERO_STOS); }))) {
            mixin(enumMixinStr___FD_ZERO_STOS);
        }
    }




    static if(!is(typeof(__have_pthread_attr_t))) {
        private enum enumMixinStr___have_pthread_attr_t = `enum __have_pthread_attr_t = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___have_pthread_attr_t); }))) {
            mixin(enumMixinStr___have_pthread_attr_t);
        }
    }




    static if(!is(typeof(_BITS_PTHREADTYPES_COMMON_H))) {
        private enum enumMixinStr__BITS_PTHREADTYPES_COMMON_H = `enum _BITS_PTHREADTYPES_COMMON_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__BITS_PTHREADTYPES_COMMON_H); }))) {
            mixin(enumMixinStr__BITS_PTHREADTYPES_COMMON_H);
        }
    }




    static if(!is(typeof(__PTHREAD_RWLOCK_INT_FLAGS_SHARED))) {
        private enum enumMixinStr___PTHREAD_RWLOCK_INT_FLAGS_SHARED = `enum __PTHREAD_RWLOCK_INT_FLAGS_SHARED = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___PTHREAD_RWLOCK_INT_FLAGS_SHARED); }))) {
            mixin(enumMixinStr___PTHREAD_RWLOCK_INT_FLAGS_SHARED);
        }
    }




    static if(!is(typeof(__PTHREAD_RWLOCK_ELISION_EXTRA))) {
        private enum enumMixinStr___PTHREAD_RWLOCK_ELISION_EXTRA = `enum __PTHREAD_RWLOCK_ELISION_EXTRA = 0 , { 0 , 0 , 0 , 0 , 0 , 0 , 0 };`;
        static if(is(typeof({ mixin(enumMixinStr___PTHREAD_RWLOCK_ELISION_EXTRA); }))) {
            mixin(enumMixinStr___PTHREAD_RWLOCK_ELISION_EXTRA);
        }
    }
    static if(!is(typeof(__PTHREAD_MUTEX_USE_UNION))) {
        private enum enumMixinStr___PTHREAD_MUTEX_USE_UNION = `enum __PTHREAD_MUTEX_USE_UNION = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___PTHREAD_MUTEX_USE_UNION); }))) {
            mixin(enumMixinStr___PTHREAD_MUTEX_USE_UNION);
        }
    }




    static if(!is(typeof(__PTHREAD_MUTEX_NUSERS_AFTER_KIND))) {
        private enum enumMixinStr___PTHREAD_MUTEX_NUSERS_AFTER_KIND = `enum __PTHREAD_MUTEX_NUSERS_AFTER_KIND = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___PTHREAD_MUTEX_NUSERS_AFTER_KIND); }))) {
            mixin(enumMixinStr___PTHREAD_MUTEX_NUSERS_AFTER_KIND);
        }
    }




    static if(!is(typeof(__PTHREAD_MUTEX_LOCK_ELISION))) {
        private enum enumMixinStr___PTHREAD_MUTEX_LOCK_ELISION = `enum __PTHREAD_MUTEX_LOCK_ELISION = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___PTHREAD_MUTEX_LOCK_ELISION); }))) {
            mixin(enumMixinStr___PTHREAD_MUTEX_LOCK_ELISION);
        }
    }
    static if(!is(typeof(__SIZEOF_PTHREAD_BARRIERATTR_T))) {
        private enum enumMixinStr___SIZEOF_PTHREAD_BARRIERATTR_T = `enum __SIZEOF_PTHREAD_BARRIERATTR_T = 4;`;
        static if(is(typeof({ mixin(enumMixinStr___SIZEOF_PTHREAD_BARRIERATTR_T); }))) {
            mixin(enumMixinStr___SIZEOF_PTHREAD_BARRIERATTR_T);
        }
    }




    static if(!is(typeof(__SIZEOF_PTHREAD_RWLOCKATTR_T))) {
        private enum enumMixinStr___SIZEOF_PTHREAD_RWLOCKATTR_T = `enum __SIZEOF_PTHREAD_RWLOCKATTR_T = 8;`;
        static if(is(typeof({ mixin(enumMixinStr___SIZEOF_PTHREAD_RWLOCKATTR_T); }))) {
            mixin(enumMixinStr___SIZEOF_PTHREAD_RWLOCKATTR_T);
        }
    }




    static if(!is(typeof(__SIZEOF_PTHREAD_CONDATTR_T))) {
        private enum enumMixinStr___SIZEOF_PTHREAD_CONDATTR_T = `enum __SIZEOF_PTHREAD_CONDATTR_T = 4;`;
        static if(is(typeof({ mixin(enumMixinStr___SIZEOF_PTHREAD_CONDATTR_T); }))) {
            mixin(enumMixinStr___SIZEOF_PTHREAD_CONDATTR_T);
        }
    }




    static if(!is(typeof(__SIZEOF_PTHREAD_COND_T))) {
        private enum enumMixinStr___SIZEOF_PTHREAD_COND_T = `enum __SIZEOF_PTHREAD_COND_T = 48;`;
        static if(is(typeof({ mixin(enumMixinStr___SIZEOF_PTHREAD_COND_T); }))) {
            mixin(enumMixinStr___SIZEOF_PTHREAD_COND_T);
        }
    }




    static if(!is(typeof(__SIZEOF_PTHREAD_MUTEXATTR_T))) {
        private enum enumMixinStr___SIZEOF_PTHREAD_MUTEXATTR_T = `enum __SIZEOF_PTHREAD_MUTEXATTR_T = 4;`;
        static if(is(typeof({ mixin(enumMixinStr___SIZEOF_PTHREAD_MUTEXATTR_T); }))) {
            mixin(enumMixinStr___SIZEOF_PTHREAD_MUTEXATTR_T);
        }
    }




    static if(!is(typeof(__SIZEOF_PTHREAD_BARRIER_T))) {
        private enum enumMixinStr___SIZEOF_PTHREAD_BARRIER_T = `enum __SIZEOF_PTHREAD_BARRIER_T = 32;`;
        static if(is(typeof({ mixin(enumMixinStr___SIZEOF_PTHREAD_BARRIER_T); }))) {
            mixin(enumMixinStr___SIZEOF_PTHREAD_BARRIER_T);
        }
    }




    static if(!is(typeof(__SIZEOF_PTHREAD_RWLOCK_T))) {
        private enum enumMixinStr___SIZEOF_PTHREAD_RWLOCK_T = `enum __SIZEOF_PTHREAD_RWLOCK_T = 56;`;
        static if(is(typeof({ mixin(enumMixinStr___SIZEOF_PTHREAD_RWLOCK_T); }))) {
            mixin(enumMixinStr___SIZEOF_PTHREAD_RWLOCK_T);
        }
    }




    static if(!is(typeof(__SIZEOF_PTHREAD_MUTEX_T))) {
        private enum enumMixinStr___SIZEOF_PTHREAD_MUTEX_T = `enum __SIZEOF_PTHREAD_MUTEX_T = 40;`;
        static if(is(typeof({ mixin(enumMixinStr___SIZEOF_PTHREAD_MUTEX_T); }))) {
            mixin(enumMixinStr___SIZEOF_PTHREAD_MUTEX_T);
        }
    }




    static if(!is(typeof(__SIZEOF_PTHREAD_ATTR_T))) {
        private enum enumMixinStr___SIZEOF_PTHREAD_ATTR_T = `enum __SIZEOF_PTHREAD_ATTR_T = 56;`;
        static if(is(typeof({ mixin(enumMixinStr___SIZEOF_PTHREAD_ATTR_T); }))) {
            mixin(enumMixinStr___SIZEOF_PTHREAD_ATTR_T);
        }
    }




    static if(!is(typeof(_BITS_PTHREADTYPES_ARCH_H))) {
        private enum enumMixinStr__BITS_PTHREADTYPES_ARCH_H = `enum _BITS_PTHREADTYPES_ARCH_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__BITS_PTHREADTYPES_ARCH_H); }))) {
            mixin(enumMixinStr__BITS_PTHREADTYPES_ARCH_H);
        }
    }




    static if(!is(typeof(__GLIBC_USE_IEC_60559_TYPES_EXT))) {
        private enum enumMixinStr___GLIBC_USE_IEC_60559_TYPES_EXT = `enum __GLIBC_USE_IEC_60559_TYPES_EXT = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___GLIBC_USE_IEC_60559_TYPES_EXT); }))) {
            mixin(enumMixinStr___GLIBC_USE_IEC_60559_TYPES_EXT);
        }
    }




    static if(!is(typeof(__GLIBC_USE_IEC_60559_FUNCS_EXT))) {
        private enum enumMixinStr___GLIBC_USE_IEC_60559_FUNCS_EXT = `enum __GLIBC_USE_IEC_60559_FUNCS_EXT = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___GLIBC_USE_IEC_60559_FUNCS_EXT); }))) {
            mixin(enumMixinStr___GLIBC_USE_IEC_60559_FUNCS_EXT);
        }
    }




    static if(!is(typeof(__GLIBC_USE_IEC_60559_BFP_EXT))) {
        private enum enumMixinStr___GLIBC_USE_IEC_60559_BFP_EXT = `enum __GLIBC_USE_IEC_60559_BFP_EXT = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___GLIBC_USE_IEC_60559_BFP_EXT); }))) {
            mixin(enumMixinStr___GLIBC_USE_IEC_60559_BFP_EXT);
        }
    }




    static if(!is(typeof(__GLIBC_USE_LIB_EXT2))) {
        private enum enumMixinStr___GLIBC_USE_LIB_EXT2 = `enum __GLIBC_USE_LIB_EXT2 = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___GLIBC_USE_LIB_EXT2); }))) {
            mixin(enumMixinStr___GLIBC_USE_LIB_EXT2);
        }
    }




    static if(!is(typeof(__HAVE_FLOAT64X_LONG_DOUBLE))) {
        private enum enumMixinStr___HAVE_FLOAT64X_LONG_DOUBLE = `enum __HAVE_FLOAT64X_LONG_DOUBLE = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_FLOAT64X_LONG_DOUBLE); }))) {
            mixin(enumMixinStr___HAVE_FLOAT64X_LONG_DOUBLE);
        }
    }




    static if(!is(typeof(__HAVE_FLOAT64X))) {
        private enum enumMixinStr___HAVE_FLOAT64X = `enum __HAVE_FLOAT64X = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_FLOAT64X); }))) {
            mixin(enumMixinStr___HAVE_FLOAT64X);
        }
    }




    static if(!is(typeof(__HAVE_DISTINCT_FLOAT128))) {
        private enum enumMixinStr___HAVE_DISTINCT_FLOAT128 = `enum __HAVE_DISTINCT_FLOAT128 = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_DISTINCT_FLOAT128); }))) {
            mixin(enumMixinStr___HAVE_DISTINCT_FLOAT128);
        }
    }




    static if(!is(typeof(__HAVE_FLOAT128))) {
        private enum enumMixinStr___HAVE_FLOAT128 = `enum __HAVE_FLOAT128 = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_FLOAT128); }))) {
            mixin(enumMixinStr___HAVE_FLOAT128);
        }
    }
    static if(!is(typeof(__CFLOAT64X))) {
        private enum enumMixinStr___CFLOAT64X = `enum __CFLOAT64X = _Complex long double;`;
        static if(is(typeof({ mixin(enumMixinStr___CFLOAT64X); }))) {
            mixin(enumMixinStr___CFLOAT64X);
        }
    }




    static if(!is(typeof(__CFLOAT32X))) {
        private enum enumMixinStr___CFLOAT32X = `enum __CFLOAT32X = _Complex double;`;
        static if(is(typeof({ mixin(enumMixinStr___CFLOAT32X); }))) {
            mixin(enumMixinStr___CFLOAT32X);
        }
    }




    static if(!is(typeof(__CFLOAT64))) {
        private enum enumMixinStr___CFLOAT64 = `enum __CFLOAT64 = _Complex double;`;
        static if(is(typeof({ mixin(enumMixinStr___CFLOAT64); }))) {
            mixin(enumMixinStr___CFLOAT64);
        }
    }




    static if(!is(typeof(__CFLOAT32))) {
        private enum enumMixinStr___CFLOAT32 = `enum __CFLOAT32 = _Complex float;`;
        static if(is(typeof({ mixin(enumMixinStr___CFLOAT32); }))) {
            mixin(enumMixinStr___CFLOAT32);
        }
    }
    static if(!is(typeof(__HAVE_FLOATN_NOT_TYPEDEF))) {
        private enum enumMixinStr___HAVE_FLOATN_NOT_TYPEDEF = `enum __HAVE_FLOATN_NOT_TYPEDEF = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_FLOATN_NOT_TYPEDEF); }))) {
            mixin(enumMixinStr___HAVE_FLOATN_NOT_TYPEDEF);
        }
    }




    static if(!is(typeof(__HAVE_DISTINCT_FLOAT128X))) {
        private enum enumMixinStr___HAVE_DISTINCT_FLOAT128X = `enum __HAVE_DISTINCT_FLOAT128X = __HAVE_FLOAT128X;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_DISTINCT_FLOAT128X); }))) {
            mixin(enumMixinStr___HAVE_DISTINCT_FLOAT128X);
        }
    }




    static if(!is(typeof(__HAVE_DISTINCT_FLOAT64X))) {
        private enum enumMixinStr___HAVE_DISTINCT_FLOAT64X = `enum __HAVE_DISTINCT_FLOAT64X = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_DISTINCT_FLOAT64X); }))) {
            mixin(enumMixinStr___HAVE_DISTINCT_FLOAT64X);
        }
    }




    static if(!is(typeof(__HAVE_DISTINCT_FLOAT32X))) {
        private enum enumMixinStr___HAVE_DISTINCT_FLOAT32X = `enum __HAVE_DISTINCT_FLOAT32X = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_DISTINCT_FLOAT32X); }))) {
            mixin(enumMixinStr___HAVE_DISTINCT_FLOAT32X);
        }
    }




    static if(!is(typeof(__HAVE_DISTINCT_FLOAT64))) {
        private enum enumMixinStr___HAVE_DISTINCT_FLOAT64 = `enum __HAVE_DISTINCT_FLOAT64 = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_DISTINCT_FLOAT64); }))) {
            mixin(enumMixinStr___HAVE_DISTINCT_FLOAT64);
        }
    }




    static if(!is(typeof(__HAVE_DISTINCT_FLOAT32))) {
        private enum enumMixinStr___HAVE_DISTINCT_FLOAT32 = `enum __HAVE_DISTINCT_FLOAT32 = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_DISTINCT_FLOAT32); }))) {
            mixin(enumMixinStr___HAVE_DISTINCT_FLOAT32);
        }
    }




    static if(!is(typeof(__HAVE_DISTINCT_FLOAT16))) {
        private enum enumMixinStr___HAVE_DISTINCT_FLOAT16 = `enum __HAVE_DISTINCT_FLOAT16 = __HAVE_FLOAT16;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_DISTINCT_FLOAT16); }))) {
            mixin(enumMixinStr___HAVE_DISTINCT_FLOAT16);
        }
    }




    static if(!is(typeof(__HAVE_FLOAT128X))) {
        private enum enumMixinStr___HAVE_FLOAT128X = `enum __HAVE_FLOAT128X = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_FLOAT128X); }))) {
            mixin(enumMixinStr___HAVE_FLOAT128X);
        }
    }




    static if(!is(typeof(__HAVE_FLOAT32X))) {
        private enum enumMixinStr___HAVE_FLOAT32X = `enum __HAVE_FLOAT32X = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_FLOAT32X); }))) {
            mixin(enumMixinStr___HAVE_FLOAT32X);
        }
    }




    static if(!is(typeof(__HAVE_FLOAT64))) {
        private enum enumMixinStr___HAVE_FLOAT64 = `enum __HAVE_FLOAT64 = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_FLOAT64); }))) {
            mixin(enumMixinStr___HAVE_FLOAT64);
        }
    }




    static if(!is(typeof(__HAVE_FLOAT32))) {
        private enum enumMixinStr___HAVE_FLOAT32 = `enum __HAVE_FLOAT32 = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_FLOAT32); }))) {
            mixin(enumMixinStr___HAVE_FLOAT32);
        }
    }




    static if(!is(typeof(__HAVE_FLOAT16))) {
        private enum enumMixinStr___HAVE_FLOAT16 = `enum __HAVE_FLOAT16 = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_FLOAT16); }))) {
            mixin(enumMixinStr___HAVE_FLOAT16);
        }
    }






    static if(!is(typeof(__BYTE_ORDER))) {
        private enum enumMixinStr___BYTE_ORDER = `enum __BYTE_ORDER = __LITTLE_ENDIAN;`;
        static if(is(typeof({ mixin(enumMixinStr___BYTE_ORDER); }))) {
            mixin(enumMixinStr___BYTE_ORDER);
        }
    }
    static if(!is(typeof(_BITS_BYTESWAP_H))) {
        private enum enumMixinStr__BITS_BYTESWAP_H = `enum _BITS_BYTESWAP_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__BITS_BYTESWAP_H); }))) {
            mixin(enumMixinStr__BITS_BYTESWAP_H);
        }
    }
    static if(!is(typeof(MB_CUR_MAX))) {
        private enum enumMixinStr_MB_CUR_MAX = `enum MB_CUR_MAX = ( __ctype_get_mb_cur_max ( ) );`;
        static if(is(typeof({ mixin(enumMixinStr_MB_CUR_MAX); }))) {
            mixin(enumMixinStr_MB_CUR_MAX);
        }
    }




    static if(!is(typeof(EXIT_SUCCESS))) {
        private enum enumMixinStr_EXIT_SUCCESS = `enum EXIT_SUCCESS = 0;`;
        static if(is(typeof({ mixin(enumMixinStr_EXIT_SUCCESS); }))) {
            mixin(enumMixinStr_EXIT_SUCCESS);
        }
    }




    static if(!is(typeof(EXIT_FAILURE))) {
        private enum enumMixinStr_EXIT_FAILURE = `enum EXIT_FAILURE = 1;`;
        static if(is(typeof({ mixin(enumMixinStr_EXIT_FAILURE); }))) {
            mixin(enumMixinStr_EXIT_FAILURE);
        }
    }




    static if(!is(typeof(RAND_MAX))) {
        private enum enumMixinStr_RAND_MAX = `enum RAND_MAX = 2147483647;`;
        static if(is(typeof({ mixin(enumMixinStr_RAND_MAX); }))) {
            mixin(enumMixinStr_RAND_MAX);
        }
    }




    static if(!is(typeof(__lldiv_t_defined))) {
        private enum enumMixinStr___lldiv_t_defined = `enum __lldiv_t_defined = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___lldiv_t_defined); }))) {
            mixin(enumMixinStr___lldiv_t_defined);
        }
    }




    static if(!is(typeof(__ldiv_t_defined))) {
        private enum enumMixinStr___ldiv_t_defined = `enum __ldiv_t_defined = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___ldiv_t_defined); }))) {
            mixin(enumMixinStr___ldiv_t_defined);
        }
    }
    static if(!is(typeof(_STDLIB_H))) {
        private enum enumMixinStr__STDLIB_H = `enum _STDLIB_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__STDLIB_H); }))) {
            mixin(enumMixinStr__STDLIB_H);
        }
    }
    static if(!is(typeof(WINT_MAX))) {
        private enum enumMixinStr_WINT_MAX = `enum WINT_MAX = ( 4294967295u );`;
        static if(is(typeof({ mixin(enumMixinStr_WINT_MAX); }))) {
            mixin(enumMixinStr_WINT_MAX);
        }
    }




    static if(!is(typeof(WINT_MIN))) {
        private enum enumMixinStr_WINT_MIN = `enum WINT_MIN = ( 0u );`;
        static if(is(typeof({ mixin(enumMixinStr_WINT_MIN); }))) {
            mixin(enumMixinStr_WINT_MIN);
        }
    }




    static if(!is(typeof(WCHAR_MAX))) {
        private enum enumMixinStr_WCHAR_MAX = `enum WCHAR_MAX = 0x7fffffff;`;
        static if(is(typeof({ mixin(enumMixinStr_WCHAR_MAX); }))) {
            mixin(enumMixinStr_WCHAR_MAX);
        }
    }




    static if(!is(typeof(WCHAR_MIN))) {
        private enum enumMixinStr_WCHAR_MIN = `enum WCHAR_MIN = ( - 0x7fffffff - 1 );`;
        static if(is(typeof({ mixin(enumMixinStr_WCHAR_MIN); }))) {
            mixin(enumMixinStr_WCHAR_MIN);
        }
    }




    static if(!is(typeof(SIZE_MAX))) {
        private enum enumMixinStr_SIZE_MAX = `enum SIZE_MAX = ( 18446744073709551615UL );`;
        static if(is(typeof({ mixin(enumMixinStr_SIZE_MAX); }))) {
            mixin(enumMixinStr_SIZE_MAX);
        }
    }




    static if(!is(typeof(SIG_ATOMIC_MAX))) {
        private enum enumMixinStr_SIG_ATOMIC_MAX = `enum SIG_ATOMIC_MAX = ( 2147483647 );`;
        static if(is(typeof({ mixin(enumMixinStr_SIG_ATOMIC_MAX); }))) {
            mixin(enumMixinStr_SIG_ATOMIC_MAX);
        }
    }




    static if(!is(typeof(SIG_ATOMIC_MIN))) {
        private enum enumMixinStr_SIG_ATOMIC_MIN = `enum SIG_ATOMIC_MIN = ( - 2147483647 - 1 );`;
        static if(is(typeof({ mixin(enumMixinStr_SIG_ATOMIC_MIN); }))) {
            mixin(enumMixinStr_SIG_ATOMIC_MIN);
        }
    }




    static if(!is(typeof(PTRDIFF_MAX))) {
        private enum enumMixinStr_PTRDIFF_MAX = `enum PTRDIFF_MAX = ( 9223372036854775807L );`;
        static if(is(typeof({ mixin(enumMixinStr_PTRDIFF_MAX); }))) {
            mixin(enumMixinStr_PTRDIFF_MAX);
        }
    }




    static if(!is(typeof(PTRDIFF_MIN))) {
        private enum enumMixinStr_PTRDIFF_MIN = `enum PTRDIFF_MIN = ( - 9223372036854775807L - 1 );`;
        static if(is(typeof({ mixin(enumMixinStr_PTRDIFF_MIN); }))) {
            mixin(enumMixinStr_PTRDIFF_MIN);
        }
    }




    static if(!is(typeof(UINTMAX_MAX))) {
        private enum enumMixinStr_UINTMAX_MAX = `enum UINTMAX_MAX = ( 18446744073709551615UL );`;
        static if(is(typeof({ mixin(enumMixinStr_UINTMAX_MAX); }))) {
            mixin(enumMixinStr_UINTMAX_MAX);
        }
    }




    static if(!is(typeof(INTMAX_MAX))) {
        private enum enumMixinStr_INTMAX_MAX = `enum INTMAX_MAX = ( 9223372036854775807L );`;
        static if(is(typeof({ mixin(enumMixinStr_INTMAX_MAX); }))) {
            mixin(enumMixinStr_INTMAX_MAX);
        }
    }




    static if(!is(typeof(INTMAX_MIN))) {
        private enum enumMixinStr_INTMAX_MIN = `enum INTMAX_MIN = ( - 9223372036854775807L - 1 );`;
        static if(is(typeof({ mixin(enumMixinStr_INTMAX_MIN); }))) {
            mixin(enumMixinStr_INTMAX_MIN);
        }
    }




    static if(!is(typeof(UINTPTR_MAX))) {
        private enum enumMixinStr_UINTPTR_MAX = `enum UINTPTR_MAX = ( 18446744073709551615UL );`;
        static if(is(typeof({ mixin(enumMixinStr_UINTPTR_MAX); }))) {
            mixin(enumMixinStr_UINTPTR_MAX);
        }
    }




    static if(!is(typeof(INTPTR_MAX))) {
        private enum enumMixinStr_INTPTR_MAX = `enum INTPTR_MAX = ( 9223372036854775807L );`;
        static if(is(typeof({ mixin(enumMixinStr_INTPTR_MAX); }))) {
            mixin(enumMixinStr_INTPTR_MAX);
        }
    }




    static if(!is(typeof(INTPTR_MIN))) {
        private enum enumMixinStr_INTPTR_MIN = `enum INTPTR_MIN = ( - 9223372036854775807L - 1 );`;
        static if(is(typeof({ mixin(enumMixinStr_INTPTR_MIN); }))) {
            mixin(enumMixinStr_INTPTR_MIN);
        }
    }




    static if(!is(typeof(UINT_FAST64_MAX))) {
        private enum enumMixinStr_UINT_FAST64_MAX = `enum UINT_FAST64_MAX = ( 18446744073709551615UL );`;
        static if(is(typeof({ mixin(enumMixinStr_UINT_FAST64_MAX); }))) {
            mixin(enumMixinStr_UINT_FAST64_MAX);
        }
    }




    static if(!is(typeof(UINT_FAST32_MAX))) {
        private enum enumMixinStr_UINT_FAST32_MAX = `enum UINT_FAST32_MAX = ( 18446744073709551615UL );`;
        static if(is(typeof({ mixin(enumMixinStr_UINT_FAST32_MAX); }))) {
            mixin(enumMixinStr_UINT_FAST32_MAX);
        }
    }




    static if(!is(typeof(UINT_FAST16_MAX))) {
        private enum enumMixinStr_UINT_FAST16_MAX = `enum UINT_FAST16_MAX = ( 18446744073709551615UL );`;
        static if(is(typeof({ mixin(enumMixinStr_UINT_FAST16_MAX); }))) {
            mixin(enumMixinStr_UINT_FAST16_MAX);
        }
    }




    static if(!is(typeof(UINT_FAST8_MAX))) {
        private enum enumMixinStr_UINT_FAST8_MAX = `enum UINT_FAST8_MAX = ( 255 );`;
        static if(is(typeof({ mixin(enumMixinStr_UINT_FAST8_MAX); }))) {
            mixin(enumMixinStr_UINT_FAST8_MAX);
        }
    }




    static if(!is(typeof(INT_FAST64_MAX))) {
        private enum enumMixinStr_INT_FAST64_MAX = `enum INT_FAST64_MAX = ( 9223372036854775807L );`;
        static if(is(typeof({ mixin(enumMixinStr_INT_FAST64_MAX); }))) {
            mixin(enumMixinStr_INT_FAST64_MAX);
        }
    }




    static if(!is(typeof(INT_FAST32_MAX))) {
        private enum enumMixinStr_INT_FAST32_MAX = `enum INT_FAST32_MAX = ( 9223372036854775807L );`;
        static if(is(typeof({ mixin(enumMixinStr_INT_FAST32_MAX); }))) {
            mixin(enumMixinStr_INT_FAST32_MAX);
        }
    }




    static if(!is(typeof(INT_FAST16_MAX))) {
        private enum enumMixinStr_INT_FAST16_MAX = `enum INT_FAST16_MAX = ( 9223372036854775807L );`;
        static if(is(typeof({ mixin(enumMixinStr_INT_FAST16_MAX); }))) {
            mixin(enumMixinStr_INT_FAST16_MAX);
        }
    }




    static if(!is(typeof(INT_FAST8_MAX))) {
        private enum enumMixinStr_INT_FAST8_MAX = `enum INT_FAST8_MAX = ( 127 );`;
        static if(is(typeof({ mixin(enumMixinStr_INT_FAST8_MAX); }))) {
            mixin(enumMixinStr_INT_FAST8_MAX);
        }
    }




    static if(!is(typeof(INT_FAST64_MIN))) {
        private enum enumMixinStr_INT_FAST64_MIN = `enum INT_FAST64_MIN = ( - 9223372036854775807L - 1 );`;
        static if(is(typeof({ mixin(enumMixinStr_INT_FAST64_MIN); }))) {
            mixin(enumMixinStr_INT_FAST64_MIN);
        }
    }




    static if(!is(typeof(INT_FAST32_MIN))) {
        private enum enumMixinStr_INT_FAST32_MIN = `enum INT_FAST32_MIN = ( - 9223372036854775807L - 1 );`;
        static if(is(typeof({ mixin(enumMixinStr_INT_FAST32_MIN); }))) {
            mixin(enumMixinStr_INT_FAST32_MIN);
        }
    }




    static if(!is(typeof(INT_FAST16_MIN))) {
        private enum enumMixinStr_INT_FAST16_MIN = `enum INT_FAST16_MIN = ( - 9223372036854775807L - 1 );`;
        static if(is(typeof({ mixin(enumMixinStr_INT_FAST16_MIN); }))) {
            mixin(enumMixinStr_INT_FAST16_MIN);
        }
    }




    static if(!is(typeof(INT_FAST8_MIN))) {
        private enum enumMixinStr_INT_FAST8_MIN = `enum INT_FAST8_MIN = ( - 128 );`;
        static if(is(typeof({ mixin(enumMixinStr_INT_FAST8_MIN); }))) {
            mixin(enumMixinStr_INT_FAST8_MIN);
        }
    }




    static if(!is(typeof(UINT_LEAST64_MAX))) {
        private enum enumMixinStr_UINT_LEAST64_MAX = `enum UINT_LEAST64_MAX = ( 18446744073709551615UL );`;
        static if(is(typeof({ mixin(enumMixinStr_UINT_LEAST64_MAX); }))) {
            mixin(enumMixinStr_UINT_LEAST64_MAX);
        }
    }




    static if(!is(typeof(UINT_LEAST32_MAX))) {
        private enum enumMixinStr_UINT_LEAST32_MAX = `enum UINT_LEAST32_MAX = ( 4294967295U );`;
        static if(is(typeof({ mixin(enumMixinStr_UINT_LEAST32_MAX); }))) {
            mixin(enumMixinStr_UINT_LEAST32_MAX);
        }
    }




    static if(!is(typeof(UINT_LEAST16_MAX))) {
        private enum enumMixinStr_UINT_LEAST16_MAX = `enum UINT_LEAST16_MAX = ( 65535 );`;
        static if(is(typeof({ mixin(enumMixinStr_UINT_LEAST16_MAX); }))) {
            mixin(enumMixinStr_UINT_LEAST16_MAX);
        }
    }




    static if(!is(typeof(UINT_LEAST8_MAX))) {
        private enum enumMixinStr_UINT_LEAST8_MAX = `enum UINT_LEAST8_MAX = ( 255 );`;
        static if(is(typeof({ mixin(enumMixinStr_UINT_LEAST8_MAX); }))) {
            mixin(enumMixinStr_UINT_LEAST8_MAX);
        }
    }




    static if(!is(typeof(INT_LEAST64_MAX))) {
        private enum enumMixinStr_INT_LEAST64_MAX = `enum INT_LEAST64_MAX = ( 9223372036854775807L );`;
        static if(is(typeof({ mixin(enumMixinStr_INT_LEAST64_MAX); }))) {
            mixin(enumMixinStr_INT_LEAST64_MAX);
        }
    }




    static if(!is(typeof(INT_LEAST32_MAX))) {
        private enum enumMixinStr_INT_LEAST32_MAX = `enum INT_LEAST32_MAX = ( 2147483647 );`;
        static if(is(typeof({ mixin(enumMixinStr_INT_LEAST32_MAX); }))) {
            mixin(enumMixinStr_INT_LEAST32_MAX);
        }
    }




    static if(!is(typeof(INT_LEAST16_MAX))) {
        private enum enumMixinStr_INT_LEAST16_MAX = `enum INT_LEAST16_MAX = ( 32767 );`;
        static if(is(typeof({ mixin(enumMixinStr_INT_LEAST16_MAX); }))) {
            mixin(enumMixinStr_INT_LEAST16_MAX);
        }
    }




    static if(!is(typeof(INT_LEAST8_MAX))) {
        private enum enumMixinStr_INT_LEAST8_MAX = `enum INT_LEAST8_MAX = ( 127 );`;
        static if(is(typeof({ mixin(enumMixinStr_INT_LEAST8_MAX); }))) {
            mixin(enumMixinStr_INT_LEAST8_MAX);
        }
    }




    static if(!is(typeof(INT_LEAST64_MIN))) {
        private enum enumMixinStr_INT_LEAST64_MIN = `enum INT_LEAST64_MIN = ( - 9223372036854775807L - 1 );`;
        static if(is(typeof({ mixin(enumMixinStr_INT_LEAST64_MIN); }))) {
            mixin(enumMixinStr_INT_LEAST64_MIN);
        }
    }




    static if(!is(typeof(INT_LEAST32_MIN))) {
        private enum enumMixinStr_INT_LEAST32_MIN = `enum INT_LEAST32_MIN = ( - 2147483647 - 1 );`;
        static if(is(typeof({ mixin(enumMixinStr_INT_LEAST32_MIN); }))) {
            mixin(enumMixinStr_INT_LEAST32_MIN);
        }
    }




    static if(!is(typeof(INT_LEAST16_MIN))) {
        private enum enumMixinStr_INT_LEAST16_MIN = `enum INT_LEAST16_MIN = ( - 32767 - 1 );`;
        static if(is(typeof({ mixin(enumMixinStr_INT_LEAST16_MIN); }))) {
            mixin(enumMixinStr_INT_LEAST16_MIN);
        }
    }




    static if(!is(typeof(INT_LEAST8_MIN))) {
        private enum enumMixinStr_INT_LEAST8_MIN = `enum INT_LEAST8_MIN = ( - 128 );`;
        static if(is(typeof({ mixin(enumMixinStr_INT_LEAST8_MIN); }))) {
            mixin(enumMixinStr_INT_LEAST8_MIN);
        }
    }




    static if(!is(typeof(UINT64_MAX))) {
        private enum enumMixinStr_UINT64_MAX = `enum UINT64_MAX = ( 18446744073709551615UL );`;
        static if(is(typeof({ mixin(enumMixinStr_UINT64_MAX); }))) {
            mixin(enumMixinStr_UINT64_MAX);
        }
    }




    static if(!is(typeof(UINT32_MAX))) {
        private enum enumMixinStr_UINT32_MAX = `enum UINT32_MAX = ( 4294967295U );`;
        static if(is(typeof({ mixin(enumMixinStr_UINT32_MAX); }))) {
            mixin(enumMixinStr_UINT32_MAX);
        }
    }




    static if(!is(typeof(UINT16_MAX))) {
        private enum enumMixinStr_UINT16_MAX = `enum UINT16_MAX = ( 65535 );`;
        static if(is(typeof({ mixin(enumMixinStr_UINT16_MAX); }))) {
            mixin(enumMixinStr_UINT16_MAX);
        }
    }




    static if(!is(typeof(UINT8_MAX))) {
        private enum enumMixinStr_UINT8_MAX = `enum UINT8_MAX = ( 255 );`;
        static if(is(typeof({ mixin(enumMixinStr_UINT8_MAX); }))) {
            mixin(enumMixinStr_UINT8_MAX);
        }
    }




    static if(!is(typeof(INT64_MAX))) {
        private enum enumMixinStr_INT64_MAX = `enum INT64_MAX = ( 9223372036854775807L );`;
        static if(is(typeof({ mixin(enumMixinStr_INT64_MAX); }))) {
            mixin(enumMixinStr_INT64_MAX);
        }
    }




    static if(!is(typeof(INT32_MAX))) {
        private enum enumMixinStr_INT32_MAX = `enum INT32_MAX = ( 2147483647 );`;
        static if(is(typeof({ mixin(enumMixinStr_INT32_MAX); }))) {
            mixin(enumMixinStr_INT32_MAX);
        }
    }




    static if(!is(typeof(INT16_MAX))) {
        private enum enumMixinStr_INT16_MAX = `enum INT16_MAX = ( 32767 );`;
        static if(is(typeof({ mixin(enumMixinStr_INT16_MAX); }))) {
            mixin(enumMixinStr_INT16_MAX);
        }
    }




    static if(!is(typeof(INT8_MAX))) {
        private enum enumMixinStr_INT8_MAX = `enum INT8_MAX = ( 127 );`;
        static if(is(typeof({ mixin(enumMixinStr_INT8_MAX); }))) {
            mixin(enumMixinStr_INT8_MAX);
        }
    }




    static if(!is(typeof(INT64_MIN))) {
        private enum enumMixinStr_INT64_MIN = `enum INT64_MIN = ( - 9223372036854775807L - 1 );`;
        static if(is(typeof({ mixin(enumMixinStr_INT64_MIN); }))) {
            mixin(enumMixinStr_INT64_MIN);
        }
    }




    static if(!is(typeof(INT32_MIN))) {
        private enum enumMixinStr_INT32_MIN = `enum INT32_MIN = ( - 2147483647 - 1 );`;
        static if(is(typeof({ mixin(enumMixinStr_INT32_MIN); }))) {
            mixin(enumMixinStr_INT32_MIN);
        }
    }




    static if(!is(typeof(INT16_MIN))) {
        private enum enumMixinStr_INT16_MIN = `enum INT16_MIN = ( - 32767 - 1 );`;
        static if(is(typeof({ mixin(enumMixinStr_INT16_MIN); }))) {
            mixin(enumMixinStr_INT16_MIN);
        }
    }




    static if(!is(typeof(INT8_MIN))) {
        private enum enumMixinStr_INT8_MIN = `enum INT8_MIN = ( - 128 );`;
        static if(is(typeof({ mixin(enumMixinStr_INT8_MIN); }))) {
            mixin(enumMixinStr_INT8_MIN);
        }
    }
    static if(!is(typeof(_STDINT_H))) {
        private enum enumMixinStr__STDINT_H = `enum _STDINT_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__STDINT_H); }))) {
            mixin(enumMixinStr__STDINT_H);
        }
    }




    static if(!is(typeof(_STDC_PREDEF_H))) {
        private enum enumMixinStr__STDC_PREDEF_H = `enum _STDC_PREDEF_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__STDC_PREDEF_H); }))) {
            mixin(enumMixinStr__STDC_PREDEF_H);
        }
    }






    static if(!is(typeof(__GLIBC_MINOR__))) {
        private enum enumMixinStr___GLIBC_MINOR__ = `enum __GLIBC_MINOR__ = 27;`;
        static if(is(typeof({ mixin(enumMixinStr___GLIBC_MINOR__); }))) {
            mixin(enumMixinStr___GLIBC_MINOR__);
        }
    }




    static if(!is(typeof(__GLIBC__))) {
        private enum enumMixinStr___GLIBC__ = `enum __GLIBC__ = 2;`;
        static if(is(typeof({ mixin(enumMixinStr___GLIBC__); }))) {
            mixin(enumMixinStr___GLIBC__);
        }
    }




    static if(!is(typeof(__GNU_LIBRARY__))) {
        private enum enumMixinStr___GNU_LIBRARY__ = `enum __GNU_LIBRARY__ = 6;`;
        static if(is(typeof({ mixin(enumMixinStr___GNU_LIBRARY__); }))) {
            mixin(enumMixinStr___GNU_LIBRARY__);
        }
    }




    static if(!is(typeof(__GLIBC_USE_DEPRECATED_GETS))) {
        private enum enumMixinStr___GLIBC_USE_DEPRECATED_GETS = `enum __GLIBC_USE_DEPRECATED_GETS = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___GLIBC_USE_DEPRECATED_GETS); }))) {
            mixin(enumMixinStr___GLIBC_USE_DEPRECATED_GETS);
        }
    }




    static if(!is(typeof(__USE_FORTIFY_LEVEL))) {
        private enum enumMixinStr___USE_FORTIFY_LEVEL = `enum __USE_FORTIFY_LEVEL = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___USE_FORTIFY_LEVEL); }))) {
            mixin(enumMixinStr___USE_FORTIFY_LEVEL);
        }
    }




    static if(!is(typeof(__USE_ATFILE))) {
        private enum enumMixinStr___USE_ATFILE = `enum __USE_ATFILE = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___USE_ATFILE); }))) {
            mixin(enumMixinStr___USE_ATFILE);
        }
    }




    static if(!is(typeof(__USE_MISC))) {
        private enum enumMixinStr___USE_MISC = `enum __USE_MISC = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___USE_MISC); }))) {
            mixin(enumMixinStr___USE_MISC);
        }
    }




    static if(!is(typeof(_ATFILE_SOURCE))) {
        private enum enumMixinStr__ATFILE_SOURCE = `enum _ATFILE_SOURCE = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__ATFILE_SOURCE); }))) {
            mixin(enumMixinStr__ATFILE_SOURCE);
        }
    }




    static if(!is(typeof(__USE_XOPEN2K8))) {
        private enum enumMixinStr___USE_XOPEN2K8 = `enum __USE_XOPEN2K8 = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___USE_XOPEN2K8); }))) {
            mixin(enumMixinStr___USE_XOPEN2K8);
        }
    }




    static if(!is(typeof(__USE_ISOC99))) {
        private enum enumMixinStr___USE_ISOC99 = `enum __USE_ISOC99 = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___USE_ISOC99); }))) {
            mixin(enumMixinStr___USE_ISOC99);
        }
    }




    static if(!is(typeof(__USE_ISOC95))) {
        private enum enumMixinStr___USE_ISOC95 = `enum __USE_ISOC95 = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___USE_ISOC95); }))) {
            mixin(enumMixinStr___USE_ISOC95);
        }
    }




    static if(!is(typeof(__USE_XOPEN2K))) {
        private enum enumMixinStr___USE_XOPEN2K = `enum __USE_XOPEN2K = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___USE_XOPEN2K); }))) {
            mixin(enumMixinStr___USE_XOPEN2K);
        }
    }




    static if(!is(typeof(__USE_POSIX199506))) {
        private enum enumMixinStr___USE_POSIX199506 = `enum __USE_POSIX199506 = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___USE_POSIX199506); }))) {
            mixin(enumMixinStr___USE_POSIX199506);
        }
    }




    static if(!is(typeof(__USE_POSIX199309))) {
        private enum enumMixinStr___USE_POSIX199309 = `enum __USE_POSIX199309 = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___USE_POSIX199309); }))) {
            mixin(enumMixinStr___USE_POSIX199309);
        }
    }




    static if(!is(typeof(__USE_POSIX2))) {
        private enum enumMixinStr___USE_POSIX2 = `enum __USE_POSIX2 = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___USE_POSIX2); }))) {
            mixin(enumMixinStr___USE_POSIX2);
        }
    }




    static if(!is(typeof(__USE_POSIX))) {
        private enum enumMixinStr___USE_POSIX = `enum __USE_POSIX = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___USE_POSIX); }))) {
            mixin(enumMixinStr___USE_POSIX);
        }
    }




    static if(!is(typeof(_POSIX_C_SOURCE))) {
        private enum enumMixinStr__POSIX_C_SOURCE = `enum _POSIX_C_SOURCE = 200809L;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_C_SOURCE); }))) {
            mixin(enumMixinStr__POSIX_C_SOURCE);
        }
    }




    static if(!is(typeof(_POSIX_SOURCE))) {
        private enum enumMixinStr__POSIX_SOURCE = `enum _POSIX_SOURCE = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_SOURCE); }))) {
            mixin(enumMixinStr__POSIX_SOURCE);
        }
    }




    static if(!is(typeof(__USE_POSIX_IMPLICITLY))) {
        private enum enumMixinStr___USE_POSIX_IMPLICITLY = `enum __USE_POSIX_IMPLICITLY = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___USE_POSIX_IMPLICITLY); }))) {
            mixin(enumMixinStr___USE_POSIX_IMPLICITLY);
        }
    }




    static if(!is(typeof(__USE_ISOC11))) {
        private enum enumMixinStr___USE_ISOC11 = `enum __USE_ISOC11 = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___USE_ISOC11); }))) {
            mixin(enumMixinStr___USE_ISOC11);
        }
    }




    static if(!is(typeof(_DEFAULT_SOURCE))) {
        private enum enumMixinStr__DEFAULT_SOURCE = `enum _DEFAULT_SOURCE = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__DEFAULT_SOURCE); }))) {
            mixin(enumMixinStr__DEFAULT_SOURCE);
        }
    }
    static if(!is(typeof(_FEATURES_H))) {
        private enum enumMixinStr__FEATURES_H = `enum _FEATURES_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__FEATURES_H); }))) {
            mixin(enumMixinStr__FEATURES_H);
        }
    }
    static if(!is(typeof(BYTE_ORDER))) {
        private enum enumMixinStr_BYTE_ORDER = `enum BYTE_ORDER = __LITTLE_ENDIAN;`;
        static if(is(typeof({ mixin(enumMixinStr_BYTE_ORDER); }))) {
            mixin(enumMixinStr_BYTE_ORDER);
        }
    }




    static if(!is(typeof(PDP_ENDIAN))) {
        private enum enumMixinStr_PDP_ENDIAN = `enum PDP_ENDIAN = __PDP_ENDIAN;`;
        static if(is(typeof({ mixin(enumMixinStr_PDP_ENDIAN); }))) {
            mixin(enumMixinStr_PDP_ENDIAN);
        }
    }




    static if(!is(typeof(BIG_ENDIAN))) {
        private enum enumMixinStr_BIG_ENDIAN = `enum BIG_ENDIAN = __BIG_ENDIAN;`;
        static if(is(typeof({ mixin(enumMixinStr_BIG_ENDIAN); }))) {
            mixin(enumMixinStr_BIG_ENDIAN);
        }
    }




    static if(!is(typeof(LITTLE_ENDIAN))) {
        private enum enumMixinStr_LITTLE_ENDIAN = `enum LITTLE_ENDIAN = __LITTLE_ENDIAN;`;
        static if(is(typeof({ mixin(enumMixinStr_LITTLE_ENDIAN); }))) {
            mixin(enumMixinStr_LITTLE_ENDIAN);
        }
    }




    static if(!is(typeof(__FLOAT_WORD_ORDER))) {
        private enum enumMixinStr___FLOAT_WORD_ORDER = `enum __FLOAT_WORD_ORDER = __LITTLE_ENDIAN;`;
        static if(is(typeof({ mixin(enumMixinStr___FLOAT_WORD_ORDER); }))) {
            mixin(enumMixinStr___FLOAT_WORD_ORDER);
        }
    }




    static if(!is(typeof(__PDP_ENDIAN))) {
        private enum enumMixinStr___PDP_ENDIAN = `enum __PDP_ENDIAN = 3412;`;
        static if(is(typeof({ mixin(enumMixinStr___PDP_ENDIAN); }))) {
            mixin(enumMixinStr___PDP_ENDIAN);
        }
    }




    static if(!is(typeof(__BIG_ENDIAN))) {
        private enum enumMixinStr___BIG_ENDIAN = `enum __BIG_ENDIAN = 4321;`;
        static if(is(typeof({ mixin(enumMixinStr___BIG_ENDIAN); }))) {
            mixin(enumMixinStr___BIG_ENDIAN);
        }
    }




    static if(!is(typeof(__LITTLE_ENDIAN))) {
        private enum enumMixinStr___LITTLE_ENDIAN = `enum __LITTLE_ENDIAN = 1234;`;
        static if(is(typeof({ mixin(enumMixinStr___LITTLE_ENDIAN); }))) {
            mixin(enumMixinStr___LITTLE_ENDIAN);
        }
    }




    static if(!is(typeof(_ENDIAN_H))) {
        private enum enumMixinStr__ENDIAN_H = `enum _ENDIAN_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__ENDIAN_H); }))) {
            mixin(enumMixinStr__ENDIAN_H);
        }
    }




    static if(!is(typeof(NULL))) {
        private enum enumMixinStr_NULL = `enum NULL = ( cast( void * ) 0 );`;
        static if(is(typeof({ mixin(enumMixinStr_NULL); }))) {
            mixin(enumMixinStr_NULL);
        }
    }
    static if(!is(typeof(_ALLOCA_H))) {
        private enum enumMixinStr__ALLOCA_H = `enum _ALLOCA_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__ALLOCA_H); }))) {
            mixin(enumMixinStr__ALLOCA_H);
        }
    }

}
