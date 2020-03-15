module grain.dpp.cuda_runtime_api;


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
    struct dim3
    {
        uint x;
        uint y;
        uint z;
    }
    struct double4
    {
        double x;
        double y;
        double z;
        double w;
    }
    struct double3
    {
        double x;
        double y;
        double z;
    }
    struct double2
    {
        double x;
        double y;
    }
    struct double1
    {
        double x;
    }
    struct ulonglong4
    {
        ulong x;
        ulong y;
        ulong z;
        ulong w;
    }
    struct longlong4
    {
        long x;
        long y;
        long z;
        long w;
    }
    struct ulonglong3
    {
        ulong x;
        ulong y;
        ulong z;
    }
    struct longlong3
    {
        long x;
        long y;
        long z;
    }
    struct ulonglong2
    {
        ulong x;
        ulong y;
    }
    struct longlong2
    {
        long x;
        long y;
    }
    struct ulonglong1
    {
        ulong x;
    }
    struct longlong1
    {
        long x;
    }
    struct float4
    {
        float x;
        float y;
        float z;
        float w;
    }
    struct float3
    {
        float x;
        float y;
        float z;
    }
    struct float2
    {
        float x;
        float y;
    }
    struct float1
    {
        float x;
    }
    struct ulong4
    {
        c_ulong x;
        c_ulong y;
        c_ulong z;
        c_ulong w;
    }
    struct long4
    {
        c_long x;
        c_long y;
        c_long z;
        c_long w;
    }
    struct ulong3
    {
        c_ulong x;
        c_ulong y;
        c_ulong z;
    }
    struct long3
    {
        c_long x;
        c_long y;
        c_long z;
    }
    struct ulong2
    {
        c_ulong x;
        c_ulong y;
    }
    struct long2
    {
        c_long x;
        c_long y;
    }
    struct ulong1
    {
        c_ulong x;
    }
    struct long1
    {
        c_long x;
    }
    struct uint4
    {
        uint x;
        uint y;
        uint z;
        uint w;
    }
    struct int4
    {
        int x;
        int y;
        int z;
        int w;
    }
    struct uint3
    {
        uint x;
        uint y;
        uint z;
    }
    struct int3
    {
        int x;
        int y;
        int z;
    }
    struct uint2
    {
        uint x;
        uint y;
    }
    struct int2
    {
        int x;
        int y;
    }
    struct uint1
    {
        uint x;
    }
    struct int1
    {
        int x;
    }
    struct ushort4
    {
        ushort x;
        ushort y;
        ushort z;
        ushort w;
    }
    struct short4
    {
        short x;
        short y;
        short z;
        short w;
    }
    struct ushort3
    {
        ushort x;
        ushort y;
        ushort z;
    }
    struct short3
    {
        short x;
        short y;
        short z;
    }
    struct ushort2
    {
        ushort x;
        ushort y;
    }
    struct short2
    {
        short x;
        short y;
    }
    struct ushort1
    {
        ushort x;
    }
    struct short1
    {
        short x;
    }
    struct uchar4
    {
        ubyte x;
        ubyte y;
        ubyte z;
        ubyte w;
    }
    struct char4
    {
        byte x;
        byte y;
        byte z;
        byte w;
    }
    struct uchar3
    {
        ubyte x;
        ubyte y;
        ubyte z;
    }
    struct char3
    {
        byte x;
        byte y;
        byte z;
    }
    struct uchar2
    {
        ubyte x;
        ubyte y;
    }
    struct char2
    {
        byte x;
        byte y;
    }
    struct uchar1
    {
        ubyte x;
    }
    struct char1
    {
        byte x;
    }
    alias cudaTextureObject_t = ulong;
    struct cudaTextureDesc
    {
        cudaTextureAddressMode[3] addressMode;
        cudaTextureFilterMode filterMode;
        cudaTextureReadMode readMode;
        int sRGB;
        float[4] borderColor;
        int normalizedCoords;
        uint maxAnisotropy;
        cudaTextureFilterMode mipmapFilterMode;
        float mipmapLevelBias;
        float minMipmapLevelClamp;
        float maxMipmapLevelClamp;
    }
    struct textureReference
    {
        int normalized;
        cudaTextureFilterMode filterMode;
        cudaTextureAddressMode[3] addressMode;
        cudaChannelFormatDesc channelDesc;
        int sRGB;
        uint maxAnisotropy;
        cudaTextureFilterMode mipmapFilterMode;
        float mipmapLevelBias;
        float minMipmapLevelClamp;
        float maxMipmapLevelClamp;
        int[15] __cudaReserved;
    }
    enum cudaTextureReadMode
    {
        cudaReadModeElementType = 0,
        cudaReadModeNormalizedFloat = 1,
    }
    enum cudaReadModeElementType = cudaTextureReadMode.cudaReadModeElementType;
    enum cudaReadModeNormalizedFloat = cudaTextureReadMode.cudaReadModeNormalizedFloat;
    enum cudaTextureFilterMode
    {
        cudaFilterModePoint = 0,
        cudaFilterModeLinear = 1,
    }
    enum cudaFilterModePoint = cudaTextureFilterMode.cudaFilterModePoint;
    enum cudaFilterModeLinear = cudaTextureFilterMode.cudaFilterModeLinear;
    enum cudaTextureAddressMode
    {
        cudaAddressModeWrap = 0,
        cudaAddressModeClamp = 1,
        cudaAddressModeMirror = 2,
        cudaAddressModeBorder = 3,
    }
    enum cudaAddressModeWrap = cudaTextureAddressMode.cudaAddressModeWrap;
    enum cudaAddressModeClamp = cudaTextureAddressMode.cudaAddressModeClamp;
    enum cudaAddressModeMirror = cudaTextureAddressMode.cudaAddressModeMirror;
    enum cudaAddressModeBorder = cudaTextureAddressMode.cudaAddressModeBorder;
    alias cudaSurfaceObject_t = ulong;
    struct surfaceReference
    {
        cudaChannelFormatDesc channelDesc;
    }
    enum cudaSurfaceFormatMode
    {
        cudaFormatModeForced = 0,
        cudaFormatModeAuto = 1,
    }
    enum cudaFormatModeForced = cudaSurfaceFormatMode.cudaFormatModeForced;
    enum cudaFormatModeAuto = cudaSurfaceFormatMode.cudaFormatModeAuto;
    enum cudaSurfaceBoundaryMode
    {
        cudaBoundaryModeZero = 0,
        cudaBoundaryModeClamp = 1,
        cudaBoundaryModeTrap = 2,
    }
    enum cudaBoundaryModeZero = cudaSurfaceBoundaryMode.cudaBoundaryModeZero;
    enum cudaBoundaryModeClamp = cudaSurfaceBoundaryMode.cudaBoundaryModeClamp;
    enum cudaBoundaryModeTrap = cudaSurfaceBoundaryMode.cudaBoundaryModeTrap;
    enum cudaGraphExecUpdateResult
    {
        cudaGraphExecUpdateSuccess = 0,
        cudaGraphExecUpdateError = 1,
        cudaGraphExecUpdateErrorTopologyChanged = 2,
        cudaGraphExecUpdateErrorNodeTypeChanged = 3,
        cudaGraphExecUpdateErrorFunctionChanged = 4,
        cudaGraphExecUpdateErrorParametersChanged = 5,
        cudaGraphExecUpdateErrorNotSupported = 6,
    }
    enum cudaGraphExecUpdateSuccess = cudaGraphExecUpdateResult.cudaGraphExecUpdateSuccess;
    enum cudaGraphExecUpdateError = cudaGraphExecUpdateResult.cudaGraphExecUpdateError;
    enum cudaGraphExecUpdateErrorTopologyChanged = cudaGraphExecUpdateResult.cudaGraphExecUpdateErrorTopologyChanged;
    enum cudaGraphExecUpdateErrorNodeTypeChanged = cudaGraphExecUpdateResult.cudaGraphExecUpdateErrorNodeTypeChanged;
    enum cudaGraphExecUpdateErrorFunctionChanged = cudaGraphExecUpdateResult.cudaGraphExecUpdateErrorFunctionChanged;
    enum cudaGraphExecUpdateErrorParametersChanged = cudaGraphExecUpdateResult.cudaGraphExecUpdateErrorParametersChanged;
    enum cudaGraphExecUpdateErrorNotSupported = cudaGraphExecUpdateResult.cudaGraphExecUpdateErrorNotSupported;
    struct CUgraphExec_st;
    alias cudaGraphExec_t = CUgraphExec_st*;
    enum cudaGraphNodeType
    {
        cudaGraphNodeTypeKernel = 0,
        cudaGraphNodeTypeMemcpy = 1,
        cudaGraphNodeTypeMemset = 2,
        cudaGraphNodeTypeHost = 3,
        cudaGraphNodeTypeGraph = 4,
        cudaGraphNodeTypeEmpty = 5,
        cudaGraphNodeTypeCount = 6,
    }
    enum cudaGraphNodeTypeKernel = cudaGraphNodeType.cudaGraphNodeTypeKernel;
    enum cudaGraphNodeTypeMemcpy = cudaGraphNodeType.cudaGraphNodeTypeMemcpy;
    enum cudaGraphNodeTypeMemset = cudaGraphNodeType.cudaGraphNodeTypeMemset;
    enum cudaGraphNodeTypeHost = cudaGraphNodeType.cudaGraphNodeTypeHost;
    enum cudaGraphNodeTypeGraph = cudaGraphNodeType.cudaGraphNodeTypeGraph;
    enum cudaGraphNodeTypeEmpty = cudaGraphNodeType.cudaGraphNodeTypeEmpty;
    enum cudaGraphNodeTypeCount = cudaGraphNodeType.cudaGraphNodeTypeCount;
    struct cudaKernelNodeParams
    {
        void* func;
        dim3 gridDim;
        dim3 blockDim;
        uint sharedMemBytes;
        void** kernelParams;
        void** extra;
    }
    struct cudaLaunchParams
    {
        void* func;
        dim3 gridDim;
        dim3 blockDim;
        void** args;
        c_ulong sharedMem;
        CUstream_st* stream;
    }
    enum cudaCGScope
    {
        cudaCGScopeInvalid = 0,
        cudaCGScopeGrid = 1,
        cudaCGScopeMultiGrid = 2,
    }
    enum cudaCGScopeInvalid = cudaCGScope.cudaCGScopeInvalid;
    enum cudaCGScopeGrid = cudaCGScope.cudaCGScopeGrid;
    enum cudaCGScopeMultiGrid = cudaCGScope.cudaCGScopeMultiGrid;
    struct CUgraphNode_st;
    alias cudaGraphNode_t = CUgraphNode_st*;
    struct CUgraph_st;
    alias cudaGraph_t = CUgraph_st*;
    struct CUexternalSemaphore_st;
    alias cudaExternalSemaphore_t = CUexternalSemaphore_st*;
    struct CUexternalMemory_st;
    alias cudaExternalMemory_t = CUexternalMemory_st*;
    alias cudaOutputMode_t = cudaOutputMode;
    alias cudaGraphicsResource_t = cudaGraphicsResource*;
    struct CUevent_st;
    alias cudaEvent_t = CUevent_st*;
    struct CUstream_st;
    alias cudaStream_t = CUstream_st*;
    alias cudaError_t = cudaError;
    struct cudaExternalSemaphoreWaitParams
    {
        static struct _Anonymous_0
        {
            static struct _Anonymous_1
            {
                ulong value;
            }
            _Anonymous_1 fence;
            static union _Anonymous_2
            {
                void* fence;
                ulong reserved;
            }
            _Anonymous_2 nvSciSync;
            static struct _Anonymous_3
            {
                ulong key;
                uint timeoutMs;
            }
            _Anonymous_3 keyedMutex;
        }
        _Anonymous_0 params;
        uint flags;
    }
    struct cudaExternalSemaphoreSignalParams
    {
        static struct _Anonymous_4
        {
            static struct _Anonymous_5
            {
                ulong value;
            }
            _Anonymous_5 fence;
            static union _Anonymous_6
            {
                void* fence;
                ulong reserved;
            }
            _Anonymous_6 nvSciSync;
            static struct _Anonymous_7
            {
                ulong key;
            }
            _Anonymous_7 keyedMutex;
        }
        _Anonymous_4 params;
        uint flags;
    }
    struct cudaExternalSemaphoreHandleDesc
    {
        cudaExternalSemaphoreHandleType type;
        static union _Anonymous_8
        {
            int fd;
            static struct _Anonymous_9
            {
                void* handle;
                const(void)* name;
            }
            _Anonymous_9 win32;
            const(void)* nvSciSyncObj;
        }
        _Anonymous_8 handle;
        uint flags;
    }
    enum cudaExternalSemaphoreHandleType
    {
        cudaExternalSemaphoreHandleTypeOpaqueFd = 1,
        cudaExternalSemaphoreHandleTypeOpaqueWin32 = 2,
        cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt = 3,
        cudaExternalSemaphoreHandleTypeD3D12Fence = 4,
        cudaExternalSemaphoreHandleTypeD3D11Fence = 5,
        cudaExternalSemaphoreHandleTypeNvSciSync = 6,
        cudaExternalSemaphoreHandleTypeKeyedMutex = 7,
        cudaExternalSemaphoreHandleTypeKeyedMutexKmt = 8,
    }
    enum cudaExternalSemaphoreHandleTypeOpaqueFd = cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeOpaqueFd;
    enum cudaExternalSemaphoreHandleTypeOpaqueWin32 = cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeOpaqueWin32;
    enum cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt = cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
    enum cudaExternalSemaphoreHandleTypeD3D12Fence = cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeD3D12Fence;
    enum cudaExternalSemaphoreHandleTypeD3D11Fence = cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeD3D11Fence;
    enum cudaExternalSemaphoreHandleTypeNvSciSync = cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeNvSciSync;
    enum cudaExternalSemaphoreHandleTypeKeyedMutex = cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeKeyedMutex;
    enum cudaExternalSemaphoreHandleTypeKeyedMutexKmt = cudaExternalSemaphoreHandleType.cudaExternalSemaphoreHandleTypeKeyedMutexKmt;
    struct cudaExternalMemoryMipmappedArrayDesc
    {
        ulong offset;
        cudaChannelFormatDesc formatDesc;
        cudaExtent extent;
        uint flags;
        uint numLevels;
    }
    struct cudaExternalMemoryBufferDesc
    {
        ulong offset;
        ulong size;
        uint flags;
    }
    struct cudaExternalMemoryHandleDesc
    {
        cudaExternalMemoryHandleType type;
        static union _Anonymous_10
        {
            int fd;
            static struct _Anonymous_11
            {
                void* handle;
                const(void)* name;
            }
            _Anonymous_11 win32;
            const(void)* nvSciBufObject;
        }
        _Anonymous_10 handle;
        ulong size;
        uint flags;
    }
    enum cudaExternalMemoryHandleType
    {
        cudaExternalMemoryHandleTypeOpaqueFd = 1,
        cudaExternalMemoryHandleTypeOpaqueWin32 = 2,
        cudaExternalMemoryHandleTypeOpaqueWin32Kmt = 3,
        cudaExternalMemoryHandleTypeD3D12Heap = 4,
        cudaExternalMemoryHandleTypeD3D12Resource = 5,
        cudaExternalMemoryHandleTypeD3D11Resource = 6,
        cudaExternalMemoryHandleTypeD3D11ResourceKmt = 7,
        cudaExternalMemoryHandleTypeNvSciBuf = 8,
    }
    enum cudaExternalMemoryHandleTypeOpaqueFd = cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeOpaqueFd;
    enum cudaExternalMemoryHandleTypeOpaqueWin32 = cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeOpaqueWin32;
    enum cudaExternalMemoryHandleTypeOpaqueWin32Kmt = cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
    enum cudaExternalMemoryHandleTypeD3D12Heap = cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeD3D12Heap;
    enum cudaExternalMemoryHandleTypeD3D12Resource = cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeD3D12Resource;
    enum cudaExternalMemoryHandleTypeD3D11Resource = cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeD3D11Resource;
    enum cudaExternalMemoryHandleTypeD3D11ResourceKmt = cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeD3D11ResourceKmt;
    enum cudaExternalMemoryHandleTypeNvSciBuf = cudaExternalMemoryHandleType.cudaExternalMemoryHandleTypeNvSciBuf;
    struct cudaIpcMemHandle_st
    {
        char[64] reserved;
    }
    alias cudaIpcMemHandle_t = cudaIpcMemHandle_st;
    struct cudaIpcEventHandle_st
    {
        char[64] reserved;
    }
    alias cudaIpcEventHandle_t = cudaIpcEventHandle_st;
    struct cudaDeviceProp
    {
        char[256] name;
        CUuuid_st uuid;
        char[8] luid;
        uint luidDeviceNodeMask;
        c_ulong totalGlobalMem;
        c_ulong sharedMemPerBlock;
        int regsPerBlock;
        int warpSize;
        c_ulong memPitch;
        int maxThreadsPerBlock;
        int[3] maxThreadsDim;
        int[3] maxGridSize;
        int clockRate;
        c_ulong totalConstMem;
        int major;
        int minor;
        c_ulong textureAlignment;
        c_ulong texturePitchAlignment;
        int deviceOverlap;
        int multiProcessorCount;
        int kernelExecTimeoutEnabled;
        int integrated;
        int canMapHostMemory;
        int computeMode;
        int maxTexture1D;
        int maxTexture1DMipmap;
        int maxTexture1DLinear;
        int[2] maxTexture2D;
        int[2] maxTexture2DMipmap;
        int[3] maxTexture2DLinear;
        int[2] maxTexture2DGather;
        int[3] maxTexture3D;
        int[3] maxTexture3DAlt;
        int maxTextureCubemap;
        int[2] maxTexture1DLayered;
        int[3] maxTexture2DLayered;
        int[2] maxTextureCubemapLayered;
        int maxSurface1D;
        int[2] maxSurface2D;
        int[3] maxSurface3D;
        int[2] maxSurface1DLayered;
        int[3] maxSurface2DLayered;
        int maxSurfaceCubemap;
        int[2] maxSurfaceCubemapLayered;
        c_ulong surfaceAlignment;
        int concurrentKernels;
        int ECCEnabled;
        int pciBusID;
        int pciDeviceID;
        int pciDomainID;
        int tccDriver;
        int asyncEngineCount;
        int unifiedAddressing;
        int memoryClockRate;
        int memoryBusWidth;
        int l2CacheSize;
        int maxThreadsPerMultiProcessor;
        int streamPrioritiesSupported;
        int globalL1CacheSupported;
        int localL1CacheSupported;
        c_ulong sharedMemPerMultiprocessor;
        int regsPerMultiprocessor;
        int managedMemory;
        int isMultiGpuBoard;
        int multiGpuBoardGroupID;
        int hostNativeAtomicSupported;
        int singleToDoublePrecisionPerfRatio;
        int pageableMemoryAccess;
        int concurrentManagedAccess;
        int computePreemptionSupported;
        int canUseHostPointerForRegisteredMem;
        int cooperativeLaunch;
        int cooperativeMultiDeviceLaunch;
        c_ulong sharedMemPerBlockOptin;
        int pageableMemoryAccessUsesHostPageTables;
        int directManagedMemAccessFromHost;
    }
    alias cudaUUID_t = CUuuid_st;
    alias CUuuid = CUuuid_st;
    struct CUuuid_st
    {
        char[16] bytes;
    }
    enum cudaDeviceP2PAttr
    {
        cudaDevP2PAttrPerformanceRank = 1,
        cudaDevP2PAttrAccessSupported = 2,
        cudaDevP2PAttrNativeAtomicSupported = 3,
        cudaDevP2PAttrCudaArrayAccessSupported = 4,
    }
    enum cudaDevP2PAttrPerformanceRank = cudaDeviceP2PAttr.cudaDevP2PAttrPerformanceRank;
    enum cudaDevP2PAttrAccessSupported = cudaDeviceP2PAttr.cudaDevP2PAttrAccessSupported;
    enum cudaDevP2PAttrNativeAtomicSupported = cudaDeviceP2PAttr.cudaDevP2PAttrNativeAtomicSupported;
    enum cudaDevP2PAttrCudaArrayAccessSupported = cudaDeviceP2PAttr.cudaDevP2PAttrCudaArrayAccessSupported;
    enum cudaDeviceAttr
    {
        cudaDevAttrMaxThreadsPerBlock = 1,
        cudaDevAttrMaxBlockDimX = 2,
        cudaDevAttrMaxBlockDimY = 3,
        cudaDevAttrMaxBlockDimZ = 4,
        cudaDevAttrMaxGridDimX = 5,
        cudaDevAttrMaxGridDimY = 6,
        cudaDevAttrMaxGridDimZ = 7,
        cudaDevAttrMaxSharedMemoryPerBlock = 8,
        cudaDevAttrTotalConstantMemory = 9,
        cudaDevAttrWarpSize = 10,
        cudaDevAttrMaxPitch = 11,
        cudaDevAttrMaxRegistersPerBlock = 12,
        cudaDevAttrClockRate = 13,
        cudaDevAttrTextureAlignment = 14,
        cudaDevAttrGpuOverlap = 15,
        cudaDevAttrMultiProcessorCount = 16,
        cudaDevAttrKernelExecTimeout = 17,
        cudaDevAttrIntegrated = 18,
        cudaDevAttrCanMapHostMemory = 19,
        cudaDevAttrComputeMode = 20,
        cudaDevAttrMaxTexture1DWidth = 21,
        cudaDevAttrMaxTexture2DWidth = 22,
        cudaDevAttrMaxTexture2DHeight = 23,
        cudaDevAttrMaxTexture3DWidth = 24,
        cudaDevAttrMaxTexture3DHeight = 25,
        cudaDevAttrMaxTexture3DDepth = 26,
        cudaDevAttrMaxTexture2DLayeredWidth = 27,
        cudaDevAttrMaxTexture2DLayeredHeight = 28,
        cudaDevAttrMaxTexture2DLayeredLayers = 29,
        cudaDevAttrSurfaceAlignment = 30,
        cudaDevAttrConcurrentKernels = 31,
        cudaDevAttrEccEnabled = 32,
        cudaDevAttrPciBusId = 33,
        cudaDevAttrPciDeviceId = 34,
        cudaDevAttrTccDriver = 35,
        cudaDevAttrMemoryClockRate = 36,
        cudaDevAttrGlobalMemoryBusWidth = 37,
        cudaDevAttrL2CacheSize = 38,
        cudaDevAttrMaxThreadsPerMultiProcessor = 39,
        cudaDevAttrAsyncEngineCount = 40,
        cudaDevAttrUnifiedAddressing = 41,
        cudaDevAttrMaxTexture1DLayeredWidth = 42,
        cudaDevAttrMaxTexture1DLayeredLayers = 43,
        cudaDevAttrMaxTexture2DGatherWidth = 45,
        cudaDevAttrMaxTexture2DGatherHeight = 46,
        cudaDevAttrMaxTexture3DWidthAlt = 47,
        cudaDevAttrMaxTexture3DHeightAlt = 48,
        cudaDevAttrMaxTexture3DDepthAlt = 49,
        cudaDevAttrPciDomainId = 50,
        cudaDevAttrTexturePitchAlignment = 51,
        cudaDevAttrMaxTextureCubemapWidth = 52,
        cudaDevAttrMaxTextureCubemapLayeredWidth = 53,
        cudaDevAttrMaxTextureCubemapLayeredLayers = 54,
        cudaDevAttrMaxSurface1DWidth = 55,
        cudaDevAttrMaxSurface2DWidth = 56,
        cudaDevAttrMaxSurface2DHeight = 57,
        cudaDevAttrMaxSurface3DWidth = 58,
        cudaDevAttrMaxSurface3DHeight = 59,
        cudaDevAttrMaxSurface3DDepth = 60,
        cudaDevAttrMaxSurface1DLayeredWidth = 61,
        cudaDevAttrMaxSurface1DLayeredLayers = 62,
        cudaDevAttrMaxSurface2DLayeredWidth = 63,
        cudaDevAttrMaxSurface2DLayeredHeight = 64,
        cudaDevAttrMaxSurface2DLayeredLayers = 65,
        cudaDevAttrMaxSurfaceCubemapWidth = 66,
        cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67,
        cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,
        cudaDevAttrMaxTexture1DLinearWidth = 69,
        cudaDevAttrMaxTexture2DLinearWidth = 70,
        cudaDevAttrMaxTexture2DLinearHeight = 71,
        cudaDevAttrMaxTexture2DLinearPitch = 72,
        cudaDevAttrMaxTexture2DMipmappedWidth = 73,
        cudaDevAttrMaxTexture2DMipmappedHeight = 74,
        cudaDevAttrComputeCapabilityMajor = 75,
        cudaDevAttrComputeCapabilityMinor = 76,
        cudaDevAttrMaxTexture1DMipmappedWidth = 77,
        cudaDevAttrStreamPrioritiesSupported = 78,
        cudaDevAttrGlobalL1CacheSupported = 79,
        cudaDevAttrLocalL1CacheSupported = 80,
        cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
        cudaDevAttrMaxRegistersPerMultiprocessor = 82,
        cudaDevAttrManagedMemory = 83,
        cudaDevAttrIsMultiGpuBoard = 84,
        cudaDevAttrMultiGpuBoardGroupID = 85,
        cudaDevAttrHostNativeAtomicSupported = 86,
        cudaDevAttrSingleToDoublePrecisionPerfRatio = 87,
        cudaDevAttrPageableMemoryAccess = 88,
        cudaDevAttrConcurrentManagedAccess = 89,
        cudaDevAttrComputePreemptionSupported = 90,
        cudaDevAttrCanUseHostPointerForRegisteredMem = 91,
        cudaDevAttrReserved92 = 92,
        cudaDevAttrReserved93 = 93,
        cudaDevAttrReserved94 = 94,
        cudaDevAttrCooperativeLaunch = 95,
        cudaDevAttrCooperativeMultiDeviceLaunch = 96,
        cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
        cudaDevAttrCanFlushRemoteWrites = 98,
        cudaDevAttrHostRegisterSupported = 99,
        cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100,
        cudaDevAttrDirectManagedMemAccessFromHost = 101,
    }
    enum cudaDevAttrMaxThreadsPerBlock = cudaDeviceAttr.cudaDevAttrMaxThreadsPerBlock;
    enum cudaDevAttrMaxBlockDimX = cudaDeviceAttr.cudaDevAttrMaxBlockDimX;
    enum cudaDevAttrMaxBlockDimY = cudaDeviceAttr.cudaDevAttrMaxBlockDimY;
    enum cudaDevAttrMaxBlockDimZ = cudaDeviceAttr.cudaDevAttrMaxBlockDimZ;
    enum cudaDevAttrMaxGridDimX = cudaDeviceAttr.cudaDevAttrMaxGridDimX;
    enum cudaDevAttrMaxGridDimY = cudaDeviceAttr.cudaDevAttrMaxGridDimY;
    enum cudaDevAttrMaxGridDimZ = cudaDeviceAttr.cudaDevAttrMaxGridDimZ;
    enum cudaDevAttrMaxSharedMemoryPerBlock = cudaDeviceAttr.cudaDevAttrMaxSharedMemoryPerBlock;
    enum cudaDevAttrTotalConstantMemory = cudaDeviceAttr.cudaDevAttrTotalConstantMemory;
    enum cudaDevAttrWarpSize = cudaDeviceAttr.cudaDevAttrWarpSize;
    enum cudaDevAttrMaxPitch = cudaDeviceAttr.cudaDevAttrMaxPitch;
    enum cudaDevAttrMaxRegistersPerBlock = cudaDeviceAttr.cudaDevAttrMaxRegistersPerBlock;
    enum cudaDevAttrClockRate = cudaDeviceAttr.cudaDevAttrClockRate;
    enum cudaDevAttrTextureAlignment = cudaDeviceAttr.cudaDevAttrTextureAlignment;
    enum cudaDevAttrGpuOverlap = cudaDeviceAttr.cudaDevAttrGpuOverlap;
    enum cudaDevAttrMultiProcessorCount = cudaDeviceAttr.cudaDevAttrMultiProcessorCount;
    enum cudaDevAttrKernelExecTimeout = cudaDeviceAttr.cudaDevAttrKernelExecTimeout;
    enum cudaDevAttrIntegrated = cudaDeviceAttr.cudaDevAttrIntegrated;
    enum cudaDevAttrCanMapHostMemory = cudaDeviceAttr.cudaDevAttrCanMapHostMemory;
    enum cudaDevAttrComputeMode = cudaDeviceAttr.cudaDevAttrComputeMode;
    enum cudaDevAttrMaxTexture1DWidth = cudaDeviceAttr.cudaDevAttrMaxTexture1DWidth;
    enum cudaDevAttrMaxTexture2DWidth = cudaDeviceAttr.cudaDevAttrMaxTexture2DWidth;
    enum cudaDevAttrMaxTexture2DHeight = cudaDeviceAttr.cudaDevAttrMaxTexture2DHeight;
    enum cudaDevAttrMaxTexture3DWidth = cudaDeviceAttr.cudaDevAttrMaxTexture3DWidth;
    enum cudaDevAttrMaxTexture3DHeight = cudaDeviceAttr.cudaDevAttrMaxTexture3DHeight;
    enum cudaDevAttrMaxTexture3DDepth = cudaDeviceAttr.cudaDevAttrMaxTexture3DDepth;
    enum cudaDevAttrMaxTexture2DLayeredWidth = cudaDeviceAttr.cudaDevAttrMaxTexture2DLayeredWidth;
    enum cudaDevAttrMaxTexture2DLayeredHeight = cudaDeviceAttr.cudaDevAttrMaxTexture2DLayeredHeight;
    enum cudaDevAttrMaxTexture2DLayeredLayers = cudaDeviceAttr.cudaDevAttrMaxTexture2DLayeredLayers;
    enum cudaDevAttrSurfaceAlignment = cudaDeviceAttr.cudaDevAttrSurfaceAlignment;
    enum cudaDevAttrConcurrentKernels = cudaDeviceAttr.cudaDevAttrConcurrentKernels;
    enum cudaDevAttrEccEnabled = cudaDeviceAttr.cudaDevAttrEccEnabled;
    enum cudaDevAttrPciBusId = cudaDeviceAttr.cudaDevAttrPciBusId;
    enum cudaDevAttrPciDeviceId = cudaDeviceAttr.cudaDevAttrPciDeviceId;
    enum cudaDevAttrTccDriver = cudaDeviceAttr.cudaDevAttrTccDriver;
    enum cudaDevAttrMemoryClockRate = cudaDeviceAttr.cudaDevAttrMemoryClockRate;
    enum cudaDevAttrGlobalMemoryBusWidth = cudaDeviceAttr.cudaDevAttrGlobalMemoryBusWidth;
    enum cudaDevAttrL2CacheSize = cudaDeviceAttr.cudaDevAttrL2CacheSize;
    enum cudaDevAttrMaxThreadsPerMultiProcessor = cudaDeviceAttr.cudaDevAttrMaxThreadsPerMultiProcessor;
    enum cudaDevAttrAsyncEngineCount = cudaDeviceAttr.cudaDevAttrAsyncEngineCount;
    enum cudaDevAttrUnifiedAddressing = cudaDeviceAttr.cudaDevAttrUnifiedAddressing;
    enum cudaDevAttrMaxTexture1DLayeredWidth = cudaDeviceAttr.cudaDevAttrMaxTexture1DLayeredWidth;
    enum cudaDevAttrMaxTexture1DLayeredLayers = cudaDeviceAttr.cudaDevAttrMaxTexture1DLayeredLayers;
    enum cudaDevAttrMaxTexture2DGatherWidth = cudaDeviceAttr.cudaDevAttrMaxTexture2DGatherWidth;
    enum cudaDevAttrMaxTexture2DGatherHeight = cudaDeviceAttr.cudaDevAttrMaxTexture2DGatherHeight;
    enum cudaDevAttrMaxTexture3DWidthAlt = cudaDeviceAttr.cudaDevAttrMaxTexture3DWidthAlt;
    enum cudaDevAttrMaxTexture3DHeightAlt = cudaDeviceAttr.cudaDevAttrMaxTexture3DHeightAlt;
    enum cudaDevAttrMaxTexture3DDepthAlt = cudaDeviceAttr.cudaDevAttrMaxTexture3DDepthAlt;
    enum cudaDevAttrPciDomainId = cudaDeviceAttr.cudaDevAttrPciDomainId;
    enum cudaDevAttrTexturePitchAlignment = cudaDeviceAttr.cudaDevAttrTexturePitchAlignment;
    enum cudaDevAttrMaxTextureCubemapWidth = cudaDeviceAttr.cudaDevAttrMaxTextureCubemapWidth;
    enum cudaDevAttrMaxTextureCubemapLayeredWidth = cudaDeviceAttr.cudaDevAttrMaxTextureCubemapLayeredWidth;
    enum cudaDevAttrMaxTextureCubemapLayeredLayers = cudaDeviceAttr.cudaDevAttrMaxTextureCubemapLayeredLayers;
    enum cudaDevAttrMaxSurface1DWidth = cudaDeviceAttr.cudaDevAttrMaxSurface1DWidth;
    enum cudaDevAttrMaxSurface2DWidth = cudaDeviceAttr.cudaDevAttrMaxSurface2DWidth;
    enum cudaDevAttrMaxSurface2DHeight = cudaDeviceAttr.cudaDevAttrMaxSurface2DHeight;
    enum cudaDevAttrMaxSurface3DWidth = cudaDeviceAttr.cudaDevAttrMaxSurface3DWidth;
    enum cudaDevAttrMaxSurface3DHeight = cudaDeviceAttr.cudaDevAttrMaxSurface3DHeight;
    enum cudaDevAttrMaxSurface3DDepth = cudaDeviceAttr.cudaDevAttrMaxSurface3DDepth;
    enum cudaDevAttrMaxSurface1DLayeredWidth = cudaDeviceAttr.cudaDevAttrMaxSurface1DLayeredWidth;
    enum cudaDevAttrMaxSurface1DLayeredLayers = cudaDeviceAttr.cudaDevAttrMaxSurface1DLayeredLayers;
    enum cudaDevAttrMaxSurface2DLayeredWidth = cudaDeviceAttr.cudaDevAttrMaxSurface2DLayeredWidth;
    enum cudaDevAttrMaxSurface2DLayeredHeight = cudaDeviceAttr.cudaDevAttrMaxSurface2DLayeredHeight;
    enum cudaDevAttrMaxSurface2DLayeredLayers = cudaDeviceAttr.cudaDevAttrMaxSurface2DLayeredLayers;
    enum cudaDevAttrMaxSurfaceCubemapWidth = cudaDeviceAttr.cudaDevAttrMaxSurfaceCubemapWidth;
    enum cudaDevAttrMaxSurfaceCubemapLayeredWidth = cudaDeviceAttr.cudaDevAttrMaxSurfaceCubemapLayeredWidth;
    enum cudaDevAttrMaxSurfaceCubemapLayeredLayers = cudaDeviceAttr.cudaDevAttrMaxSurfaceCubemapLayeredLayers;
    enum cudaDevAttrMaxTexture1DLinearWidth = cudaDeviceAttr.cudaDevAttrMaxTexture1DLinearWidth;
    enum cudaDevAttrMaxTexture2DLinearWidth = cudaDeviceAttr.cudaDevAttrMaxTexture2DLinearWidth;
    enum cudaDevAttrMaxTexture2DLinearHeight = cudaDeviceAttr.cudaDevAttrMaxTexture2DLinearHeight;
    enum cudaDevAttrMaxTexture2DLinearPitch = cudaDeviceAttr.cudaDevAttrMaxTexture2DLinearPitch;
    enum cudaDevAttrMaxTexture2DMipmappedWidth = cudaDeviceAttr.cudaDevAttrMaxTexture2DMipmappedWidth;
    enum cudaDevAttrMaxTexture2DMipmappedHeight = cudaDeviceAttr.cudaDevAttrMaxTexture2DMipmappedHeight;
    enum cudaDevAttrComputeCapabilityMajor = cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor;
    enum cudaDevAttrComputeCapabilityMinor = cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor;
    enum cudaDevAttrMaxTexture1DMipmappedWidth = cudaDeviceAttr.cudaDevAttrMaxTexture1DMipmappedWidth;
    enum cudaDevAttrStreamPrioritiesSupported = cudaDeviceAttr.cudaDevAttrStreamPrioritiesSupported;
    enum cudaDevAttrGlobalL1CacheSupported = cudaDeviceAttr.cudaDevAttrGlobalL1CacheSupported;
    enum cudaDevAttrLocalL1CacheSupported = cudaDeviceAttr.cudaDevAttrLocalL1CacheSupported;
    enum cudaDevAttrMaxSharedMemoryPerMultiprocessor = cudaDeviceAttr.cudaDevAttrMaxSharedMemoryPerMultiprocessor;
    enum cudaDevAttrMaxRegistersPerMultiprocessor = cudaDeviceAttr.cudaDevAttrMaxRegistersPerMultiprocessor;
    enum cudaDevAttrManagedMemory = cudaDeviceAttr.cudaDevAttrManagedMemory;
    enum cudaDevAttrIsMultiGpuBoard = cudaDeviceAttr.cudaDevAttrIsMultiGpuBoard;
    enum cudaDevAttrMultiGpuBoardGroupID = cudaDeviceAttr.cudaDevAttrMultiGpuBoardGroupID;
    enum cudaDevAttrHostNativeAtomicSupported = cudaDeviceAttr.cudaDevAttrHostNativeAtomicSupported;
    enum cudaDevAttrSingleToDoublePrecisionPerfRatio = cudaDeviceAttr.cudaDevAttrSingleToDoublePrecisionPerfRatio;
    enum cudaDevAttrPageableMemoryAccess = cudaDeviceAttr.cudaDevAttrPageableMemoryAccess;
    enum cudaDevAttrConcurrentManagedAccess = cudaDeviceAttr.cudaDevAttrConcurrentManagedAccess;
    enum cudaDevAttrComputePreemptionSupported = cudaDeviceAttr.cudaDevAttrComputePreemptionSupported;
    enum cudaDevAttrCanUseHostPointerForRegisteredMem = cudaDeviceAttr.cudaDevAttrCanUseHostPointerForRegisteredMem;
    enum cudaDevAttrReserved92 = cudaDeviceAttr.cudaDevAttrReserved92;
    enum cudaDevAttrReserved93 = cudaDeviceAttr.cudaDevAttrReserved93;
    enum cudaDevAttrReserved94 = cudaDeviceAttr.cudaDevAttrReserved94;
    enum cudaDevAttrCooperativeLaunch = cudaDeviceAttr.cudaDevAttrCooperativeLaunch;
    enum cudaDevAttrCooperativeMultiDeviceLaunch = cudaDeviceAttr.cudaDevAttrCooperativeMultiDeviceLaunch;
    enum cudaDevAttrMaxSharedMemoryPerBlockOptin = cudaDeviceAttr.cudaDevAttrMaxSharedMemoryPerBlockOptin;
    enum cudaDevAttrCanFlushRemoteWrites = cudaDeviceAttr.cudaDevAttrCanFlushRemoteWrites;
    enum cudaDevAttrHostRegisterSupported = cudaDeviceAttr.cudaDevAttrHostRegisterSupported;
    enum cudaDevAttrPageableMemoryAccessUsesHostPageTables = cudaDeviceAttr.cudaDevAttrPageableMemoryAccessUsesHostPageTables;
    enum cudaDevAttrDirectManagedMemAccessFromHost = cudaDeviceAttr.cudaDevAttrDirectManagedMemAccessFromHost;
    enum cudaOutputMode
    {
        cudaKeyValuePair = 0,
        cudaCSV = 1,
    }
    enum cudaKeyValuePair = cudaOutputMode.cudaKeyValuePair;
    enum cudaCSV = cudaOutputMode.cudaCSV;
    enum cudaMemRangeAttribute
    {
        cudaMemRangeAttributeReadMostly = 1,
        cudaMemRangeAttributePreferredLocation = 2,
        cudaMemRangeAttributeAccessedBy = 3,
        cudaMemRangeAttributeLastPrefetchLocation = 4,
    }
    enum cudaMemRangeAttributeReadMostly = cudaMemRangeAttribute.cudaMemRangeAttributeReadMostly;
    enum cudaMemRangeAttributePreferredLocation = cudaMemRangeAttribute.cudaMemRangeAttributePreferredLocation;
    enum cudaMemRangeAttributeAccessedBy = cudaMemRangeAttribute.cudaMemRangeAttributeAccessedBy;
    enum cudaMemRangeAttributeLastPrefetchLocation = cudaMemRangeAttribute.cudaMemRangeAttributeLastPrefetchLocation;
    enum cudaMemoryAdvise
    {
        cudaMemAdviseSetReadMostly = 1,
        cudaMemAdviseUnsetReadMostly = 2,
        cudaMemAdviseSetPreferredLocation = 3,
        cudaMemAdviseUnsetPreferredLocation = 4,
        cudaMemAdviseSetAccessedBy = 5,
        cudaMemAdviseUnsetAccessedBy = 6,
    }
    enum cudaMemAdviseSetReadMostly = cudaMemoryAdvise.cudaMemAdviseSetReadMostly;
    enum cudaMemAdviseUnsetReadMostly = cudaMemoryAdvise.cudaMemAdviseUnsetReadMostly;
    enum cudaMemAdviseSetPreferredLocation = cudaMemoryAdvise.cudaMemAdviseSetPreferredLocation;
    enum cudaMemAdviseUnsetPreferredLocation = cudaMemoryAdvise.cudaMemAdviseUnsetPreferredLocation;
    enum cudaMemAdviseSetAccessedBy = cudaMemoryAdvise.cudaMemAdviseSetAccessedBy;
    enum cudaMemAdviseUnsetAccessedBy = cudaMemoryAdvise.cudaMemAdviseUnsetAccessedBy;
    enum cudaLimit
    {
        cudaLimitStackSize = 0,
        cudaLimitPrintfFifoSize = 1,
        cudaLimitMallocHeapSize = 2,
        cudaLimitDevRuntimeSyncDepth = 3,
        cudaLimitDevRuntimePendingLaunchCount = 4,
        cudaLimitMaxL2FetchGranularity = 5,
    }
    enum cudaLimitStackSize = cudaLimit.cudaLimitStackSize;
    enum cudaLimitPrintfFifoSize = cudaLimit.cudaLimitPrintfFifoSize;
    enum cudaLimitMallocHeapSize = cudaLimit.cudaLimitMallocHeapSize;
    enum cudaLimitDevRuntimeSyncDepth = cudaLimit.cudaLimitDevRuntimeSyncDepth;
    enum cudaLimitDevRuntimePendingLaunchCount = cudaLimit.cudaLimitDevRuntimePendingLaunchCount;
    enum cudaLimitMaxL2FetchGranularity = cudaLimit.cudaLimitMaxL2FetchGranularity;
    enum cudaComputeMode
    {
        cudaComputeModeDefault = 0,
        cudaComputeModeExclusive = 1,
        cudaComputeModeProhibited = 2,
        cudaComputeModeExclusiveProcess = 3,
    }
    enum cudaComputeModeDefault = cudaComputeMode.cudaComputeModeDefault;
    enum cudaComputeModeExclusive = cudaComputeMode.cudaComputeModeExclusive;
    enum cudaComputeModeProhibited = cudaComputeMode.cudaComputeModeProhibited;
    enum cudaComputeModeExclusiveProcess = cudaComputeMode.cudaComputeModeExclusiveProcess;
    enum cudaSharedCarveout
    {
        cudaSharedmemCarveoutDefault = -1,
        cudaSharedmemCarveoutMaxShared = 100,
        cudaSharedmemCarveoutMaxL1 = 0,
    }
    enum cudaSharedmemCarveoutDefault = cudaSharedCarveout.cudaSharedmemCarveoutDefault;
    enum cudaSharedmemCarveoutMaxShared = cudaSharedCarveout.cudaSharedmemCarveoutMaxShared;
    enum cudaSharedmemCarveoutMaxL1 = cudaSharedCarveout.cudaSharedmemCarveoutMaxL1;
    enum cudaSharedMemConfig
    {
        cudaSharedMemBankSizeDefault = 0,
        cudaSharedMemBankSizeFourByte = 1,
        cudaSharedMemBankSizeEightByte = 2,
    }
    enum cudaSharedMemBankSizeDefault = cudaSharedMemConfig.cudaSharedMemBankSizeDefault;
    enum cudaSharedMemBankSizeFourByte = cudaSharedMemConfig.cudaSharedMemBankSizeFourByte;
    enum cudaSharedMemBankSizeEightByte = cudaSharedMemConfig.cudaSharedMemBankSizeEightByte;
    enum cudaFuncCache
    {
        cudaFuncCachePreferNone = 0,
        cudaFuncCachePreferShared = 1,
        cudaFuncCachePreferL1 = 2,
        cudaFuncCachePreferEqual = 3,
    }
    enum cudaFuncCachePreferNone = cudaFuncCache.cudaFuncCachePreferNone;
    enum cudaFuncCachePreferShared = cudaFuncCache.cudaFuncCachePreferShared;
    enum cudaFuncCachePreferL1 = cudaFuncCache.cudaFuncCachePreferL1;
    enum cudaFuncCachePreferEqual = cudaFuncCache.cudaFuncCachePreferEqual;
    enum cudaFuncAttribute
    {
        cudaFuncAttributeMaxDynamicSharedMemorySize = 8,
        cudaFuncAttributePreferredSharedMemoryCarveout = 9,
        cudaFuncAttributeMax = 10,
    }
    enum cudaFuncAttributeMaxDynamicSharedMemorySize = cudaFuncAttribute.cudaFuncAttributeMaxDynamicSharedMemorySize;
    enum cudaFuncAttributePreferredSharedMemoryCarveout = cudaFuncAttribute.cudaFuncAttributePreferredSharedMemoryCarveout;
    enum cudaFuncAttributeMax = cudaFuncAttribute.cudaFuncAttributeMax;
    struct cudaFuncAttributes
    {
        c_ulong sharedSizeBytes;
        c_ulong constSizeBytes;
        c_ulong localSizeBytes;
        int maxThreadsPerBlock;
        int numRegs;
        int ptxVersion;
        int binaryVersion;
        int cacheModeCA;
        int maxDynamicSharedSizeBytes;
        int preferredShmemCarveout;
    }
    struct cudaPointerAttributes
    {
        cudaMemoryType memoryType;
        cudaMemoryType type;
        int device;
        void* devicePointer;
        void* hostPointer;
        int isManaged;
    }
    struct cudaResourceViewDesc
    {
        cudaResourceViewFormat format;
        c_ulong width;
        c_ulong height;
        c_ulong depth;
        uint firstMipmapLevel;
        uint lastMipmapLevel;
        uint firstLayer;
        uint lastLayer;
    }
    struct cudaResourceDesc
    {
        cudaResourceType resType;
        static union _Anonymous_12
        {
            static struct _Anonymous_13
            {
                cudaArray* array;
            }
            _Anonymous_13 array;
            static struct _Anonymous_14
            {
                cudaMipmappedArray* mipmap;
            }
            _Anonymous_14 mipmap;
            static struct _Anonymous_15
            {
                void* devPtr;
                cudaChannelFormatDesc desc;
                c_ulong sizeInBytes;
            }
            _Anonymous_15 linear;
            static struct _Anonymous_16
            {
                void* devPtr;
                cudaChannelFormatDesc desc;
                c_ulong width;
                c_ulong height;
                c_ulong pitchInBytes;
            }
            _Anonymous_16 pitch2D;
        }
        _Anonymous_12 res;
    }
    enum cudaResourceViewFormat
    {
        cudaResViewFormatNone = 0,
        cudaResViewFormatUnsignedChar1 = 1,
        cudaResViewFormatUnsignedChar2 = 2,
        cudaResViewFormatUnsignedChar4 = 3,
        cudaResViewFormatSignedChar1 = 4,
        cudaResViewFormatSignedChar2 = 5,
        cudaResViewFormatSignedChar4 = 6,
        cudaResViewFormatUnsignedShort1 = 7,
        cudaResViewFormatUnsignedShort2 = 8,
        cudaResViewFormatUnsignedShort4 = 9,
        cudaResViewFormatSignedShort1 = 10,
        cudaResViewFormatSignedShort2 = 11,
        cudaResViewFormatSignedShort4 = 12,
        cudaResViewFormatUnsignedInt1 = 13,
        cudaResViewFormatUnsignedInt2 = 14,
        cudaResViewFormatUnsignedInt4 = 15,
        cudaResViewFormatSignedInt1 = 16,
        cudaResViewFormatSignedInt2 = 17,
        cudaResViewFormatSignedInt4 = 18,
        cudaResViewFormatHalf1 = 19,
        cudaResViewFormatHalf2 = 20,
        cudaResViewFormatHalf4 = 21,
        cudaResViewFormatFloat1 = 22,
        cudaResViewFormatFloat2 = 23,
        cudaResViewFormatFloat4 = 24,
        cudaResViewFormatUnsignedBlockCompressed1 = 25,
        cudaResViewFormatUnsignedBlockCompressed2 = 26,
        cudaResViewFormatUnsignedBlockCompressed3 = 27,
        cudaResViewFormatUnsignedBlockCompressed4 = 28,
        cudaResViewFormatSignedBlockCompressed4 = 29,
        cudaResViewFormatUnsignedBlockCompressed5 = 30,
        cudaResViewFormatSignedBlockCompressed5 = 31,
        cudaResViewFormatUnsignedBlockCompressed6H = 32,
        cudaResViewFormatSignedBlockCompressed6H = 33,
        cudaResViewFormatUnsignedBlockCompressed7 = 34,
    }
    enum cudaResViewFormatNone = cudaResourceViewFormat.cudaResViewFormatNone;
    enum cudaResViewFormatUnsignedChar1 = cudaResourceViewFormat.cudaResViewFormatUnsignedChar1;
    enum cudaResViewFormatUnsignedChar2 = cudaResourceViewFormat.cudaResViewFormatUnsignedChar2;
    enum cudaResViewFormatUnsignedChar4 = cudaResourceViewFormat.cudaResViewFormatUnsignedChar4;
    enum cudaResViewFormatSignedChar1 = cudaResourceViewFormat.cudaResViewFormatSignedChar1;
    enum cudaResViewFormatSignedChar2 = cudaResourceViewFormat.cudaResViewFormatSignedChar2;
    enum cudaResViewFormatSignedChar4 = cudaResourceViewFormat.cudaResViewFormatSignedChar4;
    enum cudaResViewFormatUnsignedShort1 = cudaResourceViewFormat.cudaResViewFormatUnsignedShort1;
    enum cudaResViewFormatUnsignedShort2 = cudaResourceViewFormat.cudaResViewFormatUnsignedShort2;
    enum cudaResViewFormatUnsignedShort4 = cudaResourceViewFormat.cudaResViewFormatUnsignedShort4;
    enum cudaResViewFormatSignedShort1 = cudaResourceViewFormat.cudaResViewFormatSignedShort1;
    enum cudaResViewFormatSignedShort2 = cudaResourceViewFormat.cudaResViewFormatSignedShort2;
    enum cudaResViewFormatSignedShort4 = cudaResourceViewFormat.cudaResViewFormatSignedShort4;
    enum cudaResViewFormatUnsignedInt1 = cudaResourceViewFormat.cudaResViewFormatUnsignedInt1;
    enum cudaResViewFormatUnsignedInt2 = cudaResourceViewFormat.cudaResViewFormatUnsignedInt2;
    enum cudaResViewFormatUnsignedInt4 = cudaResourceViewFormat.cudaResViewFormatUnsignedInt4;
    enum cudaResViewFormatSignedInt1 = cudaResourceViewFormat.cudaResViewFormatSignedInt1;
    enum cudaResViewFormatSignedInt2 = cudaResourceViewFormat.cudaResViewFormatSignedInt2;
    enum cudaResViewFormatSignedInt4 = cudaResourceViewFormat.cudaResViewFormatSignedInt4;
    enum cudaResViewFormatHalf1 = cudaResourceViewFormat.cudaResViewFormatHalf1;
    enum cudaResViewFormatHalf2 = cudaResourceViewFormat.cudaResViewFormatHalf2;
    enum cudaResViewFormatHalf4 = cudaResourceViewFormat.cudaResViewFormatHalf4;
    enum cudaResViewFormatFloat1 = cudaResourceViewFormat.cudaResViewFormatFloat1;
    enum cudaResViewFormatFloat2 = cudaResourceViewFormat.cudaResViewFormatFloat2;
    enum cudaResViewFormatFloat4 = cudaResourceViewFormat.cudaResViewFormatFloat4;
    enum cudaResViewFormatUnsignedBlockCompressed1 = cudaResourceViewFormat.cudaResViewFormatUnsignedBlockCompressed1;
    enum cudaResViewFormatUnsignedBlockCompressed2 = cudaResourceViewFormat.cudaResViewFormatUnsignedBlockCompressed2;
    enum cudaResViewFormatUnsignedBlockCompressed3 = cudaResourceViewFormat.cudaResViewFormatUnsignedBlockCompressed3;
    enum cudaResViewFormatUnsignedBlockCompressed4 = cudaResourceViewFormat.cudaResViewFormatUnsignedBlockCompressed4;
    enum cudaResViewFormatSignedBlockCompressed4 = cudaResourceViewFormat.cudaResViewFormatSignedBlockCompressed4;
    enum cudaResViewFormatUnsignedBlockCompressed5 = cudaResourceViewFormat.cudaResViewFormatUnsignedBlockCompressed5;
    enum cudaResViewFormatSignedBlockCompressed5 = cudaResourceViewFormat.cudaResViewFormatSignedBlockCompressed5;
    enum cudaResViewFormatUnsignedBlockCompressed6H = cudaResourceViewFormat.cudaResViewFormatUnsignedBlockCompressed6H;
    enum cudaResViewFormatSignedBlockCompressed6H = cudaResourceViewFormat.cudaResViewFormatSignedBlockCompressed6H;
    enum cudaResViewFormatUnsignedBlockCompressed7 = cudaResourceViewFormat.cudaResViewFormatUnsignedBlockCompressed7;
    enum cudaResourceType
    {
        cudaResourceTypeArray = 0,
        cudaResourceTypeMipmappedArray = 1,
        cudaResourceTypeLinear = 2,
        cudaResourceTypePitch2D = 3,
    }
    enum cudaResourceTypeArray = cudaResourceType.cudaResourceTypeArray;
    enum cudaResourceTypeMipmappedArray = cudaResourceType.cudaResourceTypeMipmappedArray;
    enum cudaResourceTypeLinear = cudaResourceType.cudaResourceTypeLinear;
    enum cudaResourceTypePitch2D = cudaResourceType.cudaResourceTypePitch2D;
    enum cudaGraphicsCubeFace
    {
        cudaGraphicsCubeFacePositiveX = 0,
        cudaGraphicsCubeFaceNegativeX = 1,
        cudaGraphicsCubeFacePositiveY = 2,
        cudaGraphicsCubeFaceNegativeY = 3,
        cudaGraphicsCubeFacePositiveZ = 4,
        cudaGraphicsCubeFaceNegativeZ = 5,
    }
    enum cudaGraphicsCubeFacePositiveX = cudaGraphicsCubeFace.cudaGraphicsCubeFacePositiveX;
    enum cudaGraphicsCubeFaceNegativeX = cudaGraphicsCubeFace.cudaGraphicsCubeFaceNegativeX;
    enum cudaGraphicsCubeFacePositiveY = cudaGraphicsCubeFace.cudaGraphicsCubeFacePositiveY;
    enum cudaGraphicsCubeFaceNegativeY = cudaGraphicsCubeFace.cudaGraphicsCubeFaceNegativeY;
    enum cudaGraphicsCubeFacePositiveZ = cudaGraphicsCubeFace.cudaGraphicsCubeFacePositiveZ;
    enum cudaGraphicsCubeFaceNegativeZ = cudaGraphicsCubeFace.cudaGraphicsCubeFaceNegativeZ;
    struct max_align_t
    {
        long __clang_max_align_nonce1;
        real __clang_max_align_nonce2;
    }
    enum cudaGraphicsMapFlags
    {
        cudaGraphicsMapFlagsNone = 0,
        cudaGraphicsMapFlagsReadOnly = 1,
        cudaGraphicsMapFlagsWriteDiscard = 2,
    }
    enum cudaGraphicsMapFlagsNone = cudaGraphicsMapFlags.cudaGraphicsMapFlagsNone;
    enum cudaGraphicsMapFlagsReadOnly = cudaGraphicsMapFlags.cudaGraphicsMapFlagsReadOnly;
    enum cudaGraphicsMapFlagsWriteDiscard = cudaGraphicsMapFlags.cudaGraphicsMapFlagsWriteDiscard;
    enum cudaGraphicsRegisterFlags
    {
        cudaGraphicsRegisterFlagsNone = 0,
        cudaGraphicsRegisterFlagsReadOnly = 1,
        cudaGraphicsRegisterFlagsWriteDiscard = 2,
        cudaGraphicsRegisterFlagsSurfaceLoadStore = 4,
        cudaGraphicsRegisterFlagsTextureGather = 8,
    }
    enum cudaGraphicsRegisterFlagsNone = cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsNone;
    enum cudaGraphicsRegisterFlagsReadOnly = cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly;
    enum cudaGraphicsRegisterFlagsWriteDiscard = cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard;
    enum cudaGraphicsRegisterFlagsSurfaceLoadStore = cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsSurfaceLoadStore;
    enum cudaGraphicsRegisterFlagsTextureGather = cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsTextureGather;
    struct cudaGraphicsResource;
    enum cudaStreamCaptureMode
    {
        cudaStreamCaptureModeGlobal = 0,
        cudaStreamCaptureModeThreadLocal = 1,
        cudaStreamCaptureModeRelaxed = 2,
    }
    enum cudaStreamCaptureModeGlobal = cudaStreamCaptureMode.cudaStreamCaptureModeGlobal;
    enum cudaStreamCaptureModeThreadLocal = cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal;
    enum cudaStreamCaptureModeRelaxed = cudaStreamCaptureMode.cudaStreamCaptureModeRelaxed;
    enum cudaStreamCaptureStatus
    {
        cudaStreamCaptureStatusNone = 0,
        cudaStreamCaptureStatusActive = 1,
        cudaStreamCaptureStatusInvalidated = 2,
    }
    enum cudaStreamCaptureStatusNone = cudaStreamCaptureStatus.cudaStreamCaptureStatusNone;
    enum cudaStreamCaptureStatusActive = cudaStreamCaptureStatus.cudaStreamCaptureStatusActive;
    enum cudaStreamCaptureStatusInvalidated = cudaStreamCaptureStatus.cudaStreamCaptureStatusInvalidated;
    struct cudaHostNodeParams
    {
        void function(void*) fn;
        void* userData;
    }
    alias cudaHostFn_t = void function(void*);
    struct cudaMemsetParams
    {
        void* dst;
        c_ulong pitch;
        uint value;
        uint elementSize;
        c_ulong width;
        c_ulong height;
    }
    struct cudaMemcpy3DPeerParms
    {
        cudaArray* srcArray;
        cudaPos srcPos;
        cudaPitchedPtr srcPtr;
        int srcDevice;
        cudaArray* dstArray;
        cudaPos dstPos;
        cudaPitchedPtr dstPtr;
        int dstDevice;
        cudaExtent extent;
    }
    struct cudaMemcpy3DParms
    {
        cudaArray* srcArray;
        cudaPos srcPos;
        cudaPitchedPtr srcPtr;
        cudaArray* dstArray;
        cudaPos dstPos;
        cudaPitchedPtr dstPtr;
        cudaExtent extent;
        cudaMemcpyKind kind;
    }
    struct cudaPos
    {
        c_ulong x;
        c_ulong y;
        c_ulong z;
    }
    struct cudaExtent
    {
        c_ulong width;
        c_ulong height;
        c_ulong depth;
    }
    struct cudaPitchedPtr
    {
        void* ptr;
        c_ulong pitch;
        c_ulong xsize;
        c_ulong ysize;
    }
    enum cudaMemcpyKind
    {
        cudaMemcpyHostToHost = 0,
        cudaMemcpyHostToDevice = 1,
        cudaMemcpyDeviceToHost = 2,
        cudaMemcpyDeviceToDevice = 3,
        cudaMemcpyDefault = 4,
    }
    enum cudaMemcpyHostToHost = cudaMemcpyKind.cudaMemcpyHostToHost;
    enum cudaMemcpyHostToDevice = cudaMemcpyKind.cudaMemcpyHostToDevice;
    enum cudaMemcpyDeviceToHost = cudaMemcpyKind.cudaMemcpyDeviceToHost;
    enum cudaMemcpyDeviceToDevice = cudaMemcpyKind.cudaMemcpyDeviceToDevice;
    enum cudaMemcpyDefault = cudaMemcpyKind.cudaMemcpyDefault;
    alias ptrdiff_t = c_long;
    alias size_t = c_ulong;
    enum cudaMemoryType
    {
        cudaMemoryTypeUnregistered = 0,
        cudaMemoryTypeHost = 1,
        cudaMemoryTypeDevice = 2,
        cudaMemoryTypeManaged = 3,
    }
    enum cudaMemoryTypeUnregistered = cudaMemoryType.cudaMemoryTypeUnregistered;
    enum cudaMemoryTypeHost = cudaMemoryType.cudaMemoryTypeHost;
    enum cudaMemoryTypeDevice = cudaMemoryType.cudaMemoryTypeDevice;
    enum cudaMemoryTypeManaged = cudaMemoryType.cudaMemoryTypeManaged;
    alias wchar_t = int;
    alias cudaMipmappedArray_const_t = const(cudaMipmappedArray)*;
    struct cudaMipmappedArray;
    alias cudaMipmappedArray_t = cudaMipmappedArray*;
    alias cudaArray_const_t = const(cudaArray)*;
    struct cudaArray;
    alias cudaArray_t = cudaArray*;
    struct cudaChannelFormatDesc
    {
        int x;
        int y;
        int z;
        int w;
        cudaChannelFormatKind f;
    }
    enum cudaChannelFormatKind
    {
        cudaChannelFormatKindSigned = 0,
        cudaChannelFormatKindUnsigned = 1,
        cudaChannelFormatKindFloat = 2,
        cudaChannelFormatKindNone = 3,
    }
    enum cudaChannelFormatKindSigned = cudaChannelFormatKind.cudaChannelFormatKindSigned;
    enum cudaChannelFormatKindUnsigned = cudaChannelFormatKind.cudaChannelFormatKindUnsigned;
    enum cudaChannelFormatKindFloat = cudaChannelFormatKind.cudaChannelFormatKindFloat;
    enum cudaChannelFormatKindNone = cudaChannelFormatKind.cudaChannelFormatKindNone;
    enum cudaError
    {
        cudaSuccess = 0,
        cudaErrorInvalidValue = 1,
        cudaErrorMemoryAllocation = 2,
        cudaErrorInitializationError = 3,
        cudaErrorCudartUnloading = 4,
        cudaErrorProfilerDisabled = 5,
        cudaErrorProfilerNotInitialized = 6,
        cudaErrorProfilerAlreadyStarted = 7,
        cudaErrorProfilerAlreadyStopped = 8,
        cudaErrorInvalidConfiguration = 9,
        cudaErrorInvalidPitchValue = 12,
        cudaErrorInvalidSymbol = 13,
        cudaErrorInvalidHostPointer = 16,
        cudaErrorInvalidDevicePointer = 17,
        cudaErrorInvalidTexture = 18,
        cudaErrorInvalidTextureBinding = 19,
        cudaErrorInvalidChannelDescriptor = 20,
        cudaErrorInvalidMemcpyDirection = 21,
        cudaErrorAddressOfConstant = 22,
        cudaErrorTextureFetchFailed = 23,
        cudaErrorTextureNotBound = 24,
        cudaErrorSynchronizationError = 25,
        cudaErrorInvalidFilterSetting = 26,
        cudaErrorInvalidNormSetting = 27,
        cudaErrorMixedDeviceExecution = 28,
        cudaErrorNotYetImplemented = 31,
        cudaErrorMemoryValueTooLarge = 32,
        cudaErrorInsufficientDriver = 35,
        cudaErrorInvalidSurface = 37,
        cudaErrorDuplicateVariableName = 43,
        cudaErrorDuplicateTextureName = 44,
        cudaErrorDuplicateSurfaceName = 45,
        cudaErrorDevicesUnavailable = 46,
        cudaErrorIncompatibleDriverContext = 49,
        cudaErrorMissingConfiguration = 52,
        cudaErrorPriorLaunchFailure = 53,
        cudaErrorLaunchMaxDepthExceeded = 65,
        cudaErrorLaunchFileScopedTex = 66,
        cudaErrorLaunchFileScopedSurf = 67,
        cudaErrorSyncDepthExceeded = 68,
        cudaErrorLaunchPendingCountExceeded = 69,
        cudaErrorInvalidDeviceFunction = 98,
        cudaErrorNoDevice = 100,
        cudaErrorInvalidDevice = 101,
        cudaErrorStartupFailure = 127,
        cudaErrorInvalidKernelImage = 200,
        cudaErrorDeviceUninitialized = 201,
        cudaErrorMapBufferObjectFailed = 205,
        cudaErrorUnmapBufferObjectFailed = 206,
        cudaErrorArrayIsMapped = 207,
        cudaErrorAlreadyMapped = 208,
        cudaErrorNoKernelImageForDevice = 209,
        cudaErrorAlreadyAcquired = 210,
        cudaErrorNotMapped = 211,
        cudaErrorNotMappedAsArray = 212,
        cudaErrorNotMappedAsPointer = 213,
        cudaErrorECCUncorrectable = 214,
        cudaErrorUnsupportedLimit = 215,
        cudaErrorDeviceAlreadyInUse = 216,
        cudaErrorPeerAccessUnsupported = 217,
        cudaErrorInvalidPtx = 218,
        cudaErrorInvalidGraphicsContext = 219,
        cudaErrorNvlinkUncorrectable = 220,
        cudaErrorJitCompilerNotFound = 221,
        cudaErrorInvalidSource = 300,
        cudaErrorFileNotFound = 301,
        cudaErrorSharedObjectSymbolNotFound = 302,
        cudaErrorSharedObjectInitFailed = 303,
        cudaErrorOperatingSystem = 304,
        cudaErrorInvalidResourceHandle = 400,
        cudaErrorIllegalState = 401,
        cudaErrorSymbolNotFound = 500,
        cudaErrorNotReady = 600,
        cudaErrorIllegalAddress = 700,
        cudaErrorLaunchOutOfResources = 701,
        cudaErrorLaunchTimeout = 702,
        cudaErrorLaunchIncompatibleTexturing = 703,
        cudaErrorPeerAccessAlreadyEnabled = 704,
        cudaErrorPeerAccessNotEnabled = 705,
        cudaErrorSetOnActiveProcess = 708,
        cudaErrorContextIsDestroyed = 709,
        cudaErrorAssert = 710,
        cudaErrorTooManyPeers = 711,
        cudaErrorHostMemoryAlreadyRegistered = 712,
        cudaErrorHostMemoryNotRegistered = 713,
        cudaErrorHardwareStackError = 714,
        cudaErrorIllegalInstruction = 715,
        cudaErrorMisalignedAddress = 716,
        cudaErrorInvalidAddressSpace = 717,
        cudaErrorInvalidPc = 718,
        cudaErrorLaunchFailure = 719,
        cudaErrorCooperativeLaunchTooLarge = 720,
        cudaErrorNotPermitted = 800,
        cudaErrorNotSupported = 801,
        cudaErrorSystemNotReady = 802,
        cudaErrorSystemDriverMismatch = 803,
        cudaErrorCompatNotSupportedOnDevice = 804,
        cudaErrorStreamCaptureUnsupported = 900,
        cudaErrorStreamCaptureInvalidated = 901,
        cudaErrorStreamCaptureMerge = 902,
        cudaErrorStreamCaptureUnmatched = 903,
        cudaErrorStreamCaptureUnjoined = 904,
        cudaErrorStreamCaptureIsolation = 905,
        cudaErrorStreamCaptureImplicit = 906,
        cudaErrorCapturedEvent = 907,
        cudaErrorStreamCaptureWrongThread = 908,
        cudaErrorTimeout = 909,
        cudaErrorGraphExecUpdateFailure = 910,
        cudaErrorUnknown = 999,
        cudaErrorApiFailureBase = 10000,
    }
    enum cudaSuccess = cudaError.cudaSuccess;
    enum cudaErrorInvalidValue = cudaError.cudaErrorInvalidValue;
    enum cudaErrorMemoryAllocation = cudaError.cudaErrorMemoryAllocation;
    enum cudaErrorInitializationError = cudaError.cudaErrorInitializationError;
    enum cudaErrorCudartUnloading = cudaError.cudaErrorCudartUnloading;
    enum cudaErrorProfilerDisabled = cudaError.cudaErrorProfilerDisabled;
    enum cudaErrorProfilerNotInitialized = cudaError.cudaErrorProfilerNotInitialized;
    enum cudaErrorProfilerAlreadyStarted = cudaError.cudaErrorProfilerAlreadyStarted;
    enum cudaErrorProfilerAlreadyStopped = cudaError.cudaErrorProfilerAlreadyStopped;
    enum cudaErrorInvalidConfiguration = cudaError.cudaErrorInvalidConfiguration;
    enum cudaErrorInvalidPitchValue = cudaError.cudaErrorInvalidPitchValue;
    enum cudaErrorInvalidSymbol = cudaError.cudaErrorInvalidSymbol;
    enum cudaErrorInvalidHostPointer = cudaError.cudaErrorInvalidHostPointer;
    enum cudaErrorInvalidDevicePointer = cudaError.cudaErrorInvalidDevicePointer;
    enum cudaErrorInvalidTexture = cudaError.cudaErrorInvalidTexture;
    enum cudaErrorInvalidTextureBinding = cudaError.cudaErrorInvalidTextureBinding;
    enum cudaErrorInvalidChannelDescriptor = cudaError.cudaErrorInvalidChannelDescriptor;
    enum cudaErrorInvalidMemcpyDirection = cudaError.cudaErrorInvalidMemcpyDirection;
    enum cudaErrorAddressOfConstant = cudaError.cudaErrorAddressOfConstant;
    enum cudaErrorTextureFetchFailed = cudaError.cudaErrorTextureFetchFailed;
    enum cudaErrorTextureNotBound = cudaError.cudaErrorTextureNotBound;
    enum cudaErrorSynchronizationError = cudaError.cudaErrorSynchronizationError;
    enum cudaErrorInvalidFilterSetting = cudaError.cudaErrorInvalidFilterSetting;
    enum cudaErrorInvalidNormSetting = cudaError.cudaErrorInvalidNormSetting;
    enum cudaErrorMixedDeviceExecution = cudaError.cudaErrorMixedDeviceExecution;
    enum cudaErrorNotYetImplemented = cudaError.cudaErrorNotYetImplemented;
    enum cudaErrorMemoryValueTooLarge = cudaError.cudaErrorMemoryValueTooLarge;
    enum cudaErrorInsufficientDriver = cudaError.cudaErrorInsufficientDriver;
    enum cudaErrorInvalidSurface = cudaError.cudaErrorInvalidSurface;
    enum cudaErrorDuplicateVariableName = cudaError.cudaErrorDuplicateVariableName;
    enum cudaErrorDuplicateTextureName = cudaError.cudaErrorDuplicateTextureName;
    enum cudaErrorDuplicateSurfaceName = cudaError.cudaErrorDuplicateSurfaceName;
    enum cudaErrorDevicesUnavailable = cudaError.cudaErrorDevicesUnavailable;
    enum cudaErrorIncompatibleDriverContext = cudaError.cudaErrorIncompatibleDriverContext;
    enum cudaErrorMissingConfiguration = cudaError.cudaErrorMissingConfiguration;
    enum cudaErrorPriorLaunchFailure = cudaError.cudaErrorPriorLaunchFailure;
    enum cudaErrorLaunchMaxDepthExceeded = cudaError.cudaErrorLaunchMaxDepthExceeded;
    enum cudaErrorLaunchFileScopedTex = cudaError.cudaErrorLaunchFileScopedTex;
    enum cudaErrorLaunchFileScopedSurf = cudaError.cudaErrorLaunchFileScopedSurf;
    enum cudaErrorSyncDepthExceeded = cudaError.cudaErrorSyncDepthExceeded;
    enum cudaErrorLaunchPendingCountExceeded = cudaError.cudaErrorLaunchPendingCountExceeded;
    enum cudaErrorInvalidDeviceFunction = cudaError.cudaErrorInvalidDeviceFunction;
    enum cudaErrorNoDevice = cudaError.cudaErrorNoDevice;
    enum cudaErrorInvalidDevice = cudaError.cudaErrorInvalidDevice;
    enum cudaErrorStartupFailure = cudaError.cudaErrorStartupFailure;
    enum cudaErrorInvalidKernelImage = cudaError.cudaErrorInvalidKernelImage;
    enum cudaErrorDeviceUninitialized = cudaError.cudaErrorDeviceUninitialized;
    enum cudaErrorMapBufferObjectFailed = cudaError.cudaErrorMapBufferObjectFailed;
    enum cudaErrorUnmapBufferObjectFailed = cudaError.cudaErrorUnmapBufferObjectFailed;
    enum cudaErrorArrayIsMapped = cudaError.cudaErrorArrayIsMapped;
    enum cudaErrorAlreadyMapped = cudaError.cudaErrorAlreadyMapped;
    enum cudaErrorNoKernelImageForDevice = cudaError.cudaErrorNoKernelImageForDevice;
    enum cudaErrorAlreadyAcquired = cudaError.cudaErrorAlreadyAcquired;
    enum cudaErrorNotMapped = cudaError.cudaErrorNotMapped;
    enum cudaErrorNotMappedAsArray = cudaError.cudaErrorNotMappedAsArray;
    enum cudaErrorNotMappedAsPointer = cudaError.cudaErrorNotMappedAsPointer;
    enum cudaErrorECCUncorrectable = cudaError.cudaErrorECCUncorrectable;
    enum cudaErrorUnsupportedLimit = cudaError.cudaErrorUnsupportedLimit;
    enum cudaErrorDeviceAlreadyInUse = cudaError.cudaErrorDeviceAlreadyInUse;
    enum cudaErrorPeerAccessUnsupported = cudaError.cudaErrorPeerAccessUnsupported;
    enum cudaErrorInvalidPtx = cudaError.cudaErrorInvalidPtx;
    enum cudaErrorInvalidGraphicsContext = cudaError.cudaErrorInvalidGraphicsContext;
    enum cudaErrorNvlinkUncorrectable = cudaError.cudaErrorNvlinkUncorrectable;
    enum cudaErrorJitCompilerNotFound = cudaError.cudaErrorJitCompilerNotFound;
    enum cudaErrorInvalidSource = cudaError.cudaErrorInvalidSource;
    enum cudaErrorFileNotFound = cudaError.cudaErrorFileNotFound;
    enum cudaErrorSharedObjectSymbolNotFound = cudaError.cudaErrorSharedObjectSymbolNotFound;
    enum cudaErrorSharedObjectInitFailed = cudaError.cudaErrorSharedObjectInitFailed;
    enum cudaErrorOperatingSystem = cudaError.cudaErrorOperatingSystem;
    enum cudaErrorInvalidResourceHandle = cudaError.cudaErrorInvalidResourceHandle;
    enum cudaErrorIllegalState = cudaError.cudaErrorIllegalState;
    enum cudaErrorSymbolNotFound = cudaError.cudaErrorSymbolNotFound;
    enum cudaErrorNotReady = cudaError.cudaErrorNotReady;
    enum cudaErrorIllegalAddress = cudaError.cudaErrorIllegalAddress;
    enum cudaErrorLaunchOutOfResources = cudaError.cudaErrorLaunchOutOfResources;
    enum cudaErrorLaunchTimeout = cudaError.cudaErrorLaunchTimeout;
    enum cudaErrorLaunchIncompatibleTexturing = cudaError.cudaErrorLaunchIncompatibleTexturing;
    enum cudaErrorPeerAccessAlreadyEnabled = cudaError.cudaErrorPeerAccessAlreadyEnabled;
    enum cudaErrorPeerAccessNotEnabled = cudaError.cudaErrorPeerAccessNotEnabled;
    enum cudaErrorSetOnActiveProcess = cudaError.cudaErrorSetOnActiveProcess;
    enum cudaErrorContextIsDestroyed = cudaError.cudaErrorContextIsDestroyed;
    enum cudaErrorAssert = cudaError.cudaErrorAssert;
    enum cudaErrorTooManyPeers = cudaError.cudaErrorTooManyPeers;
    enum cudaErrorHostMemoryAlreadyRegistered = cudaError.cudaErrorHostMemoryAlreadyRegistered;
    enum cudaErrorHostMemoryNotRegistered = cudaError.cudaErrorHostMemoryNotRegistered;
    enum cudaErrorHardwareStackError = cudaError.cudaErrorHardwareStackError;
    enum cudaErrorIllegalInstruction = cudaError.cudaErrorIllegalInstruction;
    enum cudaErrorMisalignedAddress = cudaError.cudaErrorMisalignedAddress;
    enum cudaErrorInvalidAddressSpace = cudaError.cudaErrorInvalidAddressSpace;
    enum cudaErrorInvalidPc = cudaError.cudaErrorInvalidPc;
    enum cudaErrorLaunchFailure = cudaError.cudaErrorLaunchFailure;
    enum cudaErrorCooperativeLaunchTooLarge = cudaError.cudaErrorCooperativeLaunchTooLarge;
    enum cudaErrorNotPermitted = cudaError.cudaErrorNotPermitted;
    enum cudaErrorNotSupported = cudaError.cudaErrorNotSupported;
    enum cudaErrorSystemNotReady = cudaError.cudaErrorSystemNotReady;
    enum cudaErrorSystemDriverMismatch = cudaError.cudaErrorSystemDriverMismatch;
    enum cudaErrorCompatNotSupportedOnDevice = cudaError.cudaErrorCompatNotSupportedOnDevice;
    enum cudaErrorStreamCaptureUnsupported = cudaError.cudaErrorStreamCaptureUnsupported;
    enum cudaErrorStreamCaptureInvalidated = cudaError.cudaErrorStreamCaptureInvalidated;
    enum cudaErrorStreamCaptureMerge = cudaError.cudaErrorStreamCaptureMerge;
    enum cudaErrorStreamCaptureUnmatched = cudaError.cudaErrorStreamCaptureUnmatched;
    enum cudaErrorStreamCaptureUnjoined = cudaError.cudaErrorStreamCaptureUnjoined;
    enum cudaErrorStreamCaptureIsolation = cudaError.cudaErrorStreamCaptureIsolation;
    enum cudaErrorStreamCaptureImplicit = cudaError.cudaErrorStreamCaptureImplicit;
    enum cudaErrorCapturedEvent = cudaError.cudaErrorCapturedEvent;
    enum cudaErrorStreamCaptureWrongThread = cudaError.cudaErrorStreamCaptureWrongThread;
    enum cudaErrorTimeout = cudaError.cudaErrorTimeout;
    enum cudaErrorGraphExecUpdateFailure = cudaError.cudaErrorGraphExecUpdateFailure;
    enum cudaErrorUnknown = cudaError.cudaErrorUnknown;
    enum cudaErrorApiFailureBase = cudaError.cudaErrorApiFailureBase;
    enum cudaRoundMode
    {
        cudaRoundNearest = 0,
        cudaRoundZero = 1,
        cudaRoundPosInf = 2,
        cudaRoundMinInf = 3,
    }
    enum cudaRoundNearest = cudaRoundMode.cudaRoundNearest;
    enum cudaRoundZero = cudaRoundMode.cudaRoundZero;
    enum cudaRoundPosInf = cudaRoundMode.cudaRoundPosInf;
    enum cudaRoundMinInf = cudaRoundMode.cudaRoundMinInf;
    cudaError cudaGetExportTable(const(void)**, const(CUuuid_st)*) @nogc nothrow;
    cudaError cudaGraphDestroy(CUgraph_st*) @nogc nothrow;
    cudaError cudaGraphExecDestroy(CUgraphExec_st*) @nogc nothrow;
    cudaError cudaGraphLaunch(CUgraphExec_st*, CUstream_st*) @nogc nothrow;
    cudaError cudaGraphExecUpdate(CUgraphExec_st*, CUgraph_st*, CUgraphNode_st**, cudaGraphExecUpdateResult*) @nogc nothrow;
    cudaError cudaGraphExecHostNodeSetParams(CUgraphExec_st*, CUgraphNode_st*, const(cudaHostNodeParams)*) @nogc nothrow;
    cudaError cudaGraphExecMemsetNodeSetParams(CUgraphExec_st*, CUgraphNode_st*, const(cudaMemsetParams)*) @nogc nothrow;
    cudaError cudaGraphExecMemcpyNodeSetParams(CUgraphExec_st*, CUgraphNode_st*, const(cudaMemcpy3DParms)*) @nogc nothrow;
    cudaError cudaGraphExecKernelNodeSetParams(CUgraphExec_st*, CUgraphNode_st*, const(cudaKernelNodeParams)*) @nogc nothrow;
    cudaError cudaGraphInstantiate(CUgraphExec_st**, CUgraph_st*, CUgraphNode_st**, char*, c_ulong) @nogc nothrow;
    cudaError cudaGraphDestroyNode(CUgraphNode_st*) @nogc nothrow;
    cudaError cudaGraphRemoveDependencies(CUgraph_st*, const(CUgraphNode_st*)*, const(CUgraphNode_st*)*, c_ulong) @nogc nothrow;
    cudaError cudaGraphAddDependencies(CUgraph_st*, const(CUgraphNode_st*)*, const(CUgraphNode_st*)*, c_ulong) @nogc nothrow;
    cudaError cudaGraphNodeGetDependentNodes(CUgraphNode_st*, CUgraphNode_st**, c_ulong*) @nogc nothrow;
    cudaError cudaGraphNodeGetDependencies(CUgraphNode_st*, CUgraphNode_st**, c_ulong*) @nogc nothrow;
    cudaError cudaGraphGetEdges(CUgraph_st*, CUgraphNode_st**, CUgraphNode_st**, c_ulong*) @nogc nothrow;
    cudaError cudaGraphGetRootNodes(CUgraph_st*, CUgraphNode_st**, c_ulong*) @nogc nothrow;
    cudaError cudaGraphGetNodes(CUgraph_st*, CUgraphNode_st**, c_ulong*) @nogc nothrow;
    cudaError cudaGraphNodeGetType(CUgraphNode_st*, cudaGraphNodeType*) @nogc nothrow;
    cudaError cudaGraphNodeFindInClone(CUgraphNode_st**, CUgraphNode_st*, CUgraph_st*) @nogc nothrow;
    cudaError cudaGraphClone(CUgraph_st**, CUgraph_st*) @nogc nothrow;
    cudaError cudaGraphAddEmptyNode(CUgraphNode_st**, CUgraph_st*, const(CUgraphNode_st*)*, c_ulong) @nogc nothrow;
    cudaError cudaGraphChildGraphNodeGetGraph(CUgraphNode_st*, CUgraph_st**) @nogc nothrow;
    cudaError cudaGraphAddChildGraphNode(CUgraphNode_st**, CUgraph_st*, const(CUgraphNode_st*)*, c_ulong, CUgraph_st*) @nogc nothrow;
    cudaError cudaGraphHostNodeSetParams(CUgraphNode_st*, const(cudaHostNodeParams)*) @nogc nothrow;
    cudaError cudaGraphHostNodeGetParams(CUgraphNode_st*, cudaHostNodeParams*) @nogc nothrow;
    cudaError cudaGraphAddHostNode(CUgraphNode_st**, CUgraph_st*, const(CUgraphNode_st*)*, c_ulong, const(cudaHostNodeParams)*) @nogc nothrow;
    cudaError cudaGraphMemsetNodeSetParams(CUgraphNode_st*, const(cudaMemsetParams)*) @nogc nothrow;
    cudaError cudaGraphMemsetNodeGetParams(CUgraphNode_st*, cudaMemsetParams*) @nogc nothrow;
    cudaError cudaGraphAddMemsetNode(CUgraphNode_st**, CUgraph_st*, const(CUgraphNode_st*)*, c_ulong, const(cudaMemsetParams)*) @nogc nothrow;
    cudaError cudaGraphMemcpyNodeSetParams(CUgraphNode_st*, const(cudaMemcpy3DParms)*) @nogc nothrow;
    cudaError cudaGraphMemcpyNodeGetParams(CUgraphNode_st*, cudaMemcpy3DParms*) @nogc nothrow;
    cudaError cudaGraphAddMemcpyNode(CUgraphNode_st**, CUgraph_st*, const(CUgraphNode_st*)*, c_ulong, const(cudaMemcpy3DParms)*) @nogc nothrow;
    cudaError cudaGraphKernelNodeSetParams(CUgraphNode_st*, const(cudaKernelNodeParams)*) @nogc nothrow;
    cudaError cudaGraphKernelNodeGetParams(CUgraphNode_st*, cudaKernelNodeParams*) @nogc nothrow;
    cudaError cudaGraphAddKernelNode(CUgraphNode_st**, CUgraph_st*, const(CUgraphNode_st*)*, c_ulong, const(cudaKernelNodeParams)*) @nogc nothrow;
    cudaError cudaGraphCreate(CUgraph_st**, uint) @nogc nothrow;
    cudaError cudaRuntimeGetVersion(int*) @nogc nothrow;
    cudaError cudaDriverGetVersion(int*) @nogc nothrow;
    cudaError cudaGetSurfaceObjectResourceDesc(cudaResourceDesc*, ulong) @nogc nothrow;
    cudaError cudaDestroySurfaceObject(ulong) @nogc nothrow;
    cudaError cudaCreateSurfaceObject(ulong*, const(cudaResourceDesc)*) @nogc nothrow;
    cudaError cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc*, ulong) @nogc nothrow;
    cudaError cudaGetTextureObjectTextureDesc(cudaTextureDesc*, ulong) @nogc nothrow;
    cudaError cudaGetTextureObjectResourceDesc(cudaResourceDesc*, ulong) @nogc nothrow;
    cudaError cudaDestroyTextureObject(ulong) @nogc nothrow;
    cudaError cudaCreateTextureObject(ulong*, const(cudaResourceDesc)*, const(cudaTextureDesc)*, const(cudaResourceViewDesc)*) @nogc nothrow;
    cudaChannelFormatDesc cudaCreateChannelDesc(int, int, int, int, cudaChannelFormatKind) @nogc nothrow;
    cudaError cudaGetChannelDesc(cudaChannelFormatDesc*, const(cudaArray)*) @nogc nothrow;
    cudaError cudaGetSurfaceReference(const(surfaceReference)**, const(void)*) @nogc nothrow;
    cudaError cudaBindSurfaceToArray(const(surfaceReference)*, const(cudaArray)*, const(cudaChannelFormatDesc)*) @nogc nothrow;
    cudaError cudaGetTextureReference(const(textureReference)**, const(void)*) @nogc nothrow;
    cudaError cudaGetTextureAlignmentOffset(c_ulong*, const(textureReference)*) @nogc nothrow;
    cudaError cudaUnbindTexture(const(textureReference)*) @nogc nothrow;
    cudaError cudaBindTextureToMipmappedArray(const(textureReference)*, const(cudaMipmappedArray)*, const(cudaChannelFormatDesc)*) @nogc nothrow;
    cudaError cudaBindTextureToArray(const(textureReference)*, const(cudaArray)*, const(cudaChannelFormatDesc)*) @nogc nothrow;
    cudaError cudaBindTexture2D(c_ulong*, const(textureReference)*, const(void)*, const(cudaChannelFormatDesc)*, c_ulong, c_ulong, c_ulong) @nogc nothrow;
    cudaError cudaBindTexture(c_ulong*, const(textureReference)*, const(void)*, const(cudaChannelFormatDesc)*, c_ulong) @nogc nothrow;
    cudaError cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray**, cudaGraphicsResource*) @nogc nothrow;
    cudaError cudaGraphicsSubResourceGetMappedArray(cudaArray**, cudaGraphicsResource*, uint, uint) @nogc nothrow;
    cudaError cudaGraphicsResourceGetMappedPointer(void**, c_ulong*, cudaGraphicsResource*) @nogc nothrow;
    cudaError cudaGraphicsUnmapResources(int, cudaGraphicsResource**, CUstream_st*) @nogc nothrow;
    cudaError cudaGraphicsMapResources(int, cudaGraphicsResource**, CUstream_st*) @nogc nothrow;
    cudaError cudaGraphicsResourceSetMapFlags(cudaGraphicsResource*, uint) @nogc nothrow;
    cudaError cudaGraphicsUnregisterResource(cudaGraphicsResource*) @nogc nothrow;
    cudaError cudaDeviceDisablePeerAccess(int) @nogc nothrow;
    cudaError cudaDeviceEnablePeerAccess(int, uint) @nogc nothrow;
    cudaError cudaDeviceCanAccessPeer(int*, int, int) @nogc nothrow;
    cudaError cudaPointerGetAttributes(cudaPointerAttributes*, const(void)*) @nogc nothrow;
    cudaError cudaMemcpyFromArrayAsync(void*, const(cudaArray)*, c_ulong, c_ulong, c_ulong, cudaMemcpyKind, CUstream_st*) @nogc nothrow;
    cudaError cudaDeviceReset() @nogc nothrow;
    cudaError cudaDeviceSynchronize() @nogc nothrow;
    cudaError cudaDeviceSetLimit(cudaLimit, c_ulong) @nogc nothrow;
    cudaError cudaDeviceGetLimit(c_ulong*, cudaLimit) @nogc nothrow;
    cudaError cudaDeviceGetCacheConfig(cudaFuncCache*) @nogc nothrow;
    cudaError cudaDeviceGetStreamPriorityRange(int*, int*) @nogc nothrow;
    cudaError cudaDeviceSetCacheConfig(cudaFuncCache) @nogc nothrow;
    cudaError cudaDeviceGetSharedMemConfig(cudaSharedMemConfig*) @nogc nothrow;
    cudaError cudaDeviceSetSharedMemConfig(cudaSharedMemConfig) @nogc nothrow;
    cudaError cudaDeviceGetByPCIBusId(int*, const(char)*) @nogc nothrow;
    cudaError cudaDeviceGetPCIBusId(char*, int, int) @nogc nothrow;
    cudaError cudaIpcGetEventHandle(cudaIpcEventHandle_st*, CUevent_st*) @nogc nothrow;
    cudaError cudaIpcOpenEventHandle(CUevent_st**, cudaIpcEventHandle_st) @nogc nothrow;
    cudaError cudaIpcGetMemHandle(cudaIpcMemHandle_st*, void*) @nogc nothrow;
    cudaError cudaIpcOpenMemHandle(void**, cudaIpcMemHandle_st, uint) @nogc nothrow;
    cudaError cudaIpcCloseMemHandle(void*) @nogc nothrow;
    cudaError cudaThreadExit() @nogc nothrow;
    cudaError cudaThreadSynchronize() @nogc nothrow;
    cudaError cudaThreadSetLimit(cudaLimit, c_ulong) @nogc nothrow;
    cudaError cudaThreadGetLimit(c_ulong*, cudaLimit) @nogc nothrow;
    cudaError cudaThreadGetCacheConfig(cudaFuncCache*) @nogc nothrow;
    cudaError cudaThreadSetCacheConfig(cudaFuncCache) @nogc nothrow;
    cudaError cudaGetLastError() @nogc nothrow;
    cudaError cudaPeekAtLastError() @nogc nothrow;
    const(char)* cudaGetErrorName(cudaError) @nogc nothrow;
    const(char)* cudaGetErrorString(cudaError) @nogc nothrow;
    cudaError cudaGetDeviceCount(int*) @nogc nothrow;
    cudaError cudaGetDeviceProperties(cudaDeviceProp*, int) @nogc nothrow;
    cudaError cudaDeviceGetAttribute(int*, cudaDeviceAttr, int) @nogc nothrow;
    cudaError cudaDeviceGetNvSciSyncAttributes(void*, int, int) @nogc nothrow;
    cudaError cudaDeviceGetP2PAttribute(int*, cudaDeviceP2PAttr, int, int) @nogc nothrow;
    cudaError cudaChooseDevice(int*, const(cudaDeviceProp)*) @nogc nothrow;
    cudaError cudaSetDevice(int) @nogc nothrow;
    cudaError cudaGetDevice(int*) @nogc nothrow;
    cudaError cudaSetValidDevices(int*, int) @nogc nothrow;
    cudaError cudaSetDeviceFlags(uint) @nogc nothrow;
    cudaError cudaGetDeviceFlags(uint*) @nogc nothrow;
    cudaError cudaStreamCreate(CUstream_st**) @nogc nothrow;
    cudaError cudaStreamCreateWithFlags(CUstream_st**, uint) @nogc nothrow;
    cudaError cudaStreamCreateWithPriority(CUstream_st**, uint, int) @nogc nothrow;
    cudaError cudaStreamGetPriority(CUstream_st*, int*) @nogc nothrow;
    cudaError cudaStreamGetFlags(CUstream_st*, uint*) @nogc nothrow;
    cudaError cudaStreamDestroy(CUstream_st*) @nogc nothrow;
    cudaError cudaStreamWaitEvent(CUstream_st*, CUevent_st*, uint) @nogc nothrow;
    alias cudaStreamCallback_t = void function(CUstream_st*, cudaError, void*);
    cudaError cudaStreamAddCallback(CUstream_st*, void function(CUstream_st*, cudaError, void*), void*, uint) @nogc nothrow;
    cudaError cudaStreamSynchronize(CUstream_st*) @nogc nothrow;
    cudaError cudaStreamQuery(CUstream_st*) @nogc nothrow;
    cudaError cudaStreamAttachMemAsync(CUstream_st*, void*, c_ulong, uint) @nogc nothrow;
    cudaError cudaStreamBeginCapture(CUstream_st*, cudaStreamCaptureMode) @nogc nothrow;
    cudaError cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode*) @nogc nothrow;
    cudaError cudaStreamEndCapture(CUstream_st*, CUgraph_st**) @nogc nothrow;
    cudaError cudaStreamIsCapturing(CUstream_st*, cudaStreamCaptureStatus*) @nogc nothrow;
    cudaError cudaStreamGetCaptureInfo(CUstream_st*, cudaStreamCaptureStatus*, ulong*) @nogc nothrow;
    cudaError cudaEventCreate(CUevent_st**) @nogc nothrow;
    cudaError cudaEventCreateWithFlags(CUevent_st**, uint) @nogc nothrow;
    cudaError cudaEventRecord(CUevent_st*, CUstream_st*) @nogc nothrow;
    cudaError cudaEventQuery(CUevent_st*) @nogc nothrow;
    cudaError cudaEventSynchronize(CUevent_st*) @nogc nothrow;
    cudaError cudaEventDestroy(CUevent_st*) @nogc nothrow;
    cudaError cudaEventElapsedTime(float*, CUevent_st*, CUevent_st*) @nogc nothrow;
    cudaError cudaImportExternalMemory(CUexternalMemory_st**, const(cudaExternalMemoryHandleDesc)*) @nogc nothrow;
    cudaError cudaExternalMemoryGetMappedBuffer(void**, CUexternalMemory_st*, const(cudaExternalMemoryBufferDesc)*) @nogc nothrow;
    cudaError cudaExternalMemoryGetMappedMipmappedArray(cudaMipmappedArray**, CUexternalMemory_st*, const(cudaExternalMemoryMipmappedArrayDesc)*) @nogc nothrow;
    cudaError cudaDestroyExternalMemory(CUexternalMemory_st*) @nogc nothrow;
    cudaError cudaImportExternalSemaphore(CUexternalSemaphore_st**, const(cudaExternalSemaphoreHandleDesc)*) @nogc nothrow;
    cudaError cudaSignalExternalSemaphoresAsync(const(CUexternalSemaphore_st*)*, const(cudaExternalSemaphoreSignalParams)*, uint, CUstream_st*) @nogc nothrow;
    cudaError cudaWaitExternalSemaphoresAsync(const(CUexternalSemaphore_st*)*, const(cudaExternalSemaphoreWaitParams)*, uint, CUstream_st*) @nogc nothrow;
    cudaError cudaDestroyExternalSemaphore(CUexternalSemaphore_st*) @nogc nothrow;
    cudaError cudaLaunchKernel(const(void)*, dim3, dim3, void**, c_ulong, CUstream_st*) @nogc nothrow;
    cudaError cudaLaunchCooperativeKernel(const(void)*, dim3, dim3, void**, c_ulong, CUstream_st*) @nogc nothrow;
    cudaError cudaLaunchCooperativeKernelMultiDevice(cudaLaunchParams*, uint, uint) @nogc nothrow;
    cudaError cudaFuncSetCacheConfig(const(void)*, cudaFuncCache) @nogc nothrow;
    cudaError cudaFuncSetSharedMemConfig(const(void)*, cudaSharedMemConfig) @nogc nothrow;
    cudaError cudaFuncGetAttributes(cudaFuncAttributes*, const(void)*) @nogc nothrow;
    cudaError cudaFuncSetAttribute(const(void)*, cudaFuncAttribute, int) @nogc nothrow;
    cudaError cudaSetDoubleForDevice(double*) @nogc nothrow;
    cudaError cudaSetDoubleForHost(double*) @nogc nothrow;
    cudaError cudaLaunchHostFunc(CUstream_st*, void function(void*), void*) @nogc nothrow;
    cudaError cudaOccupancyMaxActiveBlocksPerMultiprocessor(int*, const(void)*, int, c_ulong) @nogc nothrow;
    cudaError cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int*, const(void)*, int, c_ulong, uint) @nogc nothrow;
    cudaError cudaMallocManaged(void**, c_ulong, uint) @nogc nothrow;
    cudaError cudaMalloc(void**, c_ulong) @nogc nothrow;
    cudaError cudaMallocHost(void**, c_ulong) @nogc nothrow;
    cudaError cudaMallocPitch(void**, c_ulong*, c_ulong, c_ulong) @nogc nothrow;
    cudaError cudaMallocArray(cudaArray**, const(cudaChannelFormatDesc)*, c_ulong, c_ulong, uint) @nogc nothrow;
    cudaError cudaFree(void*) @nogc nothrow;
    cudaError cudaFreeHost(void*) @nogc nothrow;
    cudaError cudaFreeArray(cudaArray*) @nogc nothrow;
    cudaError cudaFreeMipmappedArray(cudaMipmappedArray*) @nogc nothrow;
    cudaError cudaHostAlloc(void**, c_ulong, uint) @nogc nothrow;
    cudaError cudaHostRegister(void*, c_ulong, uint) @nogc nothrow;
    cudaError cudaHostUnregister(void*) @nogc nothrow;
    cudaError cudaHostGetDevicePointer(void**, void*, uint) @nogc nothrow;
    cudaError cudaHostGetFlags(uint*, void*) @nogc nothrow;
    cudaError cudaMalloc3D(cudaPitchedPtr*, cudaExtent) @nogc nothrow;
    cudaError cudaMalloc3DArray(cudaArray**, const(cudaChannelFormatDesc)*, cudaExtent, uint) @nogc nothrow;
    cudaError cudaMallocMipmappedArray(cudaMipmappedArray**, const(cudaChannelFormatDesc)*, cudaExtent, uint, uint) @nogc nothrow;
    cudaError cudaGetMipmappedArrayLevel(cudaArray**, const(cudaMipmappedArray)*, uint) @nogc nothrow;
    cudaError cudaMemcpy3D(const(cudaMemcpy3DParms)*) @nogc nothrow;
    cudaError cudaMemcpy3DPeer(const(cudaMemcpy3DPeerParms)*) @nogc nothrow;
    cudaError cudaMemcpy3DAsync(const(cudaMemcpy3DParms)*, CUstream_st*) @nogc nothrow;
    cudaError cudaMemcpy3DPeerAsync(const(cudaMemcpy3DPeerParms)*, CUstream_st*) @nogc nothrow;
    cudaError cudaMemGetInfo(c_ulong*, c_ulong*) @nogc nothrow;
    cudaError cudaArrayGetInfo(cudaChannelFormatDesc*, cudaExtent*, uint*, cudaArray*) @nogc nothrow;
    cudaError cudaMemcpy(void*, const(void)*, c_ulong, cudaMemcpyKind) @nogc nothrow;
    cudaError cudaMemcpyPeer(void*, int, const(void)*, int, c_ulong) @nogc nothrow;
    cudaError cudaMemcpy2D(void*, c_ulong, const(void)*, c_ulong, c_ulong, c_ulong, cudaMemcpyKind) @nogc nothrow;
    cudaError cudaMemcpy2DToArray(cudaArray*, c_ulong, c_ulong, const(void)*, c_ulong, c_ulong, c_ulong, cudaMemcpyKind) @nogc nothrow;
    cudaError cudaMemcpy2DFromArray(void*, c_ulong, const(cudaArray)*, c_ulong, c_ulong, c_ulong, c_ulong, cudaMemcpyKind) @nogc nothrow;
    cudaError cudaMemcpy2DArrayToArray(cudaArray*, c_ulong, c_ulong, const(cudaArray)*, c_ulong, c_ulong, c_ulong, c_ulong, cudaMemcpyKind) @nogc nothrow;
    cudaError cudaMemcpyToSymbol(const(void)*, const(void)*, c_ulong, c_ulong, cudaMemcpyKind) @nogc nothrow;
    cudaError cudaMemcpyFromSymbol(void*, const(void)*, c_ulong, c_ulong, cudaMemcpyKind) @nogc nothrow;
    cudaError cudaMemcpyAsync(void*, const(void)*, c_ulong, cudaMemcpyKind, CUstream_st*) @nogc nothrow;
    cudaError cudaMemcpyPeerAsync(void*, int, const(void)*, int, c_ulong, CUstream_st*) @nogc nothrow;
    cudaError cudaMemcpy2DAsync(void*, c_ulong, const(void)*, c_ulong, c_ulong, c_ulong, cudaMemcpyKind, CUstream_st*) @nogc nothrow;
    cudaError cudaMemcpy2DToArrayAsync(cudaArray*, c_ulong, c_ulong, const(void)*, c_ulong, c_ulong, c_ulong, cudaMemcpyKind, CUstream_st*) @nogc nothrow;
    cudaError cudaMemcpy2DFromArrayAsync(void*, c_ulong, const(cudaArray)*, c_ulong, c_ulong, c_ulong, c_ulong, cudaMemcpyKind, CUstream_st*) @nogc nothrow;
    cudaError cudaMemcpyToSymbolAsync(const(void)*, const(void)*, c_ulong, c_ulong, cudaMemcpyKind, CUstream_st*) @nogc nothrow;
    cudaError cudaMemcpyFromSymbolAsync(void*, const(void)*, c_ulong, c_ulong, cudaMemcpyKind, CUstream_st*) @nogc nothrow;
    cudaError cudaMemset(void*, int, c_ulong) @nogc nothrow;
    cudaError cudaMemset2D(void*, c_ulong, int, c_ulong, c_ulong) @nogc nothrow;
    cudaError cudaMemset3D(cudaPitchedPtr, int, cudaExtent) @nogc nothrow;
    cudaError cudaMemsetAsync(void*, int, c_ulong, CUstream_st*) @nogc nothrow;
    cudaError cudaMemset2DAsync(void*, c_ulong, int, c_ulong, c_ulong, CUstream_st*) @nogc nothrow;
    cudaError cudaMemset3DAsync(cudaPitchedPtr, int, cudaExtent, CUstream_st*) @nogc nothrow;
    cudaError cudaGetSymbolAddress(void**, const(void)*) @nogc nothrow;
    cudaError cudaGetSymbolSize(c_ulong*, const(void)*) @nogc nothrow;
    cudaError cudaMemPrefetchAsync(const(void)*, c_ulong, int, CUstream_st*) @nogc nothrow;
    cudaError cudaMemAdvise(const(void)*, c_ulong, cudaMemoryAdvise, int) @nogc nothrow;
    cudaError cudaMemRangeGetAttribute(void*, c_ulong, cudaMemRangeAttribute, const(void)*, c_ulong) @nogc nothrow;
    cudaError cudaMemRangeGetAttributes(void**, c_ulong*, cudaMemRangeAttribute*, c_ulong, const(void)*, c_ulong) @nogc nothrow;
    cudaError cudaMemcpyToArray(cudaArray*, c_ulong, c_ulong, const(void)*, c_ulong, cudaMemcpyKind) @nogc nothrow;
    cudaError cudaMemcpyFromArray(void*, const(cudaArray)*, c_ulong, c_ulong, c_ulong, cudaMemcpyKind) @nogc nothrow;
    cudaError cudaMemcpyArrayToArray(cudaArray*, c_ulong, c_ulong, const(cudaArray)*, c_ulong, c_ulong, c_ulong, cudaMemcpyKind) @nogc nothrow;
    cudaError cudaMemcpyToArrayAsync(cudaArray*, c_ulong, c_ulong, const(void)*, c_ulong, cudaMemcpyKind, CUstream_st*) @nogc nothrow;



    static if(!is(typeof(__CUDA_DEPRECATED))) {
        private enum enumMixinStr___CUDA_DEPRECATED = `enum __CUDA_DEPRECATED = __attribute__ ( ( deprecated ) );`;
        static if(is(typeof({ mixin(enumMixinStr___CUDA_DEPRECATED); }))) {
            mixin(enumMixinStr___CUDA_DEPRECATED);
        }
    }






    static if(!is(typeof(CUDART_DEVICE))) {
        private enum enumMixinStr_CUDART_DEVICE = `enum CUDART_DEVICE = __device__;`;
        static if(is(typeof({ mixin(enumMixinStr_CUDART_DEVICE); }))) {
            mixin(enumMixinStr_CUDART_DEVICE);
        }
    }




    static if(!is(typeof(__dv))) {
        private enum enumMixinStr___dv = `enum __dv = ( v );`;
        static if(is(typeof({ mixin(enumMixinStr___dv); }))) {
            mixin(enumMixinStr___dv);
        }
    }
    static if(!is(typeof(CUDART_VERSION))) {
        private enum enumMixinStr_CUDART_VERSION = `enum CUDART_VERSION = 10020;`;
        static if(is(typeof({ mixin(enumMixinStr_CUDART_VERSION); }))) {
            mixin(enumMixinStr_CUDART_VERSION);
        }
    }
    static if(!is(typeof(__managed__))) {
        private enum enumMixinStr___managed__ = `enum __managed__ = __location__ ( managed );`;
        static if(is(typeof({ mixin(enumMixinStr___managed__); }))) {
            mixin(enumMixinStr___managed__);
        }
    }




    static if(!is(typeof(__constant__))) {
        private enum enumMixinStr___constant__ = `enum __constant__ = __location__ ( constant );`;
        static if(is(typeof({ mixin(enumMixinStr___constant__); }))) {
            mixin(enumMixinStr___constant__);
        }
    }




    static if(!is(typeof(__shared__))) {
        private enum enumMixinStr___shared__ = `enum __shared__ = __location__ ( shared );`;
        static if(is(typeof({ mixin(enumMixinStr___shared__); }))) {
            mixin(enumMixinStr___shared__);
        }
    }




    static if(!is(typeof(__global__))) {
        private enum enumMixinStr___global__ = `enum __global__ = __location__ ( global );`;
        static if(is(typeof({ mixin(enumMixinStr___global__); }))) {
            mixin(enumMixinStr___global__);
        }
    }




    static if(!is(typeof(__device__))) {
        private enum enumMixinStr___device__ = `enum __device__ = __location__ ( device );`;
        static if(is(typeof({ mixin(enumMixinStr___device__); }))) {
            mixin(enumMixinStr___device__);
        }
    }




    static if(!is(typeof(__host__))) {
        private enum enumMixinStr___host__ = `enum __host__ = __location__ ( host );`;
        static if(is(typeof({ mixin(enumMixinStr___host__); }))) {
            mixin(enumMixinStr___host__);
        }
    }
    static if(!is(typeof(__thread__))) {
        private enum enumMixinStr___thread__ = `enum __thread__ = __thread;`;
        static if(is(typeof({ mixin(enumMixinStr___thread__); }))) {
            mixin(enumMixinStr___thread__);
        }
    }






    static if(!is(typeof(__forceinline__))) {
        private enum enumMixinStr___forceinline__ = `enum __forceinline__ = __inline__ __attribute__ ( ( always_inline ) );`;
        static if(is(typeof({ mixin(enumMixinStr___forceinline__); }))) {
            mixin(enumMixinStr___forceinline__);
        }
    }




    static if(!is(typeof(__no_return__))) {
        private enum enumMixinStr___no_return__ = `enum __no_return__ = __attribute__ ( ( noreturn ) );`;
        static if(is(typeof({ mixin(enumMixinStr___no_return__); }))) {
            mixin(enumMixinStr___no_return__);
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




    static if(!is(typeof(RE_DUP_MAX))) {
        private enum enumMixinStr_RE_DUP_MAX = `enum RE_DUP_MAX = ( 0x7fff );`;
        static if(is(typeof({ mixin(enumMixinStr_RE_DUP_MAX); }))) {
            mixin(enumMixinStr_RE_DUP_MAX);
        }
    }




    static if(!is(typeof(CHARCLASS_NAME_MAX))) {
        private enum enumMixinStr_CHARCLASS_NAME_MAX = `enum CHARCLASS_NAME_MAX = 2048;`;
        static if(is(typeof({ mixin(enumMixinStr_CHARCLASS_NAME_MAX); }))) {
            mixin(enumMixinStr_CHARCLASS_NAME_MAX);
        }
    }




    static if(!is(typeof(LINE_MAX))) {
        private enum enumMixinStr_LINE_MAX = `enum LINE_MAX = _POSIX2_LINE_MAX;`;
        static if(is(typeof({ mixin(enumMixinStr_LINE_MAX); }))) {
            mixin(enumMixinStr_LINE_MAX);
        }
    }




    static if(!is(typeof(EXPR_NEST_MAX))) {
        private enum enumMixinStr_EXPR_NEST_MAX = `enum EXPR_NEST_MAX = _POSIX2_EXPR_NEST_MAX;`;
        static if(is(typeof({ mixin(enumMixinStr_EXPR_NEST_MAX); }))) {
            mixin(enumMixinStr_EXPR_NEST_MAX);
        }
    }




    static if(!is(typeof(COLL_WEIGHTS_MAX))) {
        private enum enumMixinStr_COLL_WEIGHTS_MAX = `enum COLL_WEIGHTS_MAX = 255;`;
        static if(is(typeof({ mixin(enumMixinStr_COLL_WEIGHTS_MAX); }))) {
            mixin(enumMixinStr_COLL_WEIGHTS_MAX);
        }
    }




    static if(!is(typeof(BC_STRING_MAX))) {
        private enum enumMixinStr_BC_STRING_MAX = `enum BC_STRING_MAX = _POSIX2_BC_STRING_MAX;`;
        static if(is(typeof({ mixin(enumMixinStr_BC_STRING_MAX); }))) {
            mixin(enumMixinStr_BC_STRING_MAX);
        }
    }




    static if(!is(typeof(BC_SCALE_MAX))) {
        private enum enumMixinStr_BC_SCALE_MAX = `enum BC_SCALE_MAX = _POSIX2_BC_SCALE_MAX;`;
        static if(is(typeof({ mixin(enumMixinStr_BC_SCALE_MAX); }))) {
            mixin(enumMixinStr_BC_SCALE_MAX);
        }
    }




    static if(!is(typeof(BC_DIM_MAX))) {
        private enum enumMixinStr_BC_DIM_MAX = `enum BC_DIM_MAX = _POSIX2_BC_DIM_MAX;`;
        static if(is(typeof({ mixin(enumMixinStr_BC_DIM_MAX); }))) {
            mixin(enumMixinStr_BC_DIM_MAX);
        }
    }




    static if(!is(typeof(BC_BASE_MAX))) {
        private enum enumMixinStr_BC_BASE_MAX = `enum BC_BASE_MAX = _POSIX2_BC_BASE_MAX;`;
        static if(is(typeof({ mixin(enumMixinStr_BC_BASE_MAX); }))) {
            mixin(enumMixinStr_BC_BASE_MAX);
        }
    }




    static if(!is(typeof(_POSIX2_CHARCLASS_NAME_MAX))) {
        private enum enumMixinStr__POSIX2_CHARCLASS_NAME_MAX = `enum _POSIX2_CHARCLASS_NAME_MAX = 14;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX2_CHARCLASS_NAME_MAX); }))) {
            mixin(enumMixinStr__POSIX2_CHARCLASS_NAME_MAX);
        }
    }




    static if(!is(typeof(_POSIX2_RE_DUP_MAX))) {
        private enum enumMixinStr__POSIX2_RE_DUP_MAX = `enum _POSIX2_RE_DUP_MAX = 255;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX2_RE_DUP_MAX); }))) {
            mixin(enumMixinStr__POSIX2_RE_DUP_MAX);
        }
    }




    static if(!is(typeof(_POSIX2_LINE_MAX))) {
        private enum enumMixinStr__POSIX2_LINE_MAX = `enum _POSIX2_LINE_MAX = 2048;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX2_LINE_MAX); }))) {
            mixin(enumMixinStr__POSIX2_LINE_MAX);
        }
    }




    static if(!is(typeof(_POSIX2_EXPR_NEST_MAX))) {
        private enum enumMixinStr__POSIX2_EXPR_NEST_MAX = `enum _POSIX2_EXPR_NEST_MAX = 32;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX2_EXPR_NEST_MAX); }))) {
            mixin(enumMixinStr__POSIX2_EXPR_NEST_MAX);
        }
    }




    static if(!is(typeof(_POSIX2_COLL_WEIGHTS_MAX))) {
        private enum enumMixinStr__POSIX2_COLL_WEIGHTS_MAX = `enum _POSIX2_COLL_WEIGHTS_MAX = 2;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX2_COLL_WEIGHTS_MAX); }))) {
            mixin(enumMixinStr__POSIX2_COLL_WEIGHTS_MAX);
        }
    }




    static if(!is(typeof(_POSIX2_BC_STRING_MAX))) {
        private enum enumMixinStr__POSIX2_BC_STRING_MAX = `enum _POSIX2_BC_STRING_MAX = 1000;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX2_BC_STRING_MAX); }))) {
            mixin(enumMixinStr__POSIX2_BC_STRING_MAX);
        }
    }




    static if(!is(typeof(_POSIX2_BC_SCALE_MAX))) {
        private enum enumMixinStr__POSIX2_BC_SCALE_MAX = `enum _POSIX2_BC_SCALE_MAX = 99;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX2_BC_SCALE_MAX); }))) {
            mixin(enumMixinStr__POSIX2_BC_SCALE_MAX);
        }
    }




    static if(!is(typeof(_POSIX2_BC_DIM_MAX))) {
        private enum enumMixinStr__POSIX2_BC_DIM_MAX = `enum _POSIX2_BC_DIM_MAX = 2048;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX2_BC_DIM_MAX); }))) {
            mixin(enumMixinStr__POSIX2_BC_DIM_MAX);
        }
    }




    static if(!is(typeof(_POSIX2_BC_BASE_MAX))) {
        private enum enumMixinStr__POSIX2_BC_BASE_MAX = `enum _POSIX2_BC_BASE_MAX = 99;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX2_BC_BASE_MAX); }))) {
            mixin(enumMixinStr__POSIX2_BC_BASE_MAX);
        }
    }




    static if(!is(typeof(_BITS_POSIX2_LIM_H))) {
        private enum enumMixinStr__BITS_POSIX2_LIM_H = `enum _BITS_POSIX2_LIM_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__BITS_POSIX2_LIM_H); }))) {
            mixin(enumMixinStr__BITS_POSIX2_LIM_H);
        }
    }




    static if(!is(typeof(SSIZE_MAX))) {
        private enum enumMixinStr_SSIZE_MAX = `enum SSIZE_MAX = LONG_MAX;`;
        static if(is(typeof({ mixin(enumMixinStr_SSIZE_MAX); }))) {
            mixin(enumMixinStr_SSIZE_MAX);
        }
    }




    static if(!is(typeof(_POSIX_CLOCKRES_MIN))) {
        private enum enumMixinStr__POSIX_CLOCKRES_MIN = `enum _POSIX_CLOCKRES_MIN = 20000000;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_CLOCKRES_MIN); }))) {
            mixin(enumMixinStr__POSIX_CLOCKRES_MIN);
        }
    }




    static if(!is(typeof(_POSIX_TZNAME_MAX))) {
        private enum enumMixinStr__POSIX_TZNAME_MAX = `enum _POSIX_TZNAME_MAX = 6;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_TZNAME_MAX); }))) {
            mixin(enumMixinStr__POSIX_TZNAME_MAX);
        }
    }




    static if(!is(typeof(_POSIX_TTY_NAME_MAX))) {
        private enum enumMixinStr__POSIX_TTY_NAME_MAX = `enum _POSIX_TTY_NAME_MAX = 9;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_TTY_NAME_MAX); }))) {
            mixin(enumMixinStr__POSIX_TTY_NAME_MAX);
        }
    }




    static if(!is(typeof(_POSIX_TIMER_MAX))) {
        private enum enumMixinStr__POSIX_TIMER_MAX = `enum _POSIX_TIMER_MAX = 32;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_TIMER_MAX); }))) {
            mixin(enumMixinStr__POSIX_TIMER_MAX);
        }
    }




    static if(!is(typeof(_POSIX_SYMLOOP_MAX))) {
        private enum enumMixinStr__POSIX_SYMLOOP_MAX = `enum _POSIX_SYMLOOP_MAX = 8;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_SYMLOOP_MAX); }))) {
            mixin(enumMixinStr__POSIX_SYMLOOP_MAX);
        }
    }




    static if(!is(typeof(_POSIX_SYMLINK_MAX))) {
        private enum enumMixinStr__POSIX_SYMLINK_MAX = `enum _POSIX_SYMLINK_MAX = 255;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_SYMLINK_MAX); }))) {
            mixin(enumMixinStr__POSIX_SYMLINK_MAX);
        }
    }




    static if(!is(typeof(_POSIX_STREAM_MAX))) {
        private enum enumMixinStr__POSIX_STREAM_MAX = `enum _POSIX_STREAM_MAX = 8;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_STREAM_MAX); }))) {
            mixin(enumMixinStr__POSIX_STREAM_MAX);
        }
    }




    static if(!is(typeof(_POSIX_SSIZE_MAX))) {
        private enum enumMixinStr__POSIX_SSIZE_MAX = `enum _POSIX_SSIZE_MAX = 32767;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_SSIZE_MAX); }))) {
            mixin(enumMixinStr__POSIX_SSIZE_MAX);
        }
    }




    static if(!is(typeof(_POSIX_SIGQUEUE_MAX))) {
        private enum enumMixinStr__POSIX_SIGQUEUE_MAX = `enum _POSIX_SIGQUEUE_MAX = 32;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_SIGQUEUE_MAX); }))) {
            mixin(enumMixinStr__POSIX_SIGQUEUE_MAX);
        }
    }




    static if(!is(typeof(_POSIX_SEM_VALUE_MAX))) {
        private enum enumMixinStr__POSIX_SEM_VALUE_MAX = `enum _POSIX_SEM_VALUE_MAX = 32767;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_SEM_VALUE_MAX); }))) {
            mixin(enumMixinStr__POSIX_SEM_VALUE_MAX);
        }
    }




    static if(!is(typeof(_POSIX_SEM_NSEMS_MAX))) {
        private enum enumMixinStr__POSIX_SEM_NSEMS_MAX = `enum _POSIX_SEM_NSEMS_MAX = 256;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_SEM_NSEMS_MAX); }))) {
            mixin(enumMixinStr__POSIX_SEM_NSEMS_MAX);
        }
    }




    static if(!is(typeof(_POSIX_RTSIG_MAX))) {
        private enum enumMixinStr__POSIX_RTSIG_MAX = `enum _POSIX_RTSIG_MAX = 8;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_RTSIG_MAX); }))) {
            mixin(enumMixinStr__POSIX_RTSIG_MAX);
        }
    }




    static if(!is(typeof(_POSIX_RE_DUP_MAX))) {
        private enum enumMixinStr__POSIX_RE_DUP_MAX = `enum _POSIX_RE_DUP_MAX = 255;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_RE_DUP_MAX); }))) {
            mixin(enumMixinStr__POSIX_RE_DUP_MAX);
        }
    }




    static if(!is(typeof(_POSIX_PIPE_BUF))) {
        private enum enumMixinStr__POSIX_PIPE_BUF = `enum _POSIX_PIPE_BUF = 512;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_PIPE_BUF); }))) {
            mixin(enumMixinStr__POSIX_PIPE_BUF);
        }
    }




    static if(!is(typeof(_POSIX_PATH_MAX))) {
        private enum enumMixinStr__POSIX_PATH_MAX = `enum _POSIX_PATH_MAX = 256;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_PATH_MAX); }))) {
            mixin(enumMixinStr__POSIX_PATH_MAX);
        }
    }




    static if(!is(typeof(_POSIX_OPEN_MAX))) {
        private enum enumMixinStr__POSIX_OPEN_MAX = `enum _POSIX_OPEN_MAX = 20;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_OPEN_MAX); }))) {
            mixin(enumMixinStr__POSIX_OPEN_MAX);
        }
    }




    static if(!is(typeof(_POSIX_NGROUPS_MAX))) {
        private enum enumMixinStr__POSIX_NGROUPS_MAX = `enum _POSIX_NGROUPS_MAX = 8;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_NGROUPS_MAX); }))) {
            mixin(enumMixinStr__POSIX_NGROUPS_MAX);
        }
    }




    static if(!is(typeof(_POSIX_NAME_MAX))) {
        private enum enumMixinStr__POSIX_NAME_MAX = `enum _POSIX_NAME_MAX = 14;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_NAME_MAX); }))) {
            mixin(enumMixinStr__POSIX_NAME_MAX);
        }
    }




    static if(!is(typeof(_POSIX_MQ_PRIO_MAX))) {
        private enum enumMixinStr__POSIX_MQ_PRIO_MAX = `enum _POSIX_MQ_PRIO_MAX = 32;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_MQ_PRIO_MAX); }))) {
            mixin(enumMixinStr__POSIX_MQ_PRIO_MAX);
        }
    }




    static if(!is(typeof(_POSIX_MQ_OPEN_MAX))) {
        private enum enumMixinStr__POSIX_MQ_OPEN_MAX = `enum _POSIX_MQ_OPEN_MAX = 8;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_MQ_OPEN_MAX); }))) {
            mixin(enumMixinStr__POSIX_MQ_OPEN_MAX);
        }
    }




    static if(!is(typeof(_POSIX_MAX_INPUT))) {
        private enum enumMixinStr__POSIX_MAX_INPUT = `enum _POSIX_MAX_INPUT = 255;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_MAX_INPUT); }))) {
            mixin(enumMixinStr__POSIX_MAX_INPUT);
        }
    }




    static if(!is(typeof(_POSIX_MAX_CANON))) {
        private enum enumMixinStr__POSIX_MAX_CANON = `enum _POSIX_MAX_CANON = 255;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_MAX_CANON); }))) {
            mixin(enumMixinStr__POSIX_MAX_CANON);
        }
    }




    static if(!is(typeof(_POSIX_LOGIN_NAME_MAX))) {
        private enum enumMixinStr__POSIX_LOGIN_NAME_MAX = `enum _POSIX_LOGIN_NAME_MAX = 9;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_LOGIN_NAME_MAX); }))) {
            mixin(enumMixinStr__POSIX_LOGIN_NAME_MAX);
        }
    }




    static if(!is(typeof(_POSIX_LINK_MAX))) {
        private enum enumMixinStr__POSIX_LINK_MAX = `enum _POSIX_LINK_MAX = 8;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_LINK_MAX); }))) {
            mixin(enumMixinStr__POSIX_LINK_MAX);
        }
    }




    static if(!is(typeof(_POSIX_HOST_NAME_MAX))) {
        private enum enumMixinStr__POSIX_HOST_NAME_MAX = `enum _POSIX_HOST_NAME_MAX = 255;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_HOST_NAME_MAX); }))) {
            mixin(enumMixinStr__POSIX_HOST_NAME_MAX);
        }
    }




    static if(!is(typeof(_POSIX_DELAYTIMER_MAX))) {
        private enum enumMixinStr__POSIX_DELAYTIMER_MAX = `enum _POSIX_DELAYTIMER_MAX = 32;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_DELAYTIMER_MAX); }))) {
            mixin(enumMixinStr__POSIX_DELAYTIMER_MAX);
        }
    }




    static if(!is(typeof(_POSIX_CHILD_MAX))) {
        private enum enumMixinStr__POSIX_CHILD_MAX = `enum _POSIX_CHILD_MAX = 25;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_CHILD_MAX); }))) {
            mixin(enumMixinStr__POSIX_CHILD_MAX);
        }
    }




    static if(!is(typeof(_POSIX_ARG_MAX))) {
        private enum enumMixinStr__POSIX_ARG_MAX = `enum _POSIX_ARG_MAX = 4096;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_ARG_MAX); }))) {
            mixin(enumMixinStr__POSIX_ARG_MAX);
        }
    }




    static if(!is(typeof(_POSIX_AIO_MAX))) {
        private enum enumMixinStr__POSIX_AIO_MAX = `enum _POSIX_AIO_MAX = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_AIO_MAX); }))) {
            mixin(enumMixinStr__POSIX_AIO_MAX);
        }
    }




    static if(!is(typeof(_POSIX_AIO_LISTIO_MAX))) {
        private enum enumMixinStr__POSIX_AIO_LISTIO_MAX = `enum _POSIX_AIO_LISTIO_MAX = 2;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_AIO_LISTIO_MAX); }))) {
            mixin(enumMixinStr__POSIX_AIO_LISTIO_MAX);
        }
    }




    static if(!is(typeof(_BITS_POSIX1_LIM_H))) {
        private enum enumMixinStr__BITS_POSIX1_LIM_H = `enum _BITS_POSIX1_LIM_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__BITS_POSIX1_LIM_H); }))) {
            mixin(enumMixinStr__BITS_POSIX1_LIM_H);
        }
    }




    static if(!is(typeof(SEM_VALUE_MAX))) {
        private enum enumMixinStr_SEM_VALUE_MAX = `enum SEM_VALUE_MAX = ( 2147483647 );`;
        static if(is(typeof({ mixin(enumMixinStr_SEM_VALUE_MAX); }))) {
            mixin(enumMixinStr_SEM_VALUE_MAX);
        }
    }




    static if(!is(typeof(MQ_PRIO_MAX))) {
        private enum enumMixinStr_MQ_PRIO_MAX = `enum MQ_PRIO_MAX = 32768;`;
        static if(is(typeof({ mixin(enumMixinStr_MQ_PRIO_MAX); }))) {
            mixin(enumMixinStr_MQ_PRIO_MAX);
        }
    }




    static if(!is(typeof(HOST_NAME_MAX))) {
        private enum enumMixinStr_HOST_NAME_MAX = `enum HOST_NAME_MAX = 64;`;
        static if(is(typeof({ mixin(enumMixinStr_HOST_NAME_MAX); }))) {
            mixin(enumMixinStr_HOST_NAME_MAX);
        }
    }




    static if(!is(typeof(LOGIN_NAME_MAX))) {
        private enum enumMixinStr_LOGIN_NAME_MAX = `enum LOGIN_NAME_MAX = 256;`;
        static if(is(typeof({ mixin(enumMixinStr_LOGIN_NAME_MAX); }))) {
            mixin(enumMixinStr_LOGIN_NAME_MAX);
        }
    }




    static if(!is(typeof(TTY_NAME_MAX))) {
        private enum enumMixinStr_TTY_NAME_MAX = `enum TTY_NAME_MAX = 32;`;
        static if(is(typeof({ mixin(enumMixinStr_TTY_NAME_MAX); }))) {
            mixin(enumMixinStr_TTY_NAME_MAX);
        }
    }




    static if(!is(typeof(DELAYTIMER_MAX))) {
        private enum enumMixinStr_DELAYTIMER_MAX = `enum DELAYTIMER_MAX = 2147483647;`;
        static if(is(typeof({ mixin(enumMixinStr_DELAYTIMER_MAX); }))) {
            mixin(enumMixinStr_DELAYTIMER_MAX);
        }
    }




    static if(!is(typeof(PTHREAD_STACK_MIN))) {
        private enum enumMixinStr_PTHREAD_STACK_MIN = `enum PTHREAD_STACK_MIN = 16384;`;
        static if(is(typeof({ mixin(enumMixinStr_PTHREAD_STACK_MIN); }))) {
            mixin(enumMixinStr_PTHREAD_STACK_MIN);
        }
    }




    static if(!is(typeof(AIO_PRIO_DELTA_MAX))) {
        private enum enumMixinStr_AIO_PRIO_DELTA_MAX = `enum AIO_PRIO_DELTA_MAX = 20;`;
        static if(is(typeof({ mixin(enumMixinStr_AIO_PRIO_DELTA_MAX); }))) {
            mixin(enumMixinStr_AIO_PRIO_DELTA_MAX);
        }
    }




    static if(!is(typeof(_POSIX_THREAD_THREADS_MAX))) {
        private enum enumMixinStr__POSIX_THREAD_THREADS_MAX = `enum _POSIX_THREAD_THREADS_MAX = 64;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_THREAD_THREADS_MAX); }))) {
            mixin(enumMixinStr__POSIX_THREAD_THREADS_MAX);
        }
    }




    static if(!is(typeof(PTHREAD_DESTRUCTOR_ITERATIONS))) {
        private enum enumMixinStr_PTHREAD_DESTRUCTOR_ITERATIONS = `enum PTHREAD_DESTRUCTOR_ITERATIONS = _POSIX_THREAD_DESTRUCTOR_ITERATIONS;`;
        static if(is(typeof({ mixin(enumMixinStr_PTHREAD_DESTRUCTOR_ITERATIONS); }))) {
            mixin(enumMixinStr_PTHREAD_DESTRUCTOR_ITERATIONS);
        }
    }




    static if(!is(typeof(_POSIX_THREAD_DESTRUCTOR_ITERATIONS))) {
        private enum enumMixinStr__POSIX_THREAD_DESTRUCTOR_ITERATIONS = `enum _POSIX_THREAD_DESTRUCTOR_ITERATIONS = 4;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_THREAD_DESTRUCTOR_ITERATIONS); }))) {
            mixin(enumMixinStr__POSIX_THREAD_DESTRUCTOR_ITERATIONS);
        }
    }




    static if(!is(typeof(PTHREAD_KEYS_MAX))) {
        private enum enumMixinStr_PTHREAD_KEYS_MAX = `enum PTHREAD_KEYS_MAX = 1024;`;
        static if(is(typeof({ mixin(enumMixinStr_PTHREAD_KEYS_MAX); }))) {
            mixin(enumMixinStr_PTHREAD_KEYS_MAX);
        }
    }




    static if(!is(typeof(_POSIX_THREAD_KEYS_MAX))) {
        private enum enumMixinStr__POSIX_THREAD_KEYS_MAX = `enum _POSIX_THREAD_KEYS_MAX = 128;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_THREAD_KEYS_MAX); }))) {
            mixin(enumMixinStr__POSIX_THREAD_KEYS_MAX);
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




    static if(!is(typeof(_STDC_PREDEF_H))) {
        private enum enumMixinStr__STDC_PREDEF_H = `enum _STDC_PREDEF_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__STDC_PREDEF_H); }))) {
            mixin(enumMixinStr__STDC_PREDEF_H);
        }
    }




    static if(!is(typeof(RTSIG_MAX))) {
        private enum enumMixinStr_RTSIG_MAX = `enum RTSIG_MAX = 32;`;
        static if(is(typeof({ mixin(enumMixinStr_RTSIG_MAX); }))) {
            mixin(enumMixinStr_RTSIG_MAX);
        }
    }




    static if(!is(typeof(XATTR_LIST_MAX))) {
        private enum enumMixinStr_XATTR_LIST_MAX = `enum XATTR_LIST_MAX = 65536;`;
        static if(is(typeof({ mixin(enumMixinStr_XATTR_LIST_MAX); }))) {
            mixin(enumMixinStr_XATTR_LIST_MAX);
        }
    }




    static if(!is(typeof(XATTR_SIZE_MAX))) {
        private enum enumMixinStr_XATTR_SIZE_MAX = `enum XATTR_SIZE_MAX = 65536;`;
        static if(is(typeof({ mixin(enumMixinStr_XATTR_SIZE_MAX); }))) {
            mixin(enumMixinStr_XATTR_SIZE_MAX);
        }
    }




    static if(!is(typeof(XATTR_NAME_MAX))) {
        private enum enumMixinStr_XATTR_NAME_MAX = `enum XATTR_NAME_MAX = 255;`;
        static if(is(typeof({ mixin(enumMixinStr_XATTR_NAME_MAX); }))) {
            mixin(enumMixinStr_XATTR_NAME_MAX);
        }
    }




    static if(!is(typeof(PIPE_BUF))) {
        private enum enumMixinStr_PIPE_BUF = `enum PIPE_BUF = 4096;`;
        static if(is(typeof({ mixin(enumMixinStr_PIPE_BUF); }))) {
            mixin(enumMixinStr_PIPE_BUF);
        }
    }




    static if(!is(typeof(PATH_MAX))) {
        private enum enumMixinStr_PATH_MAX = `enum PATH_MAX = 4096;`;
        static if(is(typeof({ mixin(enumMixinStr_PATH_MAX); }))) {
            mixin(enumMixinStr_PATH_MAX);
        }
    }




    static if(!is(typeof(NAME_MAX))) {
        private enum enumMixinStr_NAME_MAX = `enum NAME_MAX = 255;`;
        static if(is(typeof({ mixin(enumMixinStr_NAME_MAX); }))) {
            mixin(enumMixinStr_NAME_MAX);
        }
    }




    static if(!is(typeof(MAX_INPUT))) {
        private enum enumMixinStr_MAX_INPUT = `enum MAX_INPUT = 255;`;
        static if(is(typeof({ mixin(enumMixinStr_MAX_INPUT); }))) {
            mixin(enumMixinStr_MAX_INPUT);
        }
    }




    static if(!is(typeof(MAX_CANON))) {
        private enum enumMixinStr_MAX_CANON = `enum MAX_CANON = 255;`;
        static if(is(typeof({ mixin(enumMixinStr_MAX_CANON); }))) {
            mixin(enumMixinStr_MAX_CANON);
        }
    }




    static if(!is(typeof(LINK_MAX))) {
        private enum enumMixinStr_LINK_MAX = `enum LINK_MAX = 127;`;
        static if(is(typeof({ mixin(enumMixinStr_LINK_MAX); }))) {
            mixin(enumMixinStr_LINK_MAX);
        }
    }




    static if(!is(typeof(ARG_MAX))) {
        private enum enumMixinStr_ARG_MAX = `enum ARG_MAX = 131072;`;
        static if(is(typeof({ mixin(enumMixinStr_ARG_MAX); }))) {
            mixin(enumMixinStr_ARG_MAX);
        }
    }




    static if(!is(typeof(NGROUPS_MAX))) {
        private enum enumMixinStr_NGROUPS_MAX = `enum NGROUPS_MAX = 65536;`;
        static if(is(typeof({ mixin(enumMixinStr_NGROUPS_MAX); }))) {
            mixin(enumMixinStr_NGROUPS_MAX);
        }
    }




    static if(!is(typeof(NR_OPEN))) {
        private enum enumMixinStr_NR_OPEN = `enum NR_OPEN = 1024;`;
        static if(is(typeof({ mixin(enumMixinStr_NR_OPEN); }))) {
            mixin(enumMixinStr_NR_OPEN);
        }
    }






    static if(!is(typeof(ULLONG_MAX))) {
        private enum enumMixinStr_ULLONG_MAX = `enum ULLONG_MAX = ( LLONG_MAX * 2ULL + 1 );`;
        static if(is(typeof({ mixin(enumMixinStr_ULLONG_MAX); }))) {
            mixin(enumMixinStr_ULLONG_MAX);
        }
    }




    static if(!is(typeof(LLONG_MAX))) {
        private enum enumMixinStr_LLONG_MAX = `enum LLONG_MAX = 0x7fffffffffffffffLL;`;
        static if(is(typeof({ mixin(enumMixinStr_LLONG_MAX); }))) {
            mixin(enumMixinStr_LLONG_MAX);
        }
    }




    static if(!is(typeof(LLONG_MIN))) {
        private enum enumMixinStr_LLONG_MIN = `enum LLONG_MIN = ( - 0x7fffffffffffffffLL - 1 );`;
        static if(is(typeof({ mixin(enumMixinStr_LLONG_MIN); }))) {
            mixin(enumMixinStr_LLONG_MIN);
        }
    }




    static if(!is(typeof(MB_LEN_MAX))) {
        private enum enumMixinStr_MB_LEN_MAX = `enum MB_LEN_MAX = 16;`;
        static if(is(typeof({ mixin(enumMixinStr_MB_LEN_MAX); }))) {
            mixin(enumMixinStr_MB_LEN_MAX);
        }
    }






    static if(!is(typeof(_LIBC_LIMITS_H_))) {
        private enum enumMixinStr__LIBC_LIMITS_H_ = `enum _LIBC_LIMITS_H_ = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__LIBC_LIMITS_H_); }))) {
            mixin(enumMixinStr__LIBC_LIMITS_H_);
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




    static if(!is(typeof(cudaHostAllocDefault))) {
        private enum enumMixinStr_cudaHostAllocDefault = `enum cudaHostAllocDefault = 0x00;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaHostAllocDefault); }))) {
            mixin(enumMixinStr_cudaHostAllocDefault);
        }
    }




    static if(!is(typeof(cudaHostAllocPortable))) {
        private enum enumMixinStr_cudaHostAllocPortable = `enum cudaHostAllocPortable = 0x01;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaHostAllocPortable); }))) {
            mixin(enumMixinStr_cudaHostAllocPortable);
        }
    }




    static if(!is(typeof(cudaHostAllocMapped))) {
        private enum enumMixinStr_cudaHostAllocMapped = `enum cudaHostAllocMapped = 0x02;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaHostAllocMapped); }))) {
            mixin(enumMixinStr_cudaHostAllocMapped);
        }
    }




    static if(!is(typeof(cudaHostAllocWriteCombined))) {
        private enum enumMixinStr_cudaHostAllocWriteCombined = `enum cudaHostAllocWriteCombined = 0x04;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaHostAllocWriteCombined); }))) {
            mixin(enumMixinStr_cudaHostAllocWriteCombined);
        }
    }




    static if(!is(typeof(cudaHostRegisterDefault))) {
        private enum enumMixinStr_cudaHostRegisterDefault = `enum cudaHostRegisterDefault = 0x00;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaHostRegisterDefault); }))) {
            mixin(enumMixinStr_cudaHostRegisterDefault);
        }
    }




    static if(!is(typeof(cudaHostRegisterPortable))) {
        private enum enumMixinStr_cudaHostRegisterPortable = `enum cudaHostRegisterPortable = 0x01;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaHostRegisterPortable); }))) {
            mixin(enumMixinStr_cudaHostRegisterPortable);
        }
    }




    static if(!is(typeof(cudaHostRegisterMapped))) {
        private enum enumMixinStr_cudaHostRegisterMapped = `enum cudaHostRegisterMapped = 0x02;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaHostRegisterMapped); }))) {
            mixin(enumMixinStr_cudaHostRegisterMapped);
        }
    }




    static if(!is(typeof(cudaHostRegisterIoMemory))) {
        private enum enumMixinStr_cudaHostRegisterIoMemory = `enum cudaHostRegisterIoMemory = 0x04;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaHostRegisterIoMemory); }))) {
            mixin(enumMixinStr_cudaHostRegisterIoMemory);
        }
    }




    static if(!is(typeof(cudaPeerAccessDefault))) {
        private enum enumMixinStr_cudaPeerAccessDefault = `enum cudaPeerAccessDefault = 0x00;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaPeerAccessDefault); }))) {
            mixin(enumMixinStr_cudaPeerAccessDefault);
        }
    }




    static if(!is(typeof(cudaStreamDefault))) {
        private enum enumMixinStr_cudaStreamDefault = `enum cudaStreamDefault = 0x00;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaStreamDefault); }))) {
            mixin(enumMixinStr_cudaStreamDefault);
        }
    }




    static if(!is(typeof(cudaStreamNonBlocking))) {
        private enum enumMixinStr_cudaStreamNonBlocking = `enum cudaStreamNonBlocking = 0x01;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaStreamNonBlocking); }))) {
            mixin(enumMixinStr_cudaStreamNonBlocking);
        }
    }




    static if(!is(typeof(cudaStreamLegacy))) {
        private enum enumMixinStr_cudaStreamLegacy = `enum cudaStreamLegacy = ( cast( cudaStream_t ) 0x1 );`;
        static if(is(typeof({ mixin(enumMixinStr_cudaStreamLegacy); }))) {
            mixin(enumMixinStr_cudaStreamLegacy);
        }
    }




    static if(!is(typeof(cudaStreamPerThread))) {
        private enum enumMixinStr_cudaStreamPerThread = `enum cudaStreamPerThread = ( cast( cudaStream_t ) 0x2 );`;
        static if(is(typeof({ mixin(enumMixinStr_cudaStreamPerThread); }))) {
            mixin(enumMixinStr_cudaStreamPerThread);
        }
    }




    static if(!is(typeof(cudaEventDefault))) {
        private enum enumMixinStr_cudaEventDefault = `enum cudaEventDefault = 0x00;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaEventDefault); }))) {
            mixin(enumMixinStr_cudaEventDefault);
        }
    }




    static if(!is(typeof(cudaEventBlockingSync))) {
        private enum enumMixinStr_cudaEventBlockingSync = `enum cudaEventBlockingSync = 0x01;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaEventBlockingSync); }))) {
            mixin(enumMixinStr_cudaEventBlockingSync);
        }
    }




    static if(!is(typeof(cudaEventDisableTiming))) {
        private enum enumMixinStr_cudaEventDisableTiming = `enum cudaEventDisableTiming = 0x02;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaEventDisableTiming); }))) {
            mixin(enumMixinStr_cudaEventDisableTiming);
        }
    }




    static if(!is(typeof(cudaEventInterprocess))) {
        private enum enumMixinStr_cudaEventInterprocess = `enum cudaEventInterprocess = 0x04;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaEventInterprocess); }))) {
            mixin(enumMixinStr_cudaEventInterprocess);
        }
    }




    static if(!is(typeof(cudaDeviceScheduleAuto))) {
        private enum enumMixinStr_cudaDeviceScheduleAuto = `enum cudaDeviceScheduleAuto = 0x00;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaDeviceScheduleAuto); }))) {
            mixin(enumMixinStr_cudaDeviceScheduleAuto);
        }
    }




    static if(!is(typeof(cudaDeviceScheduleSpin))) {
        private enum enumMixinStr_cudaDeviceScheduleSpin = `enum cudaDeviceScheduleSpin = 0x01;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaDeviceScheduleSpin); }))) {
            mixin(enumMixinStr_cudaDeviceScheduleSpin);
        }
    }




    static if(!is(typeof(cudaDeviceScheduleYield))) {
        private enum enumMixinStr_cudaDeviceScheduleYield = `enum cudaDeviceScheduleYield = 0x02;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaDeviceScheduleYield); }))) {
            mixin(enumMixinStr_cudaDeviceScheduleYield);
        }
    }




    static if(!is(typeof(cudaDeviceScheduleBlockingSync))) {
        private enum enumMixinStr_cudaDeviceScheduleBlockingSync = `enum cudaDeviceScheduleBlockingSync = 0x04;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaDeviceScheduleBlockingSync); }))) {
            mixin(enumMixinStr_cudaDeviceScheduleBlockingSync);
        }
    }




    static if(!is(typeof(cudaDeviceBlockingSync))) {
        private enum enumMixinStr_cudaDeviceBlockingSync = `enum cudaDeviceBlockingSync = 0x04;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaDeviceBlockingSync); }))) {
            mixin(enumMixinStr_cudaDeviceBlockingSync);
        }
    }




    static if(!is(typeof(cudaDeviceScheduleMask))) {
        private enum enumMixinStr_cudaDeviceScheduleMask = `enum cudaDeviceScheduleMask = 0x07;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaDeviceScheduleMask); }))) {
            mixin(enumMixinStr_cudaDeviceScheduleMask);
        }
    }




    static if(!is(typeof(cudaDeviceMapHost))) {
        private enum enumMixinStr_cudaDeviceMapHost = `enum cudaDeviceMapHost = 0x08;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaDeviceMapHost); }))) {
            mixin(enumMixinStr_cudaDeviceMapHost);
        }
    }




    static if(!is(typeof(cudaDeviceLmemResizeToMax))) {
        private enum enumMixinStr_cudaDeviceLmemResizeToMax = `enum cudaDeviceLmemResizeToMax = 0x10;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaDeviceLmemResizeToMax); }))) {
            mixin(enumMixinStr_cudaDeviceLmemResizeToMax);
        }
    }




    static if(!is(typeof(cudaDeviceMask))) {
        private enum enumMixinStr_cudaDeviceMask = `enum cudaDeviceMask = 0x1f;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaDeviceMask); }))) {
            mixin(enumMixinStr_cudaDeviceMask);
        }
    }




    static if(!is(typeof(cudaArrayDefault))) {
        private enum enumMixinStr_cudaArrayDefault = `enum cudaArrayDefault = 0x00;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaArrayDefault); }))) {
            mixin(enumMixinStr_cudaArrayDefault);
        }
    }




    static if(!is(typeof(cudaArrayLayered))) {
        private enum enumMixinStr_cudaArrayLayered = `enum cudaArrayLayered = 0x01;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaArrayLayered); }))) {
            mixin(enumMixinStr_cudaArrayLayered);
        }
    }




    static if(!is(typeof(cudaArraySurfaceLoadStore))) {
        private enum enumMixinStr_cudaArraySurfaceLoadStore = `enum cudaArraySurfaceLoadStore = 0x02;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaArraySurfaceLoadStore); }))) {
            mixin(enumMixinStr_cudaArraySurfaceLoadStore);
        }
    }




    static if(!is(typeof(cudaArrayCubemap))) {
        private enum enumMixinStr_cudaArrayCubemap = `enum cudaArrayCubemap = 0x04;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaArrayCubemap); }))) {
            mixin(enumMixinStr_cudaArrayCubemap);
        }
    }




    static if(!is(typeof(cudaArrayTextureGather))) {
        private enum enumMixinStr_cudaArrayTextureGather = `enum cudaArrayTextureGather = 0x08;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaArrayTextureGather); }))) {
            mixin(enumMixinStr_cudaArrayTextureGather);
        }
    }




    static if(!is(typeof(cudaArrayColorAttachment))) {
        private enum enumMixinStr_cudaArrayColorAttachment = `enum cudaArrayColorAttachment = 0x20;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaArrayColorAttachment); }))) {
            mixin(enumMixinStr_cudaArrayColorAttachment);
        }
    }




    static if(!is(typeof(cudaIpcMemLazyEnablePeerAccess))) {
        private enum enumMixinStr_cudaIpcMemLazyEnablePeerAccess = `enum cudaIpcMemLazyEnablePeerAccess = 0x01;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaIpcMemLazyEnablePeerAccess); }))) {
            mixin(enumMixinStr_cudaIpcMemLazyEnablePeerAccess);
        }
    }




    static if(!is(typeof(cudaMemAttachGlobal))) {
        private enum enumMixinStr_cudaMemAttachGlobal = `enum cudaMemAttachGlobal = 0x01;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaMemAttachGlobal); }))) {
            mixin(enumMixinStr_cudaMemAttachGlobal);
        }
    }




    static if(!is(typeof(cudaMemAttachHost))) {
        private enum enumMixinStr_cudaMemAttachHost = `enum cudaMemAttachHost = 0x02;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaMemAttachHost); }))) {
            mixin(enumMixinStr_cudaMemAttachHost);
        }
    }




    static if(!is(typeof(cudaMemAttachSingle))) {
        private enum enumMixinStr_cudaMemAttachSingle = `enum cudaMemAttachSingle = 0x04;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaMemAttachSingle); }))) {
            mixin(enumMixinStr_cudaMemAttachSingle);
        }
    }




    static if(!is(typeof(cudaOccupancyDefault))) {
        private enum enumMixinStr_cudaOccupancyDefault = `enum cudaOccupancyDefault = 0x00;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaOccupancyDefault); }))) {
            mixin(enumMixinStr_cudaOccupancyDefault);
        }
    }




    static if(!is(typeof(cudaOccupancyDisableCachingOverride))) {
        private enum enumMixinStr_cudaOccupancyDisableCachingOverride = `enum cudaOccupancyDisableCachingOverride = 0x01;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaOccupancyDisableCachingOverride); }))) {
            mixin(enumMixinStr_cudaOccupancyDisableCachingOverride);
        }
    }




    static if(!is(typeof(cudaCpuDeviceId))) {
        private enum enumMixinStr_cudaCpuDeviceId = `enum cudaCpuDeviceId = ( cast( int ) - 1 );`;
        static if(is(typeof({ mixin(enumMixinStr_cudaCpuDeviceId); }))) {
            mixin(enumMixinStr_cudaCpuDeviceId);
        }
    }




    static if(!is(typeof(cudaInvalidDeviceId))) {
        private enum enumMixinStr_cudaInvalidDeviceId = `enum cudaInvalidDeviceId = ( cast( int ) - 2 );`;
        static if(is(typeof({ mixin(enumMixinStr_cudaInvalidDeviceId); }))) {
            mixin(enumMixinStr_cudaInvalidDeviceId);
        }
    }




    static if(!is(typeof(cudaCooperativeLaunchMultiDeviceNoPreSync))) {
        private enum enumMixinStr_cudaCooperativeLaunchMultiDeviceNoPreSync = `enum cudaCooperativeLaunchMultiDeviceNoPreSync = 0x01;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaCooperativeLaunchMultiDeviceNoPreSync); }))) {
            mixin(enumMixinStr_cudaCooperativeLaunchMultiDeviceNoPreSync);
        }
    }




    static if(!is(typeof(cudaCooperativeLaunchMultiDeviceNoPostSync))) {
        private enum enumMixinStr_cudaCooperativeLaunchMultiDeviceNoPostSync = `enum cudaCooperativeLaunchMultiDeviceNoPostSync = 0x02;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaCooperativeLaunchMultiDeviceNoPostSync); }))) {
            mixin(enumMixinStr_cudaCooperativeLaunchMultiDeviceNoPostSync);
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






    static if(!is(typeof(NULL))) {
        private enum enumMixinStr_NULL = `enum NULL = ( cast( void * ) 0 );`;
        static if(is(typeof({ mixin(enumMixinStr_NULL); }))) {
            mixin(enumMixinStr_NULL);
        }
    }
    static if(!is(typeof(CHAR_MAX))) {
        private enum enumMixinStr_CHAR_MAX = `enum CHAR_MAX = 0x7f;`;
        static if(is(typeof({ mixin(enumMixinStr_CHAR_MAX); }))) {
            mixin(enumMixinStr_CHAR_MAX);
        }
    }




    static if(!is(typeof(CHAR_MIN))) {
        private enum enumMixinStr_CHAR_MIN = `enum CHAR_MIN = SCHAR_MIN;`;
        static if(is(typeof({ mixin(enumMixinStr_CHAR_MIN); }))) {
            mixin(enumMixinStr_CHAR_MIN);
        }
    }




    static if(!is(typeof(CHAR_BIT))) {
        private enum enumMixinStr_CHAR_BIT = `enum CHAR_BIT = 8;`;
        static if(is(typeof({ mixin(enumMixinStr_CHAR_BIT); }))) {
            mixin(enumMixinStr_CHAR_BIT);
        }
    }




    static if(!is(typeof(ULONG_MAX))) {
        private enum enumMixinStr_ULONG_MAX = `enum ULONG_MAX = ( 0x7fffffffffffffffL * 2UL + 1UL );`;
        static if(is(typeof({ mixin(enumMixinStr_ULONG_MAX); }))) {
            mixin(enumMixinStr_ULONG_MAX);
        }
    }






    static if(!is(typeof(UINT_MAX))) {
        private enum enumMixinStr_UINT_MAX = `enum UINT_MAX = ( 0x7fffffff * 2U + 1U );`;
        static if(is(typeof({ mixin(enumMixinStr_UINT_MAX); }))) {
            mixin(enumMixinStr_UINT_MAX);
        }
    }




    static if(!is(typeof(USHRT_MAX))) {
        private enum enumMixinStr_USHRT_MAX = `enum USHRT_MAX = ( 0x7fff * 2 + 1 );`;
        static if(is(typeof({ mixin(enumMixinStr_USHRT_MAX); }))) {
            mixin(enumMixinStr_USHRT_MAX);
        }
    }




    static if(!is(typeof(UCHAR_MAX))) {
        private enum enumMixinStr_UCHAR_MAX = `enum UCHAR_MAX = ( 0x7f * 2 + 1 );`;
        static if(is(typeof({ mixin(enumMixinStr_UCHAR_MAX); }))) {
            mixin(enumMixinStr_UCHAR_MAX);
        }
    }




    static if(!is(typeof(LONG_MIN))) {
        private enum enumMixinStr_LONG_MIN = `enum LONG_MIN = ( - 0x7fffffffffffffffL - 1L );`;
        static if(is(typeof({ mixin(enumMixinStr_LONG_MIN); }))) {
            mixin(enumMixinStr_LONG_MIN);
        }
    }




    static if(!is(typeof(INT_MIN))) {
        private enum enumMixinStr_INT_MIN = `enum INT_MIN = ( - 0x7fffffff - 1 );`;
        static if(is(typeof({ mixin(enumMixinStr_INT_MIN); }))) {
            mixin(enumMixinStr_INT_MIN);
        }
    }




    static if(!is(typeof(SHRT_MIN))) {
        private enum enumMixinStr_SHRT_MIN = `enum SHRT_MIN = ( - 0x7fff - 1 );`;
        static if(is(typeof({ mixin(enumMixinStr_SHRT_MIN); }))) {
            mixin(enumMixinStr_SHRT_MIN);
        }
    }




    static if(!is(typeof(SCHAR_MIN))) {
        private enum enumMixinStr_SCHAR_MIN = `enum SCHAR_MIN = ( - 0x7f - 1 );`;
        static if(is(typeof({ mixin(enumMixinStr_SCHAR_MIN); }))) {
            mixin(enumMixinStr_SCHAR_MIN);
        }
    }




    static if(!is(typeof(LONG_MAX))) {
        private enum enumMixinStr_LONG_MAX = `enum LONG_MAX = 0x7fffffffffffffffL;`;
        static if(is(typeof({ mixin(enumMixinStr_LONG_MAX); }))) {
            mixin(enumMixinStr_LONG_MAX);
        }
    }




    static if(!is(typeof(INT_MAX))) {
        private enum enumMixinStr_INT_MAX = `enum INT_MAX = 0x7fffffff;`;
        static if(is(typeof({ mixin(enumMixinStr_INT_MAX); }))) {
            mixin(enumMixinStr_INT_MAX);
        }
    }




    static if(!is(typeof(SHRT_MAX))) {
        private enum enumMixinStr_SHRT_MAX = `enum SHRT_MAX = 0x7fff;`;
        static if(is(typeof({ mixin(enumMixinStr_SHRT_MAX); }))) {
            mixin(enumMixinStr_SHRT_MAX);
        }
    }




    static if(!is(typeof(SCHAR_MAX))) {
        private enum enumMixinStr_SCHAR_MAX = `enum SCHAR_MAX = 0x7f;`;
        static if(is(typeof({ mixin(enumMixinStr_SCHAR_MAX); }))) {
            mixin(enumMixinStr_SCHAR_MAX);
        }
    }
    static if(!is(typeof(cudaDevicePropDontCare))) {
        private enum enumMixinStr_cudaDevicePropDontCare = `enum cudaDevicePropDontCare = { { '\0' } , { { 0 } } , { '\0' } , 0 , 0 , 0 , 0 , 0 , 0 , 0 , { 0 , 0 , 0 } , { 0 , 0 , 0 } , 0 , 0 , - 1 , - 1 , 0 , 0 , - 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , { 0 , 0 } , { 0 , 0 } , { 0 , 0 , 0 } , { 0 , 0 } , { 0 , 0 , 0 } , { 0 , 0 , 0 } , 0 , { 0 , 0 } , { 0 , 0 , 0 } , { 0 , 0 } , 0 , { 0 , 0 } , { 0 , 0 , 0 } , { 0 , 0 } , { 0 , 0 , 0 } , 0 , { 0 , 0 } , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , };`;
        static if(is(typeof({ mixin(enumMixinStr_cudaDevicePropDontCare); }))) {
            mixin(enumMixinStr_cudaDevicePropDontCare);
        }
    }




    static if(!is(typeof(CUDA_IPC_HANDLE_SIZE))) {
        private enum enumMixinStr_CUDA_IPC_HANDLE_SIZE = `enum CUDA_IPC_HANDLE_SIZE = 64;`;
        static if(is(typeof({ mixin(enumMixinStr_CUDA_IPC_HANDLE_SIZE); }))) {
            mixin(enumMixinStr_CUDA_IPC_HANDLE_SIZE);
        }
    }




    static if(!is(typeof(cudaExternalMemoryDedicated))) {
        private enum enumMixinStr_cudaExternalMemoryDedicated = `enum cudaExternalMemoryDedicated = 0x1;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaExternalMemoryDedicated); }))) {
            mixin(enumMixinStr_cudaExternalMemoryDedicated);
        }
    }




    static if(!is(typeof(cudaExternalSemaphoreSignalSkipNvSciBufMemSync))) {
        private enum enumMixinStr_cudaExternalSemaphoreSignalSkipNvSciBufMemSync = `enum cudaExternalSemaphoreSignalSkipNvSciBufMemSync = 0x01;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaExternalSemaphoreSignalSkipNvSciBufMemSync); }))) {
            mixin(enumMixinStr_cudaExternalSemaphoreSignalSkipNvSciBufMemSync);
        }
    }




    static if(!is(typeof(cudaExternalSemaphoreWaitSkipNvSciBufMemSync))) {
        private enum enumMixinStr_cudaExternalSemaphoreWaitSkipNvSciBufMemSync = `enum cudaExternalSemaphoreWaitSkipNvSciBufMemSync = 0x02;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaExternalSemaphoreWaitSkipNvSciBufMemSync); }))) {
            mixin(enumMixinStr_cudaExternalSemaphoreWaitSkipNvSciBufMemSync);
        }
    }




    static if(!is(typeof(cudaNvSciSyncAttrSignal))) {
        private enum enumMixinStr_cudaNvSciSyncAttrSignal = `enum cudaNvSciSyncAttrSignal = 0x1;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaNvSciSyncAttrSignal); }))) {
            mixin(enumMixinStr_cudaNvSciSyncAttrSignal);
        }
    }




    static if(!is(typeof(cudaNvSciSyncAttrWait))) {
        private enum enumMixinStr_cudaNvSciSyncAttrWait = `enum cudaNvSciSyncAttrWait = 0x2;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaNvSciSyncAttrWait); }))) {
            mixin(enumMixinStr_cudaNvSciSyncAttrWait);
        }
    }






    static if(!is(typeof(cudaSurfaceType1D))) {
        private enum enumMixinStr_cudaSurfaceType1D = `enum cudaSurfaceType1D = 0x01;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaSurfaceType1D); }))) {
            mixin(enumMixinStr_cudaSurfaceType1D);
        }
    }




    static if(!is(typeof(cudaSurfaceType2D))) {
        private enum enumMixinStr_cudaSurfaceType2D = `enum cudaSurfaceType2D = 0x02;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaSurfaceType2D); }))) {
            mixin(enumMixinStr_cudaSurfaceType2D);
        }
    }




    static if(!is(typeof(cudaSurfaceType3D))) {
        private enum enumMixinStr_cudaSurfaceType3D = `enum cudaSurfaceType3D = 0x03;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaSurfaceType3D); }))) {
            mixin(enumMixinStr_cudaSurfaceType3D);
        }
    }




    static if(!is(typeof(cudaSurfaceTypeCubemap))) {
        private enum enumMixinStr_cudaSurfaceTypeCubemap = `enum cudaSurfaceTypeCubemap = 0x0C;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaSurfaceTypeCubemap); }))) {
            mixin(enumMixinStr_cudaSurfaceTypeCubemap);
        }
    }




    static if(!is(typeof(cudaSurfaceType1DLayered))) {
        private enum enumMixinStr_cudaSurfaceType1DLayered = `enum cudaSurfaceType1DLayered = 0xF1;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaSurfaceType1DLayered); }))) {
            mixin(enumMixinStr_cudaSurfaceType1DLayered);
        }
    }




    static if(!is(typeof(cudaSurfaceType2DLayered))) {
        private enum enumMixinStr_cudaSurfaceType2DLayered = `enum cudaSurfaceType2DLayered = 0xF2;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaSurfaceType2DLayered); }))) {
            mixin(enumMixinStr_cudaSurfaceType2DLayered);
        }
    }




    static if(!is(typeof(cudaSurfaceTypeCubemapLayered))) {
        private enum enumMixinStr_cudaSurfaceTypeCubemapLayered = `enum cudaSurfaceTypeCubemapLayered = 0xFC;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaSurfaceTypeCubemapLayered); }))) {
            mixin(enumMixinStr_cudaSurfaceTypeCubemapLayered);
        }
    }






    static if(!is(typeof(cudaTextureType1D))) {
        private enum enumMixinStr_cudaTextureType1D = `enum cudaTextureType1D = 0x01;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaTextureType1D); }))) {
            mixin(enumMixinStr_cudaTextureType1D);
        }
    }




    static if(!is(typeof(cudaTextureType2D))) {
        private enum enumMixinStr_cudaTextureType2D = `enum cudaTextureType2D = 0x02;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaTextureType2D); }))) {
            mixin(enumMixinStr_cudaTextureType2D);
        }
    }




    static if(!is(typeof(cudaTextureType3D))) {
        private enum enumMixinStr_cudaTextureType3D = `enum cudaTextureType3D = 0x03;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaTextureType3D); }))) {
            mixin(enumMixinStr_cudaTextureType3D);
        }
    }




    static if(!is(typeof(cudaTextureTypeCubemap))) {
        private enum enumMixinStr_cudaTextureTypeCubemap = `enum cudaTextureTypeCubemap = 0x0C;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaTextureTypeCubemap); }))) {
            mixin(enumMixinStr_cudaTextureTypeCubemap);
        }
    }




    static if(!is(typeof(cudaTextureType1DLayered))) {
        private enum enumMixinStr_cudaTextureType1DLayered = `enum cudaTextureType1DLayered = 0xF1;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaTextureType1DLayered); }))) {
            mixin(enumMixinStr_cudaTextureType1DLayered);
        }
    }




    static if(!is(typeof(cudaTextureType2DLayered))) {
        private enum enumMixinStr_cudaTextureType2DLayered = `enum cudaTextureType2DLayered = 0xF2;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaTextureType2DLayered); }))) {
            mixin(enumMixinStr_cudaTextureType2DLayered);
        }
    }




    static if(!is(typeof(cudaTextureTypeCubemapLayered))) {
        private enum enumMixinStr_cudaTextureTypeCubemapLayered = `enum cudaTextureTypeCubemapLayered = 0xFC;`;
        static if(is(typeof({ mixin(enumMixinStr_cudaTextureTypeCubemapLayered); }))) {
            mixin(enumMixinStr_cudaTextureTypeCubemapLayered);
        }
    }






    static if(!is(typeof(__cuda_builtin_vector_align8))) {
        private enum enumMixinStr___cuda_builtin_vector_align8 = `enum __cuda_builtin_vector_align8 = ( tag , members ) __attribute__ ( ( aligned ( 8 ) ) ) tag
 { members
 };`;
        static if(is(typeof({ mixin(enumMixinStr___cuda_builtin_vector_align8); }))) {
            mixin(enumMixinStr___cuda_builtin_vector_align8);
        }
    }



}
