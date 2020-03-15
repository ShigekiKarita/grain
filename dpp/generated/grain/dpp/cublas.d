module grain.dpp.cublas;


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
    enum libraryPropertyType_t
    {
        MAJOR_VERSION = 0,
        MINOR_VERSION = 1,
        PATCH_LEVEL = 2,
    }
    enum MAJOR_VERSION = libraryPropertyType_t.MAJOR_VERSION;
    enum MINOR_VERSION = libraryPropertyType_t.MINOR_VERSION;
    enum PATCH_LEVEL = libraryPropertyType_t.PATCH_LEVEL;
    alias libraryPropertyType = libraryPropertyType_t;
    enum cudaDataType_t
    {
        CUDA_R_16F = 2,
        CUDA_C_16F = 6,
        CUDA_R_32F = 0,
        CUDA_C_32F = 4,
        CUDA_R_64F = 1,
        CUDA_C_64F = 5,
        CUDA_R_8I = 3,
        CUDA_C_8I = 7,
        CUDA_R_8U = 8,
        CUDA_C_8U = 9,
        CUDA_R_32I = 10,
        CUDA_C_32I = 11,
        CUDA_R_32U = 12,
        CUDA_C_32U = 13,
    }
    enum CUDA_R_16F = cudaDataType_t.CUDA_R_16F;
    enum CUDA_C_16F = cudaDataType_t.CUDA_C_16F;
    enum CUDA_R_32F = cudaDataType_t.CUDA_R_32F;
    enum CUDA_C_32F = cudaDataType_t.CUDA_C_32F;
    enum CUDA_R_64F = cudaDataType_t.CUDA_R_64F;
    enum CUDA_C_64F = cudaDataType_t.CUDA_C_64F;
    enum CUDA_R_8I = cudaDataType_t.CUDA_R_8I;
    enum CUDA_C_8I = cudaDataType_t.CUDA_C_8I;
    enum CUDA_R_8U = cudaDataType_t.CUDA_R_8U;
    enum CUDA_C_8U = cudaDataType_t.CUDA_C_8U;
    enum CUDA_R_32I = cudaDataType_t.CUDA_R_32I;
    enum CUDA_C_32I = cudaDataType_t.CUDA_C_32I;
    enum CUDA_R_32U = cudaDataType_t.CUDA_R_32U;
    enum CUDA_C_32U = cudaDataType_t.CUDA_C_32U;
    alias cudaDataType = cudaDataType_t;
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
    struct max_align_t
    {
        long __clang_max_align_nonce1;
        real __clang_max_align_nonce2;
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
    alias ptrdiff_t = c_long;
    alias size_t = c_ulong;
    alias wchar_t = int;
    struct __half2_raw
    {
        ushort x;
        ushort y;
    }
    struct __half_raw
    {
        ushort x;
    }
    alias cublasStatus_t = _Anonymous_17;
    enum _Anonymous_17
    {
        CUBLAS_STATUS_SUCCESS = 0,
        CUBLAS_STATUS_NOT_INITIALIZED = 1,
        CUBLAS_STATUS_ALLOC_FAILED = 3,
        CUBLAS_STATUS_INVALID_VALUE = 7,
        CUBLAS_STATUS_ARCH_MISMATCH = 8,
        CUBLAS_STATUS_MAPPING_ERROR = 11,
        CUBLAS_STATUS_EXECUTION_FAILED = 13,
        CUBLAS_STATUS_INTERNAL_ERROR = 14,
        CUBLAS_STATUS_NOT_SUPPORTED = 15,
        CUBLAS_STATUS_LICENSE_ERROR = 16,
    }
    enum CUBLAS_STATUS_SUCCESS = _Anonymous_17.CUBLAS_STATUS_SUCCESS;
    enum CUBLAS_STATUS_NOT_INITIALIZED = _Anonymous_17.CUBLAS_STATUS_NOT_INITIALIZED;
    enum CUBLAS_STATUS_ALLOC_FAILED = _Anonymous_17.CUBLAS_STATUS_ALLOC_FAILED;
    enum CUBLAS_STATUS_INVALID_VALUE = _Anonymous_17.CUBLAS_STATUS_INVALID_VALUE;
    enum CUBLAS_STATUS_ARCH_MISMATCH = _Anonymous_17.CUBLAS_STATUS_ARCH_MISMATCH;
    enum CUBLAS_STATUS_MAPPING_ERROR = _Anonymous_17.CUBLAS_STATUS_MAPPING_ERROR;
    enum CUBLAS_STATUS_EXECUTION_FAILED = _Anonymous_17.CUBLAS_STATUS_EXECUTION_FAILED;
    enum CUBLAS_STATUS_INTERNAL_ERROR = _Anonymous_17.CUBLAS_STATUS_INTERNAL_ERROR;
    enum CUBLAS_STATUS_NOT_SUPPORTED = _Anonymous_17.CUBLAS_STATUS_NOT_SUPPORTED;
    enum CUBLAS_STATUS_LICENSE_ERROR = _Anonymous_17.CUBLAS_STATUS_LICENSE_ERROR;
    alias cublasFillMode_t = _Anonymous_18;
    enum _Anonymous_18
    {
        CUBLAS_FILL_MODE_LOWER = 0,
        CUBLAS_FILL_MODE_UPPER = 1,
        CUBLAS_FILL_MODE_FULL = 2,
    }
    enum CUBLAS_FILL_MODE_LOWER = _Anonymous_18.CUBLAS_FILL_MODE_LOWER;
    enum CUBLAS_FILL_MODE_UPPER = _Anonymous_18.CUBLAS_FILL_MODE_UPPER;
    enum CUBLAS_FILL_MODE_FULL = _Anonymous_18.CUBLAS_FILL_MODE_FULL;
    alias cublasDiagType_t = _Anonymous_19;
    enum _Anonymous_19
    {
        CUBLAS_DIAG_NON_UNIT = 0,
        CUBLAS_DIAG_UNIT = 1,
    }
    enum CUBLAS_DIAG_NON_UNIT = _Anonymous_19.CUBLAS_DIAG_NON_UNIT;
    enum CUBLAS_DIAG_UNIT = _Anonymous_19.CUBLAS_DIAG_UNIT;
    alias cublasSideMode_t = _Anonymous_20;
    enum _Anonymous_20
    {
        CUBLAS_SIDE_LEFT = 0,
        CUBLAS_SIDE_RIGHT = 1,
    }
    enum CUBLAS_SIDE_LEFT = _Anonymous_20.CUBLAS_SIDE_LEFT;
    enum CUBLAS_SIDE_RIGHT = _Anonymous_20.CUBLAS_SIDE_RIGHT;
    alias cublasOperation_t = _Anonymous_21;
    enum _Anonymous_21
    {
        CUBLAS_OP_N = 0,
        CUBLAS_OP_T = 1,
        CUBLAS_OP_C = 2,
        CUBLAS_OP_HERMITAN = 2,
        CUBLAS_OP_CONJG = 3,
    }
    enum CUBLAS_OP_N = _Anonymous_21.CUBLAS_OP_N;
    enum CUBLAS_OP_T = _Anonymous_21.CUBLAS_OP_T;
    enum CUBLAS_OP_C = _Anonymous_21.CUBLAS_OP_C;
    enum CUBLAS_OP_HERMITAN = _Anonymous_21.CUBLAS_OP_HERMITAN;
    enum CUBLAS_OP_CONJG = _Anonymous_21.CUBLAS_OP_CONJG;
    alias cublasPointerMode_t = _Anonymous_22;
    enum _Anonymous_22
    {
        CUBLAS_POINTER_MODE_HOST = 0,
        CUBLAS_POINTER_MODE_DEVICE = 1,
    }
    enum CUBLAS_POINTER_MODE_HOST = _Anonymous_22.CUBLAS_POINTER_MODE_HOST;
    enum CUBLAS_POINTER_MODE_DEVICE = _Anonymous_22.CUBLAS_POINTER_MODE_DEVICE;
    alias cublasAtomicsMode_t = _Anonymous_23;
    enum _Anonymous_23
    {
        CUBLAS_ATOMICS_NOT_ALLOWED = 0,
        CUBLAS_ATOMICS_ALLOWED = 1,
    }
    enum CUBLAS_ATOMICS_NOT_ALLOWED = _Anonymous_23.CUBLAS_ATOMICS_NOT_ALLOWED;
    enum CUBLAS_ATOMICS_ALLOWED = _Anonymous_23.CUBLAS_ATOMICS_ALLOWED;
    alias cublasGemmAlgo_t = _Anonymous_24;
    enum _Anonymous_24
    {
        CUBLAS_GEMM_DFALT = -1,
        CUBLAS_GEMM_DEFAULT = -1,
        CUBLAS_GEMM_ALGO0 = 0,
        CUBLAS_GEMM_ALGO1 = 1,
        CUBLAS_GEMM_ALGO2 = 2,
        CUBLAS_GEMM_ALGO3 = 3,
        CUBLAS_GEMM_ALGO4 = 4,
        CUBLAS_GEMM_ALGO5 = 5,
        CUBLAS_GEMM_ALGO6 = 6,
        CUBLAS_GEMM_ALGO7 = 7,
        CUBLAS_GEMM_ALGO8 = 8,
        CUBLAS_GEMM_ALGO9 = 9,
        CUBLAS_GEMM_ALGO10 = 10,
        CUBLAS_GEMM_ALGO11 = 11,
        CUBLAS_GEMM_ALGO12 = 12,
        CUBLAS_GEMM_ALGO13 = 13,
        CUBLAS_GEMM_ALGO14 = 14,
        CUBLAS_GEMM_ALGO15 = 15,
        CUBLAS_GEMM_ALGO16 = 16,
        CUBLAS_GEMM_ALGO17 = 17,
        CUBLAS_GEMM_ALGO18 = 18,
        CUBLAS_GEMM_ALGO19 = 19,
        CUBLAS_GEMM_ALGO20 = 20,
        CUBLAS_GEMM_ALGO21 = 21,
        CUBLAS_GEMM_ALGO22 = 22,
        CUBLAS_GEMM_ALGO23 = 23,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99,
        CUBLAS_GEMM_DFALT_TENSOR_OP = 99,
        CUBLAS_GEMM_ALGO0_TENSOR_OP = 100,
        CUBLAS_GEMM_ALGO1_TENSOR_OP = 101,
        CUBLAS_GEMM_ALGO2_TENSOR_OP = 102,
        CUBLAS_GEMM_ALGO3_TENSOR_OP = 103,
        CUBLAS_GEMM_ALGO4_TENSOR_OP = 104,
        CUBLAS_GEMM_ALGO5_TENSOR_OP = 105,
        CUBLAS_GEMM_ALGO6_TENSOR_OP = 106,
        CUBLAS_GEMM_ALGO7_TENSOR_OP = 107,
        CUBLAS_GEMM_ALGO8_TENSOR_OP = 108,
        CUBLAS_GEMM_ALGO9_TENSOR_OP = 109,
        CUBLAS_GEMM_ALGO10_TENSOR_OP = 110,
        CUBLAS_GEMM_ALGO11_TENSOR_OP = 111,
        CUBLAS_GEMM_ALGO12_TENSOR_OP = 112,
        CUBLAS_GEMM_ALGO13_TENSOR_OP = 113,
        CUBLAS_GEMM_ALGO14_TENSOR_OP = 114,
        CUBLAS_GEMM_ALGO15_TENSOR_OP = 115,
    }
    enum CUBLAS_GEMM_DFALT = _Anonymous_24.CUBLAS_GEMM_DFALT;
    enum CUBLAS_GEMM_DEFAULT = _Anonymous_24.CUBLAS_GEMM_DEFAULT;
    enum CUBLAS_GEMM_ALGO0 = _Anonymous_24.CUBLAS_GEMM_ALGO0;
    enum CUBLAS_GEMM_ALGO1 = _Anonymous_24.CUBLAS_GEMM_ALGO1;
    enum CUBLAS_GEMM_ALGO2 = _Anonymous_24.CUBLAS_GEMM_ALGO2;
    enum CUBLAS_GEMM_ALGO3 = _Anonymous_24.CUBLAS_GEMM_ALGO3;
    enum CUBLAS_GEMM_ALGO4 = _Anonymous_24.CUBLAS_GEMM_ALGO4;
    enum CUBLAS_GEMM_ALGO5 = _Anonymous_24.CUBLAS_GEMM_ALGO5;
    enum CUBLAS_GEMM_ALGO6 = _Anonymous_24.CUBLAS_GEMM_ALGO6;
    enum CUBLAS_GEMM_ALGO7 = _Anonymous_24.CUBLAS_GEMM_ALGO7;
    enum CUBLAS_GEMM_ALGO8 = _Anonymous_24.CUBLAS_GEMM_ALGO8;
    enum CUBLAS_GEMM_ALGO9 = _Anonymous_24.CUBLAS_GEMM_ALGO9;
    enum CUBLAS_GEMM_ALGO10 = _Anonymous_24.CUBLAS_GEMM_ALGO10;
    enum CUBLAS_GEMM_ALGO11 = _Anonymous_24.CUBLAS_GEMM_ALGO11;
    enum CUBLAS_GEMM_ALGO12 = _Anonymous_24.CUBLAS_GEMM_ALGO12;
    enum CUBLAS_GEMM_ALGO13 = _Anonymous_24.CUBLAS_GEMM_ALGO13;
    enum CUBLAS_GEMM_ALGO14 = _Anonymous_24.CUBLAS_GEMM_ALGO14;
    enum CUBLAS_GEMM_ALGO15 = _Anonymous_24.CUBLAS_GEMM_ALGO15;
    enum CUBLAS_GEMM_ALGO16 = _Anonymous_24.CUBLAS_GEMM_ALGO16;
    enum CUBLAS_GEMM_ALGO17 = _Anonymous_24.CUBLAS_GEMM_ALGO17;
    enum CUBLAS_GEMM_ALGO18 = _Anonymous_24.CUBLAS_GEMM_ALGO18;
    enum CUBLAS_GEMM_ALGO19 = _Anonymous_24.CUBLAS_GEMM_ALGO19;
    enum CUBLAS_GEMM_ALGO20 = _Anonymous_24.CUBLAS_GEMM_ALGO20;
    enum CUBLAS_GEMM_ALGO21 = _Anonymous_24.CUBLAS_GEMM_ALGO21;
    enum CUBLAS_GEMM_ALGO22 = _Anonymous_24.CUBLAS_GEMM_ALGO22;
    enum CUBLAS_GEMM_ALGO23 = _Anonymous_24.CUBLAS_GEMM_ALGO23;
    enum CUBLAS_GEMM_DEFAULT_TENSOR_OP = _Anonymous_24.CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    enum CUBLAS_GEMM_DFALT_TENSOR_OP = _Anonymous_24.CUBLAS_GEMM_DFALT_TENSOR_OP;
    enum CUBLAS_GEMM_ALGO0_TENSOR_OP = _Anonymous_24.CUBLAS_GEMM_ALGO0_TENSOR_OP;
    enum CUBLAS_GEMM_ALGO1_TENSOR_OP = _Anonymous_24.CUBLAS_GEMM_ALGO1_TENSOR_OP;
    enum CUBLAS_GEMM_ALGO2_TENSOR_OP = _Anonymous_24.CUBLAS_GEMM_ALGO2_TENSOR_OP;
    enum CUBLAS_GEMM_ALGO3_TENSOR_OP = _Anonymous_24.CUBLAS_GEMM_ALGO3_TENSOR_OP;
    enum CUBLAS_GEMM_ALGO4_TENSOR_OP = _Anonymous_24.CUBLAS_GEMM_ALGO4_TENSOR_OP;
    enum CUBLAS_GEMM_ALGO5_TENSOR_OP = _Anonymous_24.CUBLAS_GEMM_ALGO5_TENSOR_OP;
    enum CUBLAS_GEMM_ALGO6_TENSOR_OP = _Anonymous_24.CUBLAS_GEMM_ALGO6_TENSOR_OP;
    enum CUBLAS_GEMM_ALGO7_TENSOR_OP = _Anonymous_24.CUBLAS_GEMM_ALGO7_TENSOR_OP;
    enum CUBLAS_GEMM_ALGO8_TENSOR_OP = _Anonymous_24.CUBLAS_GEMM_ALGO8_TENSOR_OP;
    enum CUBLAS_GEMM_ALGO9_TENSOR_OP = _Anonymous_24.CUBLAS_GEMM_ALGO9_TENSOR_OP;
    enum CUBLAS_GEMM_ALGO10_TENSOR_OP = _Anonymous_24.CUBLAS_GEMM_ALGO10_TENSOR_OP;
    enum CUBLAS_GEMM_ALGO11_TENSOR_OP = _Anonymous_24.CUBLAS_GEMM_ALGO11_TENSOR_OP;
    enum CUBLAS_GEMM_ALGO12_TENSOR_OP = _Anonymous_24.CUBLAS_GEMM_ALGO12_TENSOR_OP;
    enum CUBLAS_GEMM_ALGO13_TENSOR_OP = _Anonymous_24.CUBLAS_GEMM_ALGO13_TENSOR_OP;
    enum CUBLAS_GEMM_ALGO14_TENSOR_OP = _Anonymous_24.CUBLAS_GEMM_ALGO14_TENSOR_OP;
    enum CUBLAS_GEMM_ALGO15_TENSOR_OP = _Anonymous_24.CUBLAS_GEMM_ALGO15_TENSOR_OP;
    alias cublasMath_t = _Anonymous_25;
    enum _Anonymous_25
    {
        CUBLAS_DEFAULT_MATH = 0,
        CUBLAS_TENSOR_OP_MATH = 1,
    }
    enum CUBLAS_DEFAULT_MATH = _Anonymous_25.CUBLAS_DEFAULT_MATH;
    enum CUBLAS_TENSOR_OP_MATH = _Anonymous_25.CUBLAS_TENSOR_OP_MATH;
    alias cublasDataType_t = cudaDataType_t;
    struct cublasContext;
    alias cublasHandle_t = cublasContext*;
    cublasStatus_t cublasCreate_v2(cublasContext**) @nogc nothrow;
    cublasStatus_t cublasDestroy_v2(cublasContext*) @nogc nothrow;
    cublasStatus_t cublasGetVersion_v2(cublasContext*, int*) @nogc nothrow;
    cublasStatus_t cublasGetProperty(libraryPropertyType_t, int*) @nogc nothrow;
    c_ulong cublasGetCudartVersion() @nogc nothrow;
    cublasStatus_t cublasSetStream_v2(cublasContext*, CUstream_st*) @nogc nothrow;
    cublasStatus_t cublasGetStream_v2(cublasContext*, CUstream_st**) @nogc nothrow;
    cublasStatus_t cublasGetPointerMode_v2(cublasContext*, cublasPointerMode_t*) @nogc nothrow;
    cublasStatus_t cublasSetPointerMode_v2(cublasContext*, cublasPointerMode_t) @nogc nothrow;
    cublasStatus_t cublasGetAtomicsMode(cublasContext*, cublasAtomicsMode_t*) @nogc nothrow;
    cublasStatus_t cublasSetAtomicsMode(cublasContext*, cublasAtomicsMode_t) @nogc nothrow;
    cublasStatus_t cublasGetMathMode(cublasContext*, cublasMath_t*) @nogc nothrow;
    cublasStatus_t cublasSetMathMode(cublasContext*, cublasMath_t) @nogc nothrow;
    alias cublasLogCallback = void function(const(char)*);
    cublasStatus_t cublasLoggerConfigure(int, int, int, const(char)*) @nogc nothrow;
    cublasStatus_t cublasSetLoggerCallback(void function(const(char)*)) @nogc nothrow;
    cublasStatus_t cublasGetLoggerCallback(void function(const(char)*)*) @nogc nothrow;
    cublasStatus_t cublasSetVector(int, int, const(void)*, int, void*, int) @nogc nothrow;
    cublasStatus_t cublasGetVector(int, int, const(void)*, int, void*, int) @nogc nothrow;
    cublasStatus_t cublasSetMatrix(int, int, int, const(void)*, int, void*, int) @nogc nothrow;
    cublasStatus_t cublasGetMatrix(int, int, int, const(void)*, int, void*, int) @nogc nothrow;
    cublasStatus_t cublasSetVectorAsync(int, int, const(void)*, int, void*, int, CUstream_st*) @nogc nothrow;
    cublasStatus_t cublasGetVectorAsync(int, int, const(void)*, int, void*, int, CUstream_st*) @nogc nothrow;
    cublasStatus_t cublasSetMatrixAsync(int, int, int, const(void)*, int, void*, int, CUstream_st*) @nogc nothrow;
    cublasStatus_t cublasGetMatrixAsync(int, int, int, const(void)*, int, void*, int, CUstream_st*) @nogc nothrow;
    void cublasXerbla(const(char)*, int) @nogc nothrow;
    cublasStatus_t cublasNrm2Ex(cublasContext*, int, const(void)*, cudaDataType_t, int, void*, cudaDataType_t, cudaDataType_t) @nogc nothrow;
    cublasStatus_t cublasSnrm2_v2(cublasContext*, int, const(float)*, int, float*) @nogc nothrow;
    cublasStatus_t cublasDnrm2_v2(cublasContext*, int, const(double)*, int, double*) @nogc nothrow;
    cublasStatus_t cublasScnrm2_v2(cublasContext*, int, const(float2)*, int, float*) @nogc nothrow;
    cublasStatus_t cublasDznrm2_v2(cublasContext*, int, const(double2)*, int, double*) @nogc nothrow;
    cublasStatus_t cublasDotEx(cublasContext*, int, const(void)*, cudaDataType_t, int, const(void)*, cudaDataType_t, int, void*, cudaDataType_t, cudaDataType_t) @nogc nothrow;
    cublasStatus_t cublasDotcEx(cublasContext*, int, const(void)*, cudaDataType_t, int, const(void)*, cudaDataType_t, int, void*, cudaDataType_t, cudaDataType_t) @nogc nothrow;
    cublasStatus_t cublasSdot_v2(cublasContext*, int, const(float)*, int, const(float)*, int, float*) @nogc nothrow;
    cublasStatus_t cublasDdot_v2(cublasContext*, int, const(double)*, int, const(double)*, int, double*) @nogc nothrow;
    cublasStatus_t cublasCdotu_v2(cublasContext*, int, const(float2)*, int, const(float2)*, int, float2*) @nogc nothrow;
    cublasStatus_t cublasCdotc_v2(cublasContext*, int, const(float2)*, int, const(float2)*, int, float2*) @nogc nothrow;
    cublasStatus_t cublasZdotu_v2(cublasContext*, int, const(double2)*, int, const(double2)*, int, double2*) @nogc nothrow;
    cublasStatus_t cublasZdotc_v2(cublasContext*, int, const(double2)*, int, const(double2)*, int, double2*) @nogc nothrow;
    cublasStatus_t cublasScalEx(cublasContext*, int, const(void)*, cudaDataType_t, void*, cudaDataType_t, int, cudaDataType_t) @nogc nothrow;
    cublasStatus_t cublasSscal_v2(cublasContext*, int, const(float)*, float*, int) @nogc nothrow;
    cublasStatus_t cublasDscal_v2(cublasContext*, int, const(double)*, double*, int) @nogc nothrow;
    cublasStatus_t cublasCscal_v2(cublasContext*, int, const(float2)*, float2*, int) @nogc nothrow;
    cublasStatus_t cublasCsscal_v2(cublasContext*, int, const(float)*, float2*, int) @nogc nothrow;
    cublasStatus_t cublasZscal_v2(cublasContext*, int, const(double2)*, double2*, int) @nogc nothrow;
    cublasStatus_t cublasZdscal_v2(cublasContext*, int, const(double)*, double2*, int) @nogc nothrow;
    cublasStatus_t cublasAxpyEx(cublasContext*, int, const(void)*, cudaDataType_t, const(void)*, cudaDataType_t, int, void*, cudaDataType_t, int, cudaDataType_t) @nogc nothrow;
    cublasStatus_t cublasSaxpy_v2(cublasContext*, int, const(float)*, const(float)*, int, float*, int) @nogc nothrow;
    cublasStatus_t cublasDaxpy_v2(cublasContext*, int, const(double)*, const(double)*, int, double*, int) @nogc nothrow;
    cublasStatus_t cublasCaxpy_v2(cublasContext*, int, const(float2)*, const(float2)*, int, float2*, int) @nogc nothrow;
    cublasStatus_t cublasZaxpy_v2(cublasContext*, int, const(double2)*, const(double2)*, int, double2*, int) @nogc nothrow;
    cublasStatus_t cublasCopyEx(cublasContext*, int, const(void)*, cudaDataType_t, int, void*, cudaDataType_t, int) @nogc nothrow;
    cublasStatus_t cublasScopy_v2(cublasContext*, int, const(float)*, int, float*, int) @nogc nothrow;
    cublasStatus_t cublasDcopy_v2(cublasContext*, int, const(double)*, int, double*, int) @nogc nothrow;
    cublasStatus_t cublasCcopy_v2(cublasContext*, int, const(float2)*, int, float2*, int) @nogc nothrow;
    cublasStatus_t cublasZcopy_v2(cublasContext*, int, const(double2)*, int, double2*, int) @nogc nothrow;
    cublasStatus_t cublasSswap_v2(cublasContext*, int, float*, int, float*, int) @nogc nothrow;
    cublasStatus_t cublasDswap_v2(cublasContext*, int, double*, int, double*, int) @nogc nothrow;
    cublasStatus_t cublasCswap_v2(cublasContext*, int, float2*, int, float2*, int) @nogc nothrow;
    cublasStatus_t cublasZswap_v2(cublasContext*, int, double2*, int, double2*, int) @nogc nothrow;
    cublasStatus_t cublasSwapEx(cublasContext*, int, void*, cudaDataType_t, int, void*, cudaDataType_t, int) @nogc nothrow;
    cublasStatus_t cublasIsamax_v2(cublasContext*, int, const(float)*, int, int*) @nogc nothrow;
    cublasStatus_t cublasIdamax_v2(cublasContext*, int, const(double)*, int, int*) @nogc nothrow;
    cublasStatus_t cublasIcamax_v2(cublasContext*, int, const(float2)*, int, int*) @nogc nothrow;
    cublasStatus_t cublasIzamax_v2(cublasContext*, int, const(double2)*, int, int*) @nogc nothrow;
    cublasStatus_t cublasIamaxEx(cublasContext*, int, const(void)*, cudaDataType_t, int, int*) @nogc nothrow;
    cublasStatus_t cublasIsamin_v2(cublasContext*, int, const(float)*, int, int*) @nogc nothrow;
    cublasStatus_t cublasIdamin_v2(cublasContext*, int, const(double)*, int, int*) @nogc nothrow;
    cublasStatus_t cublasIcamin_v2(cublasContext*, int, const(float2)*, int, int*) @nogc nothrow;
    cublasStatus_t cublasIzamin_v2(cublasContext*, int, const(double2)*, int, int*) @nogc nothrow;
    cublasStatus_t cublasIaminEx(cublasContext*, int, const(void)*, cudaDataType_t, int, int*) @nogc nothrow;
    cublasStatus_t cublasAsumEx(cublasContext*, int, const(void)*, cudaDataType_t, int, void*, cudaDataType_t, cudaDataType_t) @nogc nothrow;
    cublasStatus_t cublasSasum_v2(cublasContext*, int, const(float)*, int, float*) @nogc nothrow;
    cublasStatus_t cublasDasum_v2(cublasContext*, int, const(double)*, int, double*) @nogc nothrow;
    cublasStatus_t cublasScasum_v2(cublasContext*, int, const(float2)*, int, float*) @nogc nothrow;
    cublasStatus_t cublasDzasum_v2(cublasContext*, int, const(double2)*, int, double*) @nogc nothrow;
    cublasStatus_t cublasSrot_v2(cublasContext*, int, float*, int, float*, int, const(float)*, const(float)*) @nogc nothrow;
    cublasStatus_t cublasDrot_v2(cublasContext*, int, double*, int, double*, int, const(double)*, const(double)*) @nogc nothrow;
    cublasStatus_t cublasCrot_v2(cublasContext*, int, float2*, int, float2*, int, const(float)*, const(float2)*) @nogc nothrow;
    cublasStatus_t cublasCsrot_v2(cublasContext*, int, float2*, int, float2*, int, const(float)*, const(float)*) @nogc nothrow;
    cublasStatus_t cublasZrot_v2(cublasContext*, int, double2*, int, double2*, int, const(double)*, const(double2)*) @nogc nothrow;
    cublasStatus_t cublasZdrot_v2(cublasContext*, int, double2*, int, double2*, int, const(double)*, const(double)*) @nogc nothrow;
    cublasStatus_t cublasRotEx(cublasContext*, int, void*, cudaDataType_t, int, void*, cudaDataType_t, int, const(void)*, const(void)*, cudaDataType_t, cudaDataType_t) @nogc nothrow;
    cublasStatus_t cublasSrotg_v2(cublasContext*, float*, float*, float*, float*) @nogc nothrow;
    cublasStatus_t cublasDrotg_v2(cublasContext*, double*, double*, double*, double*) @nogc nothrow;
    cublasStatus_t cublasCrotg_v2(cublasContext*, float2*, float2*, float*, float2*) @nogc nothrow;
    cublasStatus_t cublasZrotg_v2(cublasContext*, double2*, double2*, double*, double2*) @nogc nothrow;
    cublasStatus_t cublasRotgEx(cublasContext*, void*, void*, cudaDataType_t, void*, void*, cudaDataType_t, cudaDataType_t) @nogc nothrow;
    cublasStatus_t cublasSrotm_v2(cublasContext*, int, float*, int, float*, int, const(float)*) @nogc nothrow;
    cublasStatus_t cublasDrotm_v2(cublasContext*, int, double*, int, double*, int, const(double)*) @nogc nothrow;
    cublasStatus_t cublasRotmEx(cublasContext*, int, void*, cudaDataType_t, int, void*, cudaDataType_t, int, const(void)*, cudaDataType_t, cudaDataType_t) @nogc nothrow;
    cublasStatus_t cublasSrotmg_v2(cublasContext*, float*, float*, float*, const(float)*, float*) @nogc nothrow;
    cublasStatus_t cublasDrotmg_v2(cublasContext*, double*, double*, double*, const(double)*, double*) @nogc nothrow;
    cublasStatus_t cublasRotmgEx(cublasContext*, void*, cudaDataType_t, void*, cudaDataType_t, void*, cudaDataType_t, const(void)*, cudaDataType_t, void*, cudaDataType_t, cudaDataType_t) @nogc nothrow;
    cublasStatus_t cublasSgemv_v2(cublasContext*, cublasOperation_t, int, int, const(float)*, const(float)*, int, const(float)*, int, const(float)*, float*, int) @nogc nothrow;
    cublasStatus_t cublasDgemv_v2(cublasContext*, cublasOperation_t, int, int, const(double)*, const(double)*, int, const(double)*, int, const(double)*, double*, int) @nogc nothrow;
    cublasStatus_t cublasCgemv_v2(cublasContext*, cublasOperation_t, int, int, const(float2)*, const(float2)*, int, const(float2)*, int, const(float2)*, float2*, int) @nogc nothrow;
    cublasStatus_t cublasZgemv_v2(cublasContext*, cublasOperation_t, int, int, const(double2)*, const(double2)*, int, const(double2)*, int, const(double2)*, double2*, int) @nogc nothrow;
    cublasStatus_t cublasSgbmv_v2(cublasContext*, cublasOperation_t, int, int, int, int, const(float)*, const(float)*, int, const(float)*, int, const(float)*, float*, int) @nogc nothrow;
    cublasStatus_t cublasDgbmv_v2(cublasContext*, cublasOperation_t, int, int, int, int, const(double)*, const(double)*, int, const(double)*, int, const(double)*, double*, int) @nogc nothrow;
    cublasStatus_t cublasCgbmv_v2(cublasContext*, cublasOperation_t, int, int, int, int, const(float2)*, const(float2)*, int, const(float2)*, int, const(float2)*, float2*, int) @nogc nothrow;
    cublasStatus_t cublasZgbmv_v2(cublasContext*, cublasOperation_t, int, int, int, int, const(double2)*, const(double2)*, int, const(double2)*, int, const(double2)*, double2*, int) @nogc nothrow;
    cublasStatus_t cublasStrmv_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const(float)*, int, float*, int) @nogc nothrow;
    cublasStatus_t cublasDtrmv_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const(double)*, int, double*, int) @nogc nothrow;
    cublasStatus_t cublasCtrmv_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const(float2)*, int, float2*, int) @nogc nothrow;
    cublasStatus_t cublasZtrmv_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const(double2)*, int, double2*, int) @nogc nothrow;
    cublasStatus_t cublasStbmv_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const(float)*, int, float*, int) @nogc nothrow;
    cublasStatus_t cublasDtbmv_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const(double)*, int, double*, int) @nogc nothrow;
    cublasStatus_t cublasCtbmv_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const(float2)*, int, float2*, int) @nogc nothrow;
    cublasStatus_t cublasZtbmv_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const(double2)*, int, double2*, int) @nogc nothrow;
    cublasStatus_t cublasStpmv_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const(float)*, float*, int) @nogc nothrow;
    cublasStatus_t cublasDtpmv_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const(double)*, double*, int) @nogc nothrow;
    cublasStatus_t cublasCtpmv_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const(float2)*, float2*, int) @nogc nothrow;
    cublasStatus_t cublasZtpmv_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const(double2)*, double2*, int) @nogc nothrow;
    cublasStatus_t cublasStrsv_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const(float)*, int, float*, int) @nogc nothrow;
    cublasStatus_t cublasDtrsv_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const(double)*, int, double*, int) @nogc nothrow;
    cublasStatus_t cublasCtrsv_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const(float2)*, int, float2*, int) @nogc nothrow;
    cublasStatus_t cublasZtrsv_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const(double2)*, int, double2*, int) @nogc nothrow;
    cublasStatus_t cublasStpsv_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const(float)*, float*, int) @nogc nothrow;
    cublasStatus_t cublasDtpsv_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const(double)*, double*, int) @nogc nothrow;
    cublasStatus_t cublasCtpsv_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const(float2)*, float2*, int) @nogc nothrow;
    cublasStatus_t cublasZtpsv_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, const(double2)*, double2*, int) @nogc nothrow;
    cublasStatus_t cublasStbsv_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const(float)*, int, float*, int) @nogc nothrow;
    cublasStatus_t cublasDtbsv_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const(double)*, int, double*, int) @nogc nothrow;
    cublasStatus_t cublasCtbsv_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const(float2)*, int, float2*, int) @nogc nothrow;
    cublasStatus_t cublasZtbsv_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const(double2)*, int, double2*, int) @nogc nothrow;
    cublasStatus_t cublasSsymv_v2(cublasContext*, cublasFillMode_t, int, const(float)*, const(float)*, int, const(float)*, int, const(float)*, float*, int) @nogc nothrow;
    cublasStatus_t cublasDsymv_v2(cublasContext*, cublasFillMode_t, int, const(double)*, const(double)*, int, const(double)*, int, const(double)*, double*, int) @nogc nothrow;
    cublasStatus_t cublasCsymv_v2(cublasContext*, cublasFillMode_t, int, const(float2)*, const(float2)*, int, const(float2)*, int, const(float2)*, float2*, int) @nogc nothrow;
    cublasStatus_t cublasZsymv_v2(cublasContext*, cublasFillMode_t, int, const(double2)*, const(double2)*, int, const(double2)*, int, const(double2)*, double2*, int) @nogc nothrow;
    cublasStatus_t cublasChemv_v2(cublasContext*, cublasFillMode_t, int, const(float2)*, const(float2)*, int, const(float2)*, int, const(float2)*, float2*, int) @nogc nothrow;
    cublasStatus_t cublasZhemv_v2(cublasContext*, cublasFillMode_t, int, const(double2)*, const(double2)*, int, const(double2)*, int, const(double2)*, double2*, int) @nogc nothrow;
    cublasStatus_t cublasSsbmv_v2(cublasContext*, cublasFillMode_t, int, int, const(float)*, const(float)*, int, const(float)*, int, const(float)*, float*, int) @nogc nothrow;
    cublasStatus_t cublasDsbmv_v2(cublasContext*, cublasFillMode_t, int, int, const(double)*, const(double)*, int, const(double)*, int, const(double)*, double*, int) @nogc nothrow;
    cublasStatus_t cublasChbmv_v2(cublasContext*, cublasFillMode_t, int, int, const(float2)*, const(float2)*, int, const(float2)*, int, const(float2)*, float2*, int) @nogc nothrow;
    cublasStatus_t cublasZhbmv_v2(cublasContext*, cublasFillMode_t, int, int, const(double2)*, const(double2)*, int, const(double2)*, int, const(double2)*, double2*, int) @nogc nothrow;
    cublasStatus_t cublasSspmv_v2(cublasContext*, cublasFillMode_t, int, const(float)*, const(float)*, const(float)*, int, const(float)*, float*, int) @nogc nothrow;
    cublasStatus_t cublasDspmv_v2(cublasContext*, cublasFillMode_t, int, const(double)*, const(double)*, const(double)*, int, const(double)*, double*, int) @nogc nothrow;
    cublasStatus_t cublasChpmv_v2(cublasContext*, cublasFillMode_t, int, const(float2)*, const(float2)*, const(float2)*, int, const(float2)*, float2*, int) @nogc nothrow;
    cublasStatus_t cublasZhpmv_v2(cublasContext*, cublasFillMode_t, int, const(double2)*, const(double2)*, const(double2)*, int, const(double2)*, double2*, int) @nogc nothrow;
    cublasStatus_t cublasSger_v2(cublasContext*, int, int, const(float)*, const(float)*, int, const(float)*, int, float*, int) @nogc nothrow;
    cublasStatus_t cublasDger_v2(cublasContext*, int, int, const(double)*, const(double)*, int, const(double)*, int, double*, int) @nogc nothrow;
    cublasStatus_t cublasCgeru_v2(cublasContext*, int, int, const(float2)*, const(float2)*, int, const(float2)*, int, float2*, int) @nogc nothrow;
    cublasStatus_t cublasCgerc_v2(cublasContext*, int, int, const(float2)*, const(float2)*, int, const(float2)*, int, float2*, int) @nogc nothrow;
    cublasStatus_t cublasZgeru_v2(cublasContext*, int, int, const(double2)*, const(double2)*, int, const(double2)*, int, double2*, int) @nogc nothrow;
    cublasStatus_t cublasZgerc_v2(cublasContext*, int, int, const(double2)*, const(double2)*, int, const(double2)*, int, double2*, int) @nogc nothrow;
    cublasStatus_t cublasSsyr_v2(cublasContext*, cublasFillMode_t, int, const(float)*, const(float)*, int, float*, int) @nogc nothrow;
    cublasStatus_t cublasDsyr_v2(cublasContext*, cublasFillMode_t, int, const(double)*, const(double)*, int, double*, int) @nogc nothrow;
    cublasStatus_t cublasCsyr_v2(cublasContext*, cublasFillMode_t, int, const(float2)*, const(float2)*, int, float2*, int) @nogc nothrow;
    cublasStatus_t cublasZsyr_v2(cublasContext*, cublasFillMode_t, int, const(double2)*, const(double2)*, int, double2*, int) @nogc nothrow;
    cublasStatus_t cublasCher_v2(cublasContext*, cublasFillMode_t, int, const(float)*, const(float2)*, int, float2*, int) @nogc nothrow;
    cublasStatus_t cublasZher_v2(cublasContext*, cublasFillMode_t, int, const(double)*, const(double2)*, int, double2*, int) @nogc nothrow;
    cublasStatus_t cublasSspr_v2(cublasContext*, cublasFillMode_t, int, const(float)*, const(float)*, int, float*) @nogc nothrow;
    cublasStatus_t cublasDspr_v2(cublasContext*, cublasFillMode_t, int, const(double)*, const(double)*, int, double*) @nogc nothrow;
    cublasStatus_t cublasChpr_v2(cublasContext*, cublasFillMode_t, int, const(float)*, const(float2)*, int, float2*) @nogc nothrow;
    cublasStatus_t cublasZhpr_v2(cublasContext*, cublasFillMode_t, int, const(double)*, const(double2)*, int, double2*) @nogc nothrow;
    cublasStatus_t cublasSsyr2_v2(cublasContext*, cublasFillMode_t, int, const(float)*, const(float)*, int, const(float)*, int, float*, int) @nogc nothrow;
    cublasStatus_t cublasDsyr2_v2(cublasContext*, cublasFillMode_t, int, const(double)*, const(double)*, int, const(double)*, int, double*, int) @nogc nothrow;
    cublasStatus_t cublasCsyr2_v2(cublasContext*, cublasFillMode_t, int, const(float2)*, const(float2)*, int, const(float2)*, int, float2*, int) @nogc nothrow;
    cublasStatus_t cublasZsyr2_v2(cublasContext*, cublasFillMode_t, int, const(double2)*, const(double2)*, int, const(double2)*, int, double2*, int) @nogc nothrow;
    cublasStatus_t cublasCher2_v2(cublasContext*, cublasFillMode_t, int, const(float2)*, const(float2)*, int, const(float2)*, int, float2*, int) @nogc nothrow;
    cublasStatus_t cublasZher2_v2(cublasContext*, cublasFillMode_t, int, const(double2)*, const(double2)*, int, const(double2)*, int, double2*, int) @nogc nothrow;
    cublasStatus_t cublasSspr2_v2(cublasContext*, cublasFillMode_t, int, const(float)*, const(float)*, int, const(float)*, int, float*) @nogc nothrow;
    cublasStatus_t cublasDspr2_v2(cublasContext*, cublasFillMode_t, int, const(double)*, const(double)*, int, const(double)*, int, double*) @nogc nothrow;
    cublasStatus_t cublasChpr2_v2(cublasContext*, cublasFillMode_t, int, const(float2)*, const(float2)*, int, const(float2)*, int, float2*) @nogc nothrow;
    cublasStatus_t cublasZhpr2_v2(cublasContext*, cublasFillMode_t, int, const(double2)*, const(double2)*, int, const(double2)*, int, double2*) @nogc nothrow;
    cublasStatus_t cublasSgemm_v2(cublasContext*, cublasOperation_t, cublasOperation_t, int, int, int, const(float)*, const(float)*, int, const(float)*, int, const(float)*, float*, int) @nogc nothrow;
    cublasStatus_t cublasDgemm_v2(cublasContext*, cublasOperation_t, cublasOperation_t, int, int, int, const(double)*, const(double)*, int, const(double)*, int, const(double)*, double*, int) @nogc nothrow;
    cublasStatus_t cublasCgemm_v2(cublasContext*, cublasOperation_t, cublasOperation_t, int, int, int, const(float2)*, const(float2)*, int, const(float2)*, int, const(float2)*, float2*, int) @nogc nothrow;
    cublasStatus_t cublasCgemm3m(cublasContext*, cublasOperation_t, cublasOperation_t, int, int, int, const(float2)*, const(float2)*, int, const(float2)*, int, const(float2)*, float2*, int) @nogc nothrow;
    cublasStatus_t cublasCgemm3mEx(cublasContext*, cublasOperation_t, cublasOperation_t, int, int, int, const(float2)*, const(void)*, cudaDataType_t, int, const(void)*, cudaDataType_t, int, const(float2)*, void*, cudaDataType_t, int) @nogc nothrow;
    cublasStatus_t cublasZgemm_v2(cublasContext*, cublasOperation_t, cublasOperation_t, int, int, int, const(double2)*, const(double2)*, int, const(double2)*, int, const(double2)*, double2*, int) @nogc nothrow;
    cublasStatus_t cublasZgemm3m(cublasContext*, cublasOperation_t, cublasOperation_t, int, int, int, const(double2)*, const(double2)*, int, const(double2)*, int, const(double2)*, double2*, int) @nogc nothrow;
    cublasStatus_t cublasSgemmEx(cublasContext*, cublasOperation_t, cublasOperation_t, int, int, int, const(float)*, const(void)*, cudaDataType_t, int, const(void)*, cudaDataType_t, int, const(float)*, void*, cudaDataType_t, int) @nogc nothrow;
    cublasStatus_t cublasGemmEx(cublasContext*, cublasOperation_t, cublasOperation_t, int, int, int, const(void)*, const(void)*, cudaDataType_t, int, const(void)*, cudaDataType_t, int, const(void)*, void*, cudaDataType_t, int, cudaDataType_t, cublasGemmAlgo_t) @nogc nothrow;
    cublasStatus_t cublasCgemmEx(cublasContext*, cublasOperation_t, cublasOperation_t, int, int, int, const(float2)*, const(void)*, cudaDataType_t, int, const(void)*, cudaDataType_t, int, const(float2)*, void*, cudaDataType_t, int) @nogc nothrow;
    cublasStatus_t cublasUint8gemmBias(cublasContext*, cublasOperation_t, cublasOperation_t, cublasOperation_t, int, int, int, const(ubyte)*, int, int, const(ubyte)*, int, int, ubyte*, int, int, int, int) @nogc nothrow;
    cublasStatus_t cublasSsyrk_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, int, int, const(float)*, const(float)*, int, const(float)*, float*, int) @nogc nothrow;
    cublasStatus_t cublasDsyrk_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, int, int, const(double)*, const(double)*, int, const(double)*, double*, int) @nogc nothrow;
    cublasStatus_t cublasCsyrk_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, int, int, const(float2)*, const(float2)*, int, const(float2)*, float2*, int) @nogc nothrow;
    cublasStatus_t cublasZsyrk_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, int, int, const(double2)*, const(double2)*, int, const(double2)*, double2*, int) @nogc nothrow;
    cublasStatus_t cublasCsyrkEx(cublasContext*, cublasFillMode_t, cublasOperation_t, int, int, const(float2)*, const(void)*, cudaDataType_t, int, const(float2)*, void*, cudaDataType_t, int) @nogc nothrow;
    cublasStatus_t cublasCsyrk3mEx(cublasContext*, cublasFillMode_t, cublasOperation_t, int, int, const(float2)*, const(void)*, cudaDataType_t, int, const(float2)*, void*, cudaDataType_t, int) @nogc nothrow;
    cublasStatus_t cublasCherk_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, int, int, const(float)*, const(float2)*, int, const(float)*, float2*, int) @nogc nothrow;
    cublasStatus_t cublasZherk_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, int, int, const(double)*, const(double2)*, int, const(double)*, double2*, int) @nogc nothrow;
    cublasStatus_t cublasCherkEx(cublasContext*, cublasFillMode_t, cublasOperation_t, int, int, const(float)*, const(void)*, cudaDataType_t, int, const(float)*, void*, cudaDataType_t, int) @nogc nothrow;
    cublasStatus_t cublasCherk3mEx(cublasContext*, cublasFillMode_t, cublasOperation_t, int, int, const(float)*, const(void)*, cudaDataType_t, int, const(float)*, void*, cudaDataType_t, int) @nogc nothrow;
    cublasStatus_t cublasSsyr2k_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, int, int, const(float)*, const(float)*, int, const(float)*, int, const(float)*, float*, int) @nogc nothrow;
    cublasStatus_t cublasDsyr2k_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, int, int, const(double)*, const(double)*, int, const(double)*, int, const(double)*, double*, int) @nogc nothrow;
    cublasStatus_t cublasCsyr2k_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, int, int, const(float2)*, const(float2)*, int, const(float2)*, int, const(float2)*, float2*, int) @nogc nothrow;
    cublasStatus_t cublasZsyr2k_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, int, int, const(double2)*, const(double2)*, int, const(double2)*, int, const(double2)*, double2*, int) @nogc nothrow;
    cublasStatus_t cublasCher2k_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, int, int, const(float2)*, const(float2)*, int, const(float2)*, int, const(float)*, float2*, int) @nogc nothrow;
    cublasStatus_t cublasZher2k_v2(cublasContext*, cublasFillMode_t, cublasOperation_t, int, int, const(double2)*, const(double2)*, int, const(double2)*, int, const(double)*, double2*, int) @nogc nothrow;
    cublasStatus_t cublasSsyrkx(cublasContext*, cublasFillMode_t, cublasOperation_t, int, int, const(float)*, const(float)*, int, const(float)*, int, const(float)*, float*, int) @nogc nothrow;
    cublasStatus_t cublasDsyrkx(cublasContext*, cublasFillMode_t, cublasOperation_t, int, int, const(double)*, const(double)*, int, const(double)*, int, const(double)*, double*, int) @nogc nothrow;
    cublasStatus_t cublasCsyrkx(cublasContext*, cublasFillMode_t, cublasOperation_t, int, int, const(float2)*, const(float2)*, int, const(float2)*, int, const(float2)*, float2*, int) @nogc nothrow;
    cublasStatus_t cublasZsyrkx(cublasContext*, cublasFillMode_t, cublasOperation_t, int, int, const(double2)*, const(double2)*, int, const(double2)*, int, const(double2)*, double2*, int) @nogc nothrow;
    cublasStatus_t cublasCherkx(cublasContext*, cublasFillMode_t, cublasOperation_t, int, int, const(float2)*, const(float2)*, int, const(float2)*, int, const(float)*, float2*, int) @nogc nothrow;
    cublasStatus_t cublasZherkx(cublasContext*, cublasFillMode_t, cublasOperation_t, int, int, const(double2)*, const(double2)*, int, const(double2)*, int, const(double)*, double2*, int) @nogc nothrow;
    cublasStatus_t cublasSsymm_v2(cublasContext*, cublasSideMode_t, cublasFillMode_t, int, int, const(float)*, const(float)*, int, const(float)*, int, const(float)*, float*, int) @nogc nothrow;
    cublasStatus_t cublasDsymm_v2(cublasContext*, cublasSideMode_t, cublasFillMode_t, int, int, const(double)*, const(double)*, int, const(double)*, int, const(double)*, double*, int) @nogc nothrow;
    cublasStatus_t cublasCsymm_v2(cublasContext*, cublasSideMode_t, cublasFillMode_t, int, int, const(float2)*, const(float2)*, int, const(float2)*, int, const(float2)*, float2*, int) @nogc nothrow;
    cublasStatus_t cublasZsymm_v2(cublasContext*, cublasSideMode_t, cublasFillMode_t, int, int, const(double2)*, const(double2)*, int, const(double2)*, int, const(double2)*, double2*, int) @nogc nothrow;
    cublasStatus_t cublasChemm_v2(cublasContext*, cublasSideMode_t, cublasFillMode_t, int, int, const(float2)*, const(float2)*, int, const(float2)*, int, const(float2)*, float2*, int) @nogc nothrow;
    cublasStatus_t cublasZhemm_v2(cublasContext*, cublasSideMode_t, cublasFillMode_t, int, int, const(double2)*, const(double2)*, int, const(double2)*, int, const(double2)*, double2*, int) @nogc nothrow;
    cublasStatus_t cublasStrsm_v2(cublasContext*, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const(float)*, const(float)*, int, float*, int) @nogc nothrow;
    cublasStatus_t cublasDtrsm_v2(cublasContext*, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const(double)*, const(double)*, int, double*, int) @nogc nothrow;
    cublasStatus_t cublasCtrsm_v2(cublasContext*, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const(float2)*, const(float2)*, int, float2*, int) @nogc nothrow;
    cublasStatus_t cublasZtrsm_v2(cublasContext*, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const(double2)*, const(double2)*, int, double2*, int) @nogc nothrow;
    cublasStatus_t cublasStrmm_v2(cublasContext*, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const(float)*, const(float)*, int, const(float)*, int, float*, int) @nogc nothrow;
    cublasStatus_t cublasDtrmm_v2(cublasContext*, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const(double)*, const(double)*, int, const(double)*, int, double*, int) @nogc nothrow;
    cublasStatus_t cublasCtrmm_v2(cublasContext*, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const(float2)*, const(float2)*, int, const(float2)*, int, float2*, int) @nogc nothrow;
    cublasStatus_t cublasZtrmm_v2(cublasContext*, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const(double2)*, const(double2)*, int, const(double2)*, int, double2*, int) @nogc nothrow;
    cublasStatus_t cublasSgemmBatched(cublasContext*, cublasOperation_t, cublasOperation_t, int, int, int, const(float)*, const(const(float)*)*, int, const(const(float)*)*, int, const(float)*, float**, int, int) @nogc nothrow;
    cublasStatus_t cublasDgemmBatched(cublasContext*, cublasOperation_t, cublasOperation_t, int, int, int, const(double)*, const(const(double)*)*, int, const(const(double)*)*, int, const(double)*, double**, int, int) @nogc nothrow;
    cublasStatus_t cublasCgemmBatched(cublasContext*, cublasOperation_t, cublasOperation_t, int, int, int, const(float2)*, const(const(float2)*)*, int, const(const(float2)*)*, int, const(float2)*, float2**, int, int) @nogc nothrow;
    cublasStatus_t cublasCgemm3mBatched(cublasContext*, cublasOperation_t, cublasOperation_t, int, int, int, const(float2)*, const(const(float2)*)*, int, const(const(float2)*)*, int, const(float2)*, float2**, int, int) @nogc nothrow;
    cublasStatus_t cublasZgemmBatched(cublasContext*, cublasOperation_t, cublasOperation_t, int, int, int, const(double2)*, const(const(double2)*)*, int, const(const(double2)*)*, int, const(double2)*, double2**, int, int) @nogc nothrow;
    cublasStatus_t cublasGemmBatchedEx(cublasContext*, cublasOperation_t, cublasOperation_t, int, int, int, const(void)*, const(const(void)*)*, cudaDataType_t, int, const(const(void)*)*, cudaDataType_t, int, const(void)*, void**, cudaDataType_t, int, int, cudaDataType_t, cublasGemmAlgo_t) @nogc nothrow;
    cublasStatus_t cublasGemmStridedBatchedEx(cublasContext*, cublasOperation_t, cublasOperation_t, int, int, int, const(void)*, const(void)*, cudaDataType_t, int, long, const(void)*, cudaDataType_t, int, long, const(void)*, void*, cudaDataType_t, int, long, int, cudaDataType_t, cublasGemmAlgo_t) @nogc nothrow;
    cublasStatus_t cublasSgemmStridedBatched(cublasContext*, cublasOperation_t, cublasOperation_t, int, int, int, const(float)*, const(float)*, int, long, const(float)*, int, long, const(float)*, float*, int, long, int) @nogc nothrow;
    cublasStatus_t cublasDgemmStridedBatched(cublasContext*, cublasOperation_t, cublasOperation_t, int, int, int, const(double)*, const(double)*, int, long, const(double)*, int, long, const(double)*, double*, int, long, int) @nogc nothrow;
    cublasStatus_t cublasCgemmStridedBatched(cublasContext*, cublasOperation_t, cublasOperation_t, int, int, int, const(float2)*, const(float2)*, int, long, const(float2)*, int, long, const(float2)*, float2*, int, long, int) @nogc nothrow;
    cublasStatus_t cublasCgemm3mStridedBatched(cublasContext*, cublasOperation_t, cublasOperation_t, int, int, int, const(float2)*, const(float2)*, int, long, const(float2)*, int, long, const(float2)*, float2*, int, long, int) @nogc nothrow;
    cublasStatus_t cublasZgemmStridedBatched(cublasContext*, cublasOperation_t, cublasOperation_t, int, int, int, const(double2)*, const(double2)*, int, long, const(double2)*, int, long, const(double2)*, double2*, int, long, int) @nogc nothrow;
    cublasStatus_t cublasSgeam(cublasContext*, cublasOperation_t, cublasOperation_t, int, int, const(float)*, const(float)*, int, const(float)*, const(float)*, int, float*, int) @nogc nothrow;
    cublasStatus_t cublasDgeam(cublasContext*, cublasOperation_t, cublasOperation_t, int, int, const(double)*, const(double)*, int, const(double)*, const(double)*, int, double*, int) @nogc nothrow;
    cublasStatus_t cublasCgeam(cublasContext*, cublasOperation_t, cublasOperation_t, int, int, const(float2)*, const(float2)*, int, const(float2)*, const(float2)*, int, float2*, int) @nogc nothrow;
    cublasStatus_t cublasZgeam(cublasContext*, cublasOperation_t, cublasOperation_t, int, int, const(double2)*, const(double2)*, int, const(double2)*, const(double2)*, int, double2*, int) @nogc nothrow;
    cublasStatus_t cublasSgetrfBatched(cublasContext*, int, float**, int, int*, int*, int) @nogc nothrow;
    cublasStatus_t cublasDgetrfBatched(cublasContext*, int, double**, int, int*, int*, int) @nogc nothrow;
    cublasStatus_t cublasCgetrfBatched(cublasContext*, int, float2**, int, int*, int*, int) @nogc nothrow;
    cublasStatus_t cublasZgetrfBatched(cublasContext*, int, double2**, int, int*, int*, int) @nogc nothrow;
    cublasStatus_t cublasSgetriBatched(cublasContext*, int, const(const(float)*)*, int, const(int)*, float**, int, int*, int) @nogc nothrow;
    cublasStatus_t cublasDgetriBatched(cublasContext*, int, const(const(double)*)*, int, const(int)*, double**, int, int*, int) @nogc nothrow;
    cublasStatus_t cublasCgetriBatched(cublasContext*, int, const(const(float2)*)*, int, const(int)*, float2**, int, int*, int) @nogc nothrow;
    cublasStatus_t cublasZgetriBatched(cublasContext*, int, const(const(double2)*)*, int, const(int)*, double2**, int, int*, int) @nogc nothrow;
    cublasStatus_t cublasSgetrsBatched(cublasContext*, cublasOperation_t, int, int, const(const(float)*)*, int, const(int)*, float**, int, int*, int) @nogc nothrow;
    cublasStatus_t cublasDgetrsBatched(cublasContext*, cublasOperation_t, int, int, const(const(double)*)*, int, const(int)*, double**, int, int*, int) @nogc nothrow;
    cublasStatus_t cublasCgetrsBatched(cublasContext*, cublasOperation_t, int, int, const(const(float2)*)*, int, const(int)*, float2**, int, int*, int) @nogc nothrow;
    cublasStatus_t cublasZgetrsBatched(cublasContext*, cublasOperation_t, int, int, const(const(double2)*)*, int, const(int)*, double2**, int, int*, int) @nogc nothrow;
    cublasStatus_t cublasStrsmBatched(cublasContext*, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const(float)*, const(const(float)*)*, int, float**, int, int) @nogc nothrow;
    cublasStatus_t cublasDtrsmBatched(cublasContext*, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const(double)*, const(const(double)*)*, int, double**, int, int) @nogc nothrow;
    cublasStatus_t cublasCtrsmBatched(cublasContext*, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const(float2)*, const(const(float2)*)*, int, float2**, int, int) @nogc nothrow;
    cublasStatus_t cublasZtrsmBatched(cublasContext*, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, int, int, const(double2)*, const(const(double2)*)*, int, double2**, int, int) @nogc nothrow;
    cublasStatus_t cublasSmatinvBatched(cublasContext*, int, const(const(float)*)*, int, float**, int, int*, int) @nogc nothrow;
    cublasStatus_t cublasDmatinvBatched(cublasContext*, int, const(const(double)*)*, int, double**, int, int*, int) @nogc nothrow;
    cublasStatus_t cublasCmatinvBatched(cublasContext*, int, const(const(float2)*)*, int, float2**, int, int*, int) @nogc nothrow;
    cublasStatus_t cublasZmatinvBatched(cublasContext*, int, const(const(double2)*)*, int, double2**, int, int*, int) @nogc nothrow;
    cublasStatus_t cublasSgeqrfBatched(cublasContext*, int, int, float**, int, float**, int*, int) @nogc nothrow;
    cublasStatus_t cublasDgeqrfBatched(cublasContext*, int, int, double**, int, double**, int*, int) @nogc nothrow;
    cublasStatus_t cublasCgeqrfBatched(cublasContext*, int, int, float2**, int, float2**, int*, int) @nogc nothrow;
    cublasStatus_t cublasZgeqrfBatched(cublasContext*, int, int, double2**, int, double2**, int*, int) @nogc nothrow;
    cublasStatus_t cublasSgelsBatched(cublasContext*, cublasOperation_t, int, int, int, float**, int, float**, int, int*, int*, int) @nogc nothrow;
    cublasStatus_t cublasDgelsBatched(cublasContext*, cublasOperation_t, int, int, int, double**, int, double**, int, int*, int*, int) @nogc nothrow;
    cublasStatus_t cublasCgelsBatched(cublasContext*, cublasOperation_t, int, int, int, float2**, int, float2**, int, int*, int*, int) @nogc nothrow;
    cublasStatus_t cublasZgelsBatched(cublasContext*, cublasOperation_t, int, int, int, double2**, int, double2**, int, int*, int*, int) @nogc nothrow;
    cublasStatus_t cublasSdgmm(cublasContext*, cublasSideMode_t, int, int, const(float)*, int, const(float)*, int, float*, int) @nogc nothrow;
    cublasStatus_t cublasDdgmm(cublasContext*, cublasSideMode_t, int, int, const(double)*, int, const(double)*, int, double*, int) @nogc nothrow;
    cublasStatus_t cublasCdgmm(cublasContext*, cublasSideMode_t, int, int, const(float2)*, int, const(float2)*, int, float2*, int) @nogc nothrow;
    cublasStatus_t cublasZdgmm(cublasContext*, cublasSideMode_t, int, int, const(double2)*, int, const(double2)*, int, double2*, int) @nogc nothrow;
    cublasStatus_t cublasStpttr(cublasContext*, cublasFillMode_t, int, const(float)*, float*, int) @nogc nothrow;
    cublasStatus_t cublasDtpttr(cublasContext*, cublasFillMode_t, int, const(double)*, double*, int) @nogc nothrow;
    cublasStatus_t cublasCtpttr(cublasContext*, cublasFillMode_t, int, const(float2)*, float2*, int) @nogc nothrow;
    cublasStatus_t cublasZtpttr(cublasContext*, cublasFillMode_t, int, const(double2)*, double2*, int) @nogc nothrow;
    cublasStatus_t cublasStrttp(cublasContext*, cublasFillMode_t, int, const(float)*, int, float*) @nogc nothrow;
    cublasStatus_t cublasDtrttp(cublasContext*, cublasFillMode_t, int, const(double)*, int, double*) @nogc nothrow;
    cublasStatus_t cublasCtrttp(cublasContext*, cublasFillMode_t, int, const(float2)*, int, float2*) @nogc nothrow;
    cublasStatus_t cublasZtrttp(cublasContext*, cublasFillMode_t, int, const(double2)*, int, double2*) @nogc nothrow;
    static double2 cuCfma(double2, double2, double2) @nogc nothrow;
    static float2 cuCfmaf(float2, float2, float2) @nogc nothrow;
    static float2 cuComplexDoubleToFloat(double2) @nogc nothrow;
    static double2 cuComplexFloatToDouble(float2) @nogc nothrow;
    static float2 make_cuComplex(float, float) @nogc nothrow;
    alias cuComplex = float2;
    static double cuCabs(double2) @nogc nothrow;
    static double2 cuCdiv(double2, double2) @nogc nothrow;
    static double2 cuCmul(double2, double2) @nogc nothrow;
    static double2 cuCsub(double2, double2) @nogc nothrow;
    static double2 cuCadd(double2, double2) @nogc nothrow;
    static double2 cuConj(double2) @nogc nothrow;
    static double2 make_cuDoubleComplex(double, double) @nogc nothrow;
    static double cuCimag(double2) @nogc nothrow;
    static double cuCreal(double2) @nogc nothrow;
    alias cuDoubleComplex = double2;
    static float cuCabsf(float2) @nogc nothrow;
    static float2 cuCdivf(float2, float2) @nogc nothrow;
    static float2 cuCmulf(float2, float2) @nogc nothrow;
    static float2 cuCsubf(float2, float2) @nogc nothrow;
    static float2 cuCaddf(float2, float2) @nogc nothrow;
    static float2 cuConjf(float2) @nogc nothrow;
    static float2 make_cuFloatComplex(float, float) @nogc nothrow;
    static float cuCimagf(float2) @nogc nothrow;
    static float cuCrealf(float2) @nogc nothrow;
    alias cuFloatComplex = float2;
    alias __sig_atomic_t = int;
    alias __socklen_t = uint;
    alias __intptr_t = c_long;
    alias __caddr_t = char*;
    alias __loff_t = c_long;
    alias __syscall_ulong_t = c_ulong;
    alias __syscall_slong_t = c_long;
    alias __ssize_t = c_long;
    alias __fsword_t = c_long;
    alias __fsfilcnt64_t = c_ulong;
    alias __fsfilcnt_t = c_ulong;
    alias __fsblkcnt64_t = c_ulong;
    alias __fsblkcnt_t = c_ulong;
    alias __blkcnt64_t = c_long;
    alias __blkcnt_t = c_long;
    alias __blksize_t = c_long;
    alias __timer_t = void*;
    alias __clockid_t = int;
    alias __key_t = int;
    alias __daddr_t = int;
    alias __suseconds_t = c_long;
    alias float_t = float;
    alias double_t = double;
    alias __useconds_t = uint;
    alias __time_t = c_long;
    alias __id_t = uint;
    alias __rlim64_t = c_ulong;
    alias __rlim_t = c_ulong;
    alias __clock_t = c_long;
    struct __fsid_t
    {
        int[2] __val;
    }
    alias __pid_t = int;
    alias __off64_t = c_long;
    alias __off_t = c_long;
    alias __nlink_t = c_ulong;
    alias __mode_t = uint;
    alias __ino64_t = c_ulong;
    extern __gshared int signgam;
    enum _Anonymous_26
    {
        FP_NAN = 0,
        FP_INFINITE = 1,
        FP_ZERO = 2,
        FP_SUBNORMAL = 3,
        FP_NORMAL = 4,
    }
    enum FP_NAN = _Anonymous_26.FP_NAN;
    enum FP_INFINITE = _Anonymous_26.FP_INFINITE;
    enum FP_ZERO = _Anonymous_26.FP_ZERO;
    enum FP_SUBNORMAL = _Anonymous_26.FP_SUBNORMAL;
    enum FP_NORMAL = _Anonymous_26.FP_NORMAL;
    alias __ino_t = c_ulong;
    alias __gid_t = uint;
    alias __uid_t = uint;
    alias __dev_t = c_ulong;
    alias __uintmax_t = c_ulong;
    alias __intmax_t = c_long;
    alias __u_quad_t = c_ulong;
    alias __quad_t = c_long;
    alias __uint64_t = c_ulong;
    alias __int64_t = c_long;
    alias __uint32_t = uint;
    alias __int32_t = int;
    alias __uint16_t = ushort;
    alias __int16_t = short;
    alias __uint8_t = ubyte;
    alias __int8_t = byte;
    alias __u_long = c_ulong;
    alias __u_int = uint;
    alias __u_short = ushort;
    alias __u_char = ubyte;
    real __scalbl(real, real) @nogc nothrow;
    double scalb(double, double) @nogc nothrow;
    double __scalb(double, double) @nogc nothrow;
    real scalbl(real, real) @nogc nothrow;
    float __scalbf(float, float) @nogc nothrow;
    float scalbf(float, float) @nogc nothrow;
    alias _Float32 = float;
    alias _Float64 = double;
    alias _Float32x = double;
    alias _Float64x = real;
    double fma(double, double, double) @nogc nothrow;
    real fmal(real, real, real) @nogc nothrow;
    double __fma(double, double, double) @nogc nothrow;
    real __fmal(real, real, real) @nogc nothrow;
    float fmaf(float, float, float) @nogc nothrow;
    float __fmaf(float, float, float) @nogc nothrow;
    double fmin(double, double) @nogc nothrow;
    real __fminl(real, real) @nogc nothrow;
    real fminl(real, real) @nogc nothrow;
    double __fmin(double, double) @nogc nothrow;
    float fminf(float, float) @nogc nothrow;
    float __fminf(float, float) @nogc nothrow;
    double fmax(double, double) @nogc nothrow;
    real __fmaxl(real, real) @nogc nothrow;
    real fmaxl(real, real) @nogc nothrow;
    double __fmax(double, double) @nogc nothrow;
    float fmaxf(float, float) @nogc nothrow;
    float __fmaxf(float, float) @nogc nothrow;
    double fdim(double, double) @nogc nothrow;
    float __fdimf(float, float) @nogc nothrow;
    float fdimf(float, float) @nogc nothrow;
    real fdiml(real, real) @nogc nothrow;
    real __fdiml(real, real) @nogc nothrow;
    double __fdim(double, double) @nogc nothrow;
    long llround(double) @nogc nothrow;
    long __llroundf(float) @nogc nothrow;
    long llroundl(real) @nogc nothrow;
    long __llroundl(real) @nogc nothrow;
    long llroundf(float) @nogc nothrow;
    long __llround(double) @nogc nothrow;
    c_long lround(double) @nogc nothrow;
    c_long lroundf(float) @nogc nothrow;
    c_long __lroundf(float) @nogc nothrow;
    c_long __lround(double) @nogc nothrow;
    c_long __lroundl(real) @nogc nothrow;
    c_long lroundl(real) @nogc nothrow;
    long llrint(double) @nogc nothrow;
    long llrintl(real) @nogc nothrow;
    int __fpclassify(double) @nogc nothrow;
    int __fpclassifyf(float) @nogc nothrow;
    int __fpclassifyl(real) @nogc nothrow;
    int __signbitl(real) @nogc nothrow;
    int __signbit(double) @nogc nothrow;
    int __signbitf(float) @nogc nothrow;
    int __isinff(float) @nogc nothrow;
    int __isinf(double) @nogc nothrow;
    int __isinfl(real) @nogc nothrow;
    int __finitef(float) @nogc nothrow;
    int __finitel(real) @nogc nothrow;
    int __finite(double) @nogc nothrow;
    int __isnanl(real) @nogc nothrow;
    int __isnan(double) @nogc nothrow;
    int __isnanf(float) @nogc nothrow;
    int __iseqsigl(real, real) @nogc nothrow;
    int __iseqsig(double, double) @nogc nothrow;
    int __iseqsigf(float, float) @nogc nothrow;
    int __issignalingl(real) @nogc nothrow;
    int __issignalingf(float) @nogc nothrow;
    int __issignaling(double) @nogc nothrow;
    float __acosf(float) @nogc nothrow;
    float acosf(float) @nogc nothrow;
    real __acosl(real) @nogc nothrow;
    double __acos(double) @nogc nothrow;
    real acosl(real) @nogc nothrow;
    double acos(double) @nogc nothrow;
    double __asin(double) @nogc nothrow;
    real __asinl(real) @nogc nothrow;
    real asinl(real) @nogc nothrow;
    float __asinf(float) @nogc nothrow;
    float asinf(float) @nogc nothrow;
    double asin(double) @nogc nothrow;
    float __atanf(float) @nogc nothrow;
    double __atan(double) @nogc nothrow;
    real atanl(real) @nogc nothrow;
    real __atanl(real) @nogc nothrow;
    float atanf(float) @nogc nothrow;
    double atan(double) @nogc nothrow;
    double __atan2(double, double) @nogc nothrow;
    float __atan2f(float, float) @nogc nothrow;
    real atan2l(real, real) @nogc nothrow;
    real __atan2l(real, real) @nogc nothrow;
    float atan2f(float, float) @nogc nothrow;
    double atan2(double, double) @nogc nothrow;
    float __cosf(float) @nogc nothrow;
    real __cosl(real) @nogc nothrow;
    real cosl(real) @nogc nothrow;
    float cosf(float) @nogc nothrow;
    double __cos(double) @nogc nothrow;
    double cos(double) @nogc nothrow;
    real sinl(real) @nogc nothrow;
    double __sin(double) @nogc nothrow;
    float sinf(float) @nogc nothrow;
    float __sinf(float) @nogc nothrow;
    real __sinl(real) @nogc nothrow;
    double sin(double) @nogc nothrow;
    double __tan(double) @nogc nothrow;
    float tanf(float) @nogc nothrow;
    float __tanf(float) @nogc nothrow;
    real __tanl(real) @nogc nothrow;
    real tanl(real) @nogc nothrow;
    double tan(double) @nogc nothrow;
    real coshl(real) @nogc nothrow;
    float coshf(float) @nogc nothrow;
    real __coshl(real) @nogc nothrow;
    float __coshf(float) @nogc nothrow;
    double __cosh(double) @nogc nothrow;
    double cosh(double) @nogc nothrow;
    double __sinh(double) @nogc nothrow;
    real __sinhl(real) @nogc nothrow;
    float __sinhf(float) @nogc nothrow;
    float sinhf(float) @nogc nothrow;
    real sinhl(real) @nogc nothrow;
    double sinh(double) @nogc nothrow;
    float __tanhf(float) @nogc nothrow;
    double __tanh(double) @nogc nothrow;
    real tanhl(real) @nogc nothrow;
    real __tanhl(real) @nogc nothrow;
    float tanhf(float) @nogc nothrow;
    double tanh(double) @nogc nothrow;
    double __acosh(double) @nogc nothrow;
    real acoshl(real) @nogc nothrow;
    float __acoshf(float) @nogc nothrow;
    float acoshf(float) @nogc nothrow;
    real __acoshl(real) @nogc nothrow;
    double acosh(double) @nogc nothrow;
    real asinhl(real) @nogc nothrow;
    real __asinhl(real) @nogc nothrow;
    double __asinh(double) @nogc nothrow;
    float asinhf(float) @nogc nothrow;
    float __asinhf(float) @nogc nothrow;
    double asinh(double) @nogc nothrow;
    double __atanh(double) @nogc nothrow;
    real __atanhl(real) @nogc nothrow;
    real atanhl(real) @nogc nothrow;
    float __atanhf(float) @nogc nothrow;
    float atanhf(float) @nogc nothrow;
    double atanh(double) @nogc nothrow;
    float __expf(float) @nogc nothrow;
    double __exp(double) @nogc nothrow;
    real __expl(real) @nogc nothrow;
    real expl(real) @nogc nothrow;
    float expf(float) @nogc nothrow;
    double exp(double) @nogc nothrow;
    double __frexp(double, int*) @nogc nothrow;
    float frexpf(float, int*) @nogc nothrow;
    float __frexpf(float, int*) @nogc nothrow;
    real frexpl(real, int*) @nogc nothrow;
    real __frexpl(real, int*) @nogc nothrow;
    double frexp(double, int*) @nogc nothrow;
    double __ldexp(double, int) @nogc nothrow;
    real ldexpl(real, int) @nogc nothrow;
    float ldexpf(float, int) @nogc nothrow;
    float __ldexpf(float, int) @nogc nothrow;
    real __ldexpl(real, int) @nogc nothrow;
    double ldexp(double, int) @nogc nothrow;
    double __log(double) @nogc nothrow;
    real logl(real) @nogc nothrow;
    real __logl(real) @nogc nothrow;
    float __logf(float) @nogc nothrow;
    float logf(float) @nogc nothrow;
    double log(double) @nogc nothrow;
    real log10l(real) @nogc nothrow;
    double __log10(double) @nogc nothrow;
    real __log10l(real) @nogc nothrow;
    float log10f(float) @nogc nothrow;
    float __log10f(float) @nogc nothrow;
    double log10(double) @nogc nothrow;
    float modff(float, float*) @nogc nothrow;
    float __modff(float, float*) @nogc nothrow;
    real modfl(real, real*) @nogc nothrow;
    double __modf(double, double*) @nogc nothrow;
    real __modfl(real, real*) @nogc nothrow;
    double modf(double, double*) @nogc nothrow;
    double __expm1(double) @nogc nothrow;
    float __expm1f(float) @nogc nothrow;
    float expm1f(float) @nogc nothrow;
    real expm1l(real) @nogc nothrow;
    real __expm1l(real) @nogc nothrow;
    double expm1(double) @nogc nothrow;
    real log1pl(real) @nogc nothrow;
    real __log1pl(real) @nogc nothrow;
    double __log1p(double) @nogc nothrow;
    float log1pf(float) @nogc nothrow;
    float __log1pf(float) @nogc nothrow;
    double log1p(double) @nogc nothrow;
    double __logb(double) @nogc nothrow;
    real logbl(real) @nogc nothrow;
    float __logbf(float) @nogc nothrow;
    float logbf(float) @nogc nothrow;
    real __logbl(real) @nogc nothrow;
    double logb(double) @nogc nothrow;
    double __exp2(double) @nogc nothrow;
    real __exp2l(real) @nogc nothrow;
    float __exp2f(float) @nogc nothrow;
    real exp2l(real) @nogc nothrow;
    float exp2f(float) @nogc nothrow;
    double exp2(double) @nogc nothrow;
    real __log2l(real) @nogc nothrow;
    float log2f(float) @nogc nothrow;
    real log2l(real) @nogc nothrow;
    float __log2f(float) @nogc nothrow;
    double __log2(double) @nogc nothrow;
    double log2(double) @nogc nothrow;
    double __pow(double, double) @nogc nothrow;
    float __powf(float, float) @nogc nothrow;
    float powf(float, float) @nogc nothrow;
    real powl(real, real) @nogc nothrow;
    real __powl(real, real) @nogc nothrow;
    double pow(double, double) @nogc nothrow;
    float sqrtf(float) @nogc nothrow;
    float __sqrtf(float) @nogc nothrow;
    real __sqrtl(real) @nogc nothrow;
    real sqrtl(real) @nogc nothrow;
    double __sqrt(double) @nogc nothrow;
    double sqrt(double) @nogc nothrow;
    float __hypotf(float, float) @nogc nothrow;
    float hypotf(float, float) @nogc nothrow;
    real __hypotl(real, real) @nogc nothrow;
    real hypotl(real, real) @nogc nothrow;
    double __hypot(double, double) @nogc nothrow;
    double hypot(double, double) @nogc nothrow;
    float cbrtf(float) @nogc nothrow;
    double __cbrt(double) @nogc nothrow;
    float __cbrtf(float) @nogc nothrow;
    real cbrtl(real) @nogc nothrow;
    real __cbrtl(real) @nogc nothrow;
    double cbrt(double) @nogc nothrow;
    double __ceil(double) @nogc nothrow;
    float __ceilf(float) @nogc nothrow;
    float ceilf(float) @nogc nothrow;
    real __ceill(real) @nogc nothrow;
    real ceill(real) @nogc nothrow;
    double ceil(double) @nogc nothrow;
    float __fabsf(float) @nogc nothrow;
    float fabsf(float) @nogc nothrow;
    real __fabsl(real) @nogc nothrow;
    real fabsl(real) @nogc nothrow;
    double __fabs(double) @nogc nothrow;
    double fabs(double) @nogc nothrow;
    float floorf(float) @nogc nothrow;
    float __floorf(float) @nogc nothrow;
    double __floor(double) @nogc nothrow;
    real floorl(real) @nogc nothrow;
    real __floorl(real) @nogc nothrow;
    double floor(double) @nogc nothrow;
    real fmodl(real, real) @nogc nothrow;
    float __fmodf(float, float) @nogc nothrow;
    float fmodf(float, float) @nogc nothrow;
    double __fmod(double, double) @nogc nothrow;
    real __fmodl(real, real) @nogc nothrow;
    double fmod(double, double) @nogc nothrow;
    int isinfl(real) @nogc nothrow;
    pragma(mangle, "isinf") int isinf_(double) @nogc nothrow;
    int isinff(float) @nogc nothrow;
    int finitel(real) @nogc nothrow;
    int finitef(float) @nogc nothrow;
    int finite(double) @nogc nothrow;
    real __dreml(real, real) @nogc nothrow;
    float dremf(float, float) @nogc nothrow;
    double drem(double, double) @nogc nothrow;
    double __drem(double, double) @nogc nothrow;
    float __dremf(float, float) @nogc nothrow;
    real dreml(real, real) @nogc nothrow;
    float __significandf(float) @nogc nothrow;
    float significandf(float) @nogc nothrow;
    double __significand(double) @nogc nothrow;
    double significand(double) @nogc nothrow;
    real significandl(real) @nogc nothrow;
    real __significandl(real) @nogc nothrow;
    float __copysignf(float, float) @nogc nothrow;
    float copysignf(float, float) @nogc nothrow;
    double __copysign(double, double) @nogc nothrow;
    real copysignl(real, real) @nogc nothrow;
    real __copysignl(real, real) @nogc nothrow;
    double copysign(double, double) @nogc nothrow;
    real nanl(const(char)*) @nogc nothrow;
    float nanf(const(char)*) @nogc nothrow;
    real __nanl(const(char)*) @nogc nothrow;
    double __nan(const(char)*) @nogc nothrow;
    float __nanf(const(char)*) @nogc nothrow;
    double nan(const(char)*) @nogc nothrow;
    int isnanl(real) @nogc nothrow;
    int isnanf(float) @nogc nothrow;
    pragma(mangle, "isnan") int isnan_(double) @nogc nothrow;
    float __j0f(float) @nogc nothrow;
    real j0l(real) @nogc nothrow;
    float j0f(float) @nogc nothrow;
    real __j0l(real) @nogc nothrow;
    double j0(double) @nogc nothrow;
    double __j0(double) @nogc nothrow;
    float __j1f(float) @nogc nothrow;
    float j1f(float) @nogc nothrow;
    real j1l(real) @nogc nothrow;
    double j1(double) @nogc nothrow;
    double __j1(double) @nogc nothrow;
    real __j1l(real) @nogc nothrow;
    float __jnf(int, float) @nogc nothrow;
    float jnf(int, float) @nogc nothrow;
    real __jnl(int, real) @nogc nothrow;
    real jnl(int, real) @nogc nothrow;
    double jn(int, double) @nogc nothrow;
    double __jn(int, double) @nogc nothrow;
    double __y0(double) @nogc nothrow;
    float __y0f(float) @nogc nothrow;
    double y0(double) @nogc nothrow;
    real __y0l(real) @nogc nothrow;
    real y0l(real) @nogc nothrow;
    float y0f(float) @nogc nothrow;
    float __y1f(float) @nogc nothrow;
    float y1f(float) @nogc nothrow;
    real __y1l(real) @nogc nothrow;
    double y1(double) @nogc nothrow;
    double __y1(double) @nogc nothrow;
    real y1l(real) @nogc nothrow;
    double yn(int, double) @nogc nothrow;
    real ynl(int, real) @nogc nothrow;
    double __yn(int, double) @nogc nothrow;
    real __ynl(int, real) @nogc nothrow;
    float ynf(int, float) @nogc nothrow;
    float __ynf(int, float) @nogc nothrow;
    float __erff(float) @nogc nothrow;
    real __erfl(real) @nogc nothrow;
    real erfl(real) @nogc nothrow;
    float erff(float) @nogc nothrow;
    double __erf(double) @nogc nothrow;
    double erf(double) @nogc nothrow;
    float __erfcf(float) @nogc nothrow;
    float erfcf(float) @nogc nothrow;
    real __erfcl(real) @nogc nothrow;
    real erfcl(real) @nogc nothrow;
    double __erfc(double) @nogc nothrow;
    double erfc(double) @nogc nothrow;
    float lgammaf(float) @nogc nothrow;
    float __lgammaf(float) @nogc nothrow;
    real __lgammal(real) @nogc nothrow;
    real lgammal(real) @nogc nothrow;
    double __lgamma(double) @nogc nothrow;
    double lgamma(double) @nogc nothrow;
    float tgammaf(float) @nogc nothrow;
    float __tgammaf(float) @nogc nothrow;
    real __tgammal(real) @nogc nothrow;
    real tgammal(real) @nogc nothrow;
    double __tgamma(double) @nogc nothrow;
    double tgamma(double) @nogc nothrow;
    real gammal(real) @nogc nothrow;
    real __gammal(real) @nogc nothrow;
    double gamma(double) @nogc nothrow;
    double __gamma(double) @nogc nothrow;
    float gammaf(float) @nogc nothrow;
    float __gammaf(float) @nogc nothrow;
    double lgamma_r(double, int*) @nogc nothrow;
    double __lgamma_r(double, int*) @nogc nothrow;
    float lgammaf_r(float, int*) @nogc nothrow;
    real lgammal_r(real, int*) @nogc nothrow;
    real __lgammal_r(real, int*) @nogc nothrow;
    float __lgammaf_r(float, int*) @nogc nothrow;
    float rintf(float) @nogc nothrow;
    float __rintf(float) @nogc nothrow;
    double __rint(double) @nogc nothrow;
    real rintl(real) @nogc nothrow;
    real __rintl(real) @nogc nothrow;
    double rint(double) @nogc nothrow;
    float nextafterf(float, float) @nogc nothrow;
    float __nextafterf(float, float) @nogc nothrow;
    real __nextafterl(real, real) @nogc nothrow;
    real nextafterl(real, real) @nogc nothrow;
    double __nextafter(double, double) @nogc nothrow;
    double nextafter(double, double) @nogc nothrow;
    float nexttowardf(float, real) @nogc nothrow;
    real nexttowardl(real, real) @nogc nothrow;
    float __nexttowardf(float, real) @nogc nothrow;
    double __nexttoward(double, real) @nogc nothrow;
    real __nexttowardl(real, real) @nogc nothrow;
    double nexttoward(double, real) @nogc nothrow;
    real __remainderl(real, real) @nogc nothrow;
    float __remainderf(float, float) @nogc nothrow;
    real remainderl(real, real) @nogc nothrow;
    double __remainder(double, double) @nogc nothrow;
    float remainderf(float, float) @nogc nothrow;
    double remainder(double, double) @nogc nothrow;
    real __scalbnl(real, int) @nogc nothrow;
    real scalbnl(real, int) @nogc nothrow;
    float scalbnf(float, int) @nogc nothrow;
    float __scalbnf(float, int) @nogc nothrow;
    double __scalbn(double, int) @nogc nothrow;
    double scalbn(double, int) @nogc nothrow;
    int ilogbf(float) @nogc nothrow;
    int __ilogbf(float) @nogc nothrow;
    int __ilogbl(real) @nogc nothrow;
    int __ilogb(double) @nogc nothrow;
    int ilogbl(real) @nogc nothrow;
    int ilogb(double) @nogc nothrow;
    float scalblnf(float, c_long) @nogc nothrow;
    float __scalblnf(float, c_long) @nogc nothrow;
    real scalblnl(real, c_long) @nogc nothrow;
    real __scalblnl(real, c_long) @nogc nothrow;
    double __scalbln(double, c_long) @nogc nothrow;
    double scalbln(double, c_long) @nogc nothrow;
    real nearbyintl(real) @nogc nothrow;
    real __nearbyintl(real) @nogc nothrow;
    float __nearbyintf(float) @nogc nothrow;
    double __nearbyint(double) @nogc nothrow;
    float nearbyintf(float) @nogc nothrow;
    double nearbyint(double) @nogc nothrow;
    float roundf(float) @nogc nothrow;
    float __roundf(float) @nogc nothrow;
    double __round(double) @nogc nothrow;
    real roundl(real) @nogc nothrow;
    real __roundl(real) @nogc nothrow;
    double round(double) @nogc nothrow;
    real __truncl(real) @nogc nothrow;
    real truncl(real) @nogc nothrow;
    double __trunc(double) @nogc nothrow;
    float __truncf(float) @nogc nothrow;
    float truncf(float) @nogc nothrow;
    double trunc(double) @nogc nothrow;
    float __remquof(float, float, int*) @nogc nothrow;
    float remquof(float, float, int*) @nogc nothrow;
    real __remquol(real, real, int*) @nogc nothrow;
    real remquol(real, real, int*) @nogc nothrow;
    double __remquo(double, double, int*) @nogc nothrow;
    double remquo(double, double, int*) @nogc nothrow;
    c_long __lrintf(float) @nogc nothrow;
    c_long lrintf(float) @nogc nothrow;
    c_long __lrint(double) @nogc nothrow;
    c_long lrintl(real) @nogc nothrow;
    c_long __lrintl(real) @nogc nothrow;
    c_long lrint(double) @nogc nothrow;
    long __llrintf(float) @nogc nothrow;
    long llrintf(float) @nogc nothrow;
    long __llrintl(real) @nogc nothrow;
    long __llrint(double) @nogc nothrow;



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
    static if(!is(typeof(_BITS_LIBM_SIMD_DECL_STUBS_H))) {
        private enum enumMixinStr__BITS_LIBM_SIMD_DECL_STUBS_H = `enum _BITS_LIBM_SIMD_DECL_STUBS_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__BITS_LIBM_SIMD_DECL_STUBS_H); }))) {
            mixin(enumMixinStr__BITS_LIBM_SIMD_DECL_STUBS_H);
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




    static if(!is(typeof(__FP_LOGBNAN_IS_MIN))) {
        private enum enumMixinStr___FP_LOGBNAN_IS_MIN = `enum __FP_LOGBNAN_IS_MIN = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___FP_LOGBNAN_IS_MIN); }))) {
            mixin(enumMixinStr___FP_LOGBNAN_IS_MIN);
        }
    }




    static if(!is(typeof(__FP_LOGB0_IS_MIN))) {
        private enum enumMixinStr___FP_LOGB0_IS_MIN = `enum __FP_LOGB0_IS_MIN = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___FP_LOGB0_IS_MIN); }))) {
            mixin(enumMixinStr___FP_LOGB0_IS_MIN);
        }
    }




    static if(!is(typeof(__GLIBC_FLT_EVAL_METHOD))) {
        private enum enumMixinStr___GLIBC_FLT_EVAL_METHOD = `enum __GLIBC_FLT_EVAL_METHOD = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___GLIBC_FLT_EVAL_METHOD); }))) {
            mixin(enumMixinStr___GLIBC_FLT_EVAL_METHOD);
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




    static if(!is(typeof(_BITS_POSIX1_LIM_H))) {
        private enum enumMixinStr__BITS_POSIX1_LIM_H = `enum _BITS_POSIX1_LIM_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__BITS_POSIX1_LIM_H); }))) {
            mixin(enumMixinStr__BITS_POSIX1_LIM_H);
        }
    }




    static if(!is(typeof(_POSIX_AIO_LISTIO_MAX))) {
        private enum enumMixinStr__POSIX_AIO_LISTIO_MAX = `enum _POSIX_AIO_LISTIO_MAX = 2;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_AIO_LISTIO_MAX); }))) {
            mixin(enumMixinStr__POSIX_AIO_LISTIO_MAX);
        }
    }




    static if(!is(typeof(_POSIX_AIO_MAX))) {
        private enum enumMixinStr__POSIX_AIO_MAX = `enum _POSIX_AIO_MAX = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_AIO_MAX); }))) {
            mixin(enumMixinStr__POSIX_AIO_MAX);
        }
    }




    static if(!is(typeof(_POSIX_ARG_MAX))) {
        private enum enumMixinStr__POSIX_ARG_MAX = `enum _POSIX_ARG_MAX = 4096;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_ARG_MAX); }))) {
            mixin(enumMixinStr__POSIX_ARG_MAX);
        }
    }




    static if(!is(typeof(__HAVE_FLOAT64))) {
        private enum enumMixinStr___HAVE_FLOAT64 = `enum __HAVE_FLOAT64 = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_FLOAT64); }))) {
            mixin(enumMixinStr___HAVE_FLOAT64);
        }
    }




    static if(!is(typeof(_POSIX_CHILD_MAX))) {
        private enum enumMixinStr__POSIX_CHILD_MAX = `enum _POSIX_CHILD_MAX = 25;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_CHILD_MAX); }))) {
            mixin(enumMixinStr__POSIX_CHILD_MAX);
        }
    }




    static if(!is(typeof(_POSIX_DELAYTIMER_MAX))) {
        private enum enumMixinStr__POSIX_DELAYTIMER_MAX = `enum _POSIX_DELAYTIMER_MAX = 32;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_DELAYTIMER_MAX); }))) {
            mixin(enumMixinStr__POSIX_DELAYTIMER_MAX);
        }
    }




    static if(!is(typeof(_POSIX_HOST_NAME_MAX))) {
        private enum enumMixinStr__POSIX_HOST_NAME_MAX = `enum _POSIX_HOST_NAME_MAX = 255;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_HOST_NAME_MAX); }))) {
            mixin(enumMixinStr__POSIX_HOST_NAME_MAX);
        }
    }




    static if(!is(typeof(_POSIX_LINK_MAX))) {
        private enum enumMixinStr__POSIX_LINK_MAX = `enum _POSIX_LINK_MAX = 8;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_LINK_MAX); }))) {
            mixin(enumMixinStr__POSIX_LINK_MAX);
        }
    }




    static if(!is(typeof(_POSIX_LOGIN_NAME_MAX))) {
        private enum enumMixinStr__POSIX_LOGIN_NAME_MAX = `enum _POSIX_LOGIN_NAME_MAX = 9;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_LOGIN_NAME_MAX); }))) {
            mixin(enumMixinStr__POSIX_LOGIN_NAME_MAX);
        }
    }




    static if(!is(typeof(_POSIX_MAX_CANON))) {
        private enum enumMixinStr__POSIX_MAX_CANON = `enum _POSIX_MAX_CANON = 255;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_MAX_CANON); }))) {
            mixin(enumMixinStr__POSIX_MAX_CANON);
        }
    }




    static if(!is(typeof(_POSIX_MAX_INPUT))) {
        private enum enumMixinStr__POSIX_MAX_INPUT = `enum _POSIX_MAX_INPUT = 255;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_MAX_INPUT); }))) {
            mixin(enumMixinStr__POSIX_MAX_INPUT);
        }
    }




    static if(!is(typeof(_POSIX_MQ_OPEN_MAX))) {
        private enum enumMixinStr__POSIX_MQ_OPEN_MAX = `enum _POSIX_MQ_OPEN_MAX = 8;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_MQ_OPEN_MAX); }))) {
            mixin(enumMixinStr__POSIX_MQ_OPEN_MAX);
        }
    }




    static if(!is(typeof(_POSIX_MQ_PRIO_MAX))) {
        private enum enumMixinStr__POSIX_MQ_PRIO_MAX = `enum _POSIX_MQ_PRIO_MAX = 32;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_MQ_PRIO_MAX); }))) {
            mixin(enumMixinStr__POSIX_MQ_PRIO_MAX);
        }
    }




    static if(!is(typeof(_POSIX_NAME_MAX))) {
        private enum enumMixinStr__POSIX_NAME_MAX = `enum _POSIX_NAME_MAX = 14;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_NAME_MAX); }))) {
            mixin(enumMixinStr__POSIX_NAME_MAX);
        }
    }




    static if(!is(typeof(__HAVE_FLOAT32))) {
        private enum enumMixinStr___HAVE_FLOAT32 = `enum __HAVE_FLOAT32 = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_FLOAT32); }))) {
            mixin(enumMixinStr___HAVE_FLOAT32);
        }
    }




    static if(!is(typeof(_POSIX_NGROUPS_MAX))) {
        private enum enumMixinStr__POSIX_NGROUPS_MAX = `enum _POSIX_NGROUPS_MAX = 8;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_NGROUPS_MAX); }))) {
            mixin(enumMixinStr__POSIX_NGROUPS_MAX);
        }
    }




    static if(!is(typeof(__HAVE_FLOAT16))) {
        private enum enumMixinStr___HAVE_FLOAT16 = `enum __HAVE_FLOAT16 = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_FLOAT16); }))) {
            mixin(enumMixinStr___HAVE_FLOAT16);
        }
    }




    static if(!is(typeof(_POSIX_OPEN_MAX))) {
        private enum enumMixinStr__POSIX_OPEN_MAX = `enum _POSIX_OPEN_MAX = 20;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_OPEN_MAX); }))) {
            mixin(enumMixinStr__POSIX_OPEN_MAX);
        }
    }






    static if(!is(typeof(_POSIX_PATH_MAX))) {
        private enum enumMixinStr__POSIX_PATH_MAX = `enum _POSIX_PATH_MAX = 256;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_PATH_MAX); }))) {
            mixin(enumMixinStr__POSIX_PATH_MAX);
        }
    }




    static if(!is(typeof(_POSIX_PIPE_BUF))) {
        private enum enumMixinStr__POSIX_PIPE_BUF = `enum _POSIX_PIPE_BUF = 512;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_PIPE_BUF); }))) {
            mixin(enumMixinStr__POSIX_PIPE_BUF);
        }
    }




    static if(!is(typeof(_POSIX_RE_DUP_MAX))) {
        private enum enumMixinStr__POSIX_RE_DUP_MAX = `enum _POSIX_RE_DUP_MAX = 255;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_RE_DUP_MAX); }))) {
            mixin(enumMixinStr__POSIX_RE_DUP_MAX);
        }
    }




    static if(!is(typeof(_POSIX_RTSIG_MAX))) {
        private enum enumMixinStr__POSIX_RTSIG_MAX = `enum _POSIX_RTSIG_MAX = 8;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_RTSIG_MAX); }))) {
            mixin(enumMixinStr__POSIX_RTSIG_MAX);
        }
    }




    static if(!is(typeof(_POSIX_SEM_NSEMS_MAX))) {
        private enum enumMixinStr__POSIX_SEM_NSEMS_MAX = `enum _POSIX_SEM_NSEMS_MAX = 256;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_SEM_NSEMS_MAX); }))) {
            mixin(enumMixinStr__POSIX_SEM_NSEMS_MAX);
        }
    }




    static if(!is(typeof(_POSIX_SEM_VALUE_MAX))) {
        private enum enumMixinStr__POSIX_SEM_VALUE_MAX = `enum _POSIX_SEM_VALUE_MAX = 32767;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_SEM_VALUE_MAX); }))) {
            mixin(enumMixinStr__POSIX_SEM_VALUE_MAX);
        }
    }




    static if(!is(typeof(_POSIX_SIGQUEUE_MAX))) {
        private enum enumMixinStr__POSIX_SIGQUEUE_MAX = `enum _POSIX_SIGQUEUE_MAX = 32;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_SIGQUEUE_MAX); }))) {
            mixin(enumMixinStr__POSIX_SIGQUEUE_MAX);
        }
    }




    static if(!is(typeof(_POSIX_SSIZE_MAX))) {
        private enum enumMixinStr__POSIX_SSIZE_MAX = `enum _POSIX_SSIZE_MAX = 32767;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_SSIZE_MAX); }))) {
            mixin(enumMixinStr__POSIX_SSIZE_MAX);
        }
    }




    static if(!is(typeof(_POSIX_STREAM_MAX))) {
        private enum enumMixinStr__POSIX_STREAM_MAX = `enum _POSIX_STREAM_MAX = 8;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_STREAM_MAX); }))) {
            mixin(enumMixinStr__POSIX_STREAM_MAX);
        }
    }




    static if(!is(typeof(_POSIX_SYMLINK_MAX))) {
        private enum enumMixinStr__POSIX_SYMLINK_MAX = `enum _POSIX_SYMLINK_MAX = 255;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_SYMLINK_MAX); }))) {
            mixin(enumMixinStr__POSIX_SYMLINK_MAX);
        }
    }




    static if(!is(typeof(_POSIX_SYMLOOP_MAX))) {
        private enum enumMixinStr__POSIX_SYMLOOP_MAX = `enum _POSIX_SYMLOOP_MAX = 8;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_SYMLOOP_MAX); }))) {
            mixin(enumMixinStr__POSIX_SYMLOOP_MAX);
        }
    }




    static if(!is(typeof(_POSIX_TIMER_MAX))) {
        private enum enumMixinStr__POSIX_TIMER_MAX = `enum _POSIX_TIMER_MAX = 32;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_TIMER_MAX); }))) {
            mixin(enumMixinStr__POSIX_TIMER_MAX);
        }
    }




    static if(!is(typeof(_POSIX_TTY_NAME_MAX))) {
        private enum enumMixinStr__POSIX_TTY_NAME_MAX = `enum _POSIX_TTY_NAME_MAX = 9;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_TTY_NAME_MAX); }))) {
            mixin(enumMixinStr__POSIX_TTY_NAME_MAX);
        }
    }




    static if(!is(typeof(_POSIX_TZNAME_MAX))) {
        private enum enumMixinStr__POSIX_TZNAME_MAX = `enum _POSIX_TZNAME_MAX = 6;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_TZNAME_MAX); }))) {
            mixin(enumMixinStr__POSIX_TZNAME_MAX);
        }
    }




    static if(!is(typeof(_POSIX_CLOCKRES_MIN))) {
        private enum enumMixinStr__POSIX_CLOCKRES_MIN = `enum _POSIX_CLOCKRES_MIN = 20000000;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_CLOCKRES_MIN); }))) {
            mixin(enumMixinStr__POSIX_CLOCKRES_MIN);
        }
    }




    static if(!is(typeof(SSIZE_MAX))) {
        private enum enumMixinStr_SSIZE_MAX = `enum SSIZE_MAX = LONG_MAX;`;
        static if(is(typeof({ mixin(enumMixinStr_SSIZE_MAX); }))) {
            mixin(enumMixinStr_SSIZE_MAX);
        }
    }




    static if(!is(typeof(_BITS_POSIX2_LIM_H))) {
        private enum enumMixinStr__BITS_POSIX2_LIM_H = `enum _BITS_POSIX2_LIM_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__BITS_POSIX2_LIM_H); }))) {
            mixin(enumMixinStr__BITS_POSIX2_LIM_H);
        }
    }




    static if(!is(typeof(_POSIX2_BC_BASE_MAX))) {
        private enum enumMixinStr__POSIX2_BC_BASE_MAX = `enum _POSIX2_BC_BASE_MAX = 99;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX2_BC_BASE_MAX); }))) {
            mixin(enumMixinStr__POSIX2_BC_BASE_MAX);
        }
    }




    static if(!is(typeof(_POSIX2_BC_DIM_MAX))) {
        private enum enumMixinStr__POSIX2_BC_DIM_MAX = `enum _POSIX2_BC_DIM_MAX = 2048;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX2_BC_DIM_MAX); }))) {
            mixin(enumMixinStr__POSIX2_BC_DIM_MAX);
        }
    }




    static if(!is(typeof(_POSIX2_BC_SCALE_MAX))) {
        private enum enumMixinStr__POSIX2_BC_SCALE_MAX = `enum _POSIX2_BC_SCALE_MAX = 99;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX2_BC_SCALE_MAX); }))) {
            mixin(enumMixinStr__POSIX2_BC_SCALE_MAX);
        }
    }




    static if(!is(typeof(_POSIX2_BC_STRING_MAX))) {
        private enum enumMixinStr__POSIX2_BC_STRING_MAX = `enum _POSIX2_BC_STRING_MAX = 1000;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX2_BC_STRING_MAX); }))) {
            mixin(enumMixinStr__POSIX2_BC_STRING_MAX);
        }
    }




    static if(!is(typeof(_POSIX2_COLL_WEIGHTS_MAX))) {
        private enum enumMixinStr__POSIX2_COLL_WEIGHTS_MAX = `enum _POSIX2_COLL_WEIGHTS_MAX = 2;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX2_COLL_WEIGHTS_MAX); }))) {
            mixin(enumMixinStr__POSIX2_COLL_WEIGHTS_MAX);
        }
    }




    static if(!is(typeof(_POSIX2_EXPR_NEST_MAX))) {
        private enum enumMixinStr__POSIX2_EXPR_NEST_MAX = `enum _POSIX2_EXPR_NEST_MAX = 32;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX2_EXPR_NEST_MAX); }))) {
            mixin(enumMixinStr__POSIX2_EXPR_NEST_MAX);
        }
    }




    static if(!is(typeof(_POSIX2_LINE_MAX))) {
        private enum enumMixinStr__POSIX2_LINE_MAX = `enum _POSIX2_LINE_MAX = 2048;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX2_LINE_MAX); }))) {
            mixin(enumMixinStr__POSIX2_LINE_MAX);
        }
    }




    static if(!is(typeof(_POSIX2_RE_DUP_MAX))) {
        private enum enumMixinStr__POSIX2_RE_DUP_MAX = `enum _POSIX2_RE_DUP_MAX = 255;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX2_RE_DUP_MAX); }))) {
            mixin(enumMixinStr__POSIX2_RE_DUP_MAX);
        }
    }




    static if(!is(typeof(_POSIX2_CHARCLASS_NAME_MAX))) {
        private enum enumMixinStr__POSIX2_CHARCLASS_NAME_MAX = `enum _POSIX2_CHARCLASS_NAME_MAX = 14;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX2_CHARCLASS_NAME_MAX); }))) {
            mixin(enumMixinStr__POSIX2_CHARCLASS_NAME_MAX);
        }
    }




    static if(!is(typeof(BC_BASE_MAX))) {
        private enum enumMixinStr_BC_BASE_MAX = `enum BC_BASE_MAX = 99;`;
        static if(is(typeof({ mixin(enumMixinStr_BC_BASE_MAX); }))) {
            mixin(enumMixinStr_BC_BASE_MAX);
        }
    }




    static if(!is(typeof(BC_DIM_MAX))) {
        private enum enumMixinStr_BC_DIM_MAX = `enum BC_DIM_MAX = 2048;`;
        static if(is(typeof({ mixin(enumMixinStr_BC_DIM_MAX); }))) {
            mixin(enumMixinStr_BC_DIM_MAX);
        }
    }




    static if(!is(typeof(BC_SCALE_MAX))) {
        private enum enumMixinStr_BC_SCALE_MAX = `enum BC_SCALE_MAX = 99;`;
        static if(is(typeof({ mixin(enumMixinStr_BC_SCALE_MAX); }))) {
            mixin(enumMixinStr_BC_SCALE_MAX);
        }
    }




    static if(!is(typeof(BC_STRING_MAX))) {
        private enum enumMixinStr_BC_STRING_MAX = `enum BC_STRING_MAX = 1000;`;
        static if(is(typeof({ mixin(enumMixinStr_BC_STRING_MAX); }))) {
            mixin(enumMixinStr_BC_STRING_MAX);
        }
    }




    static if(!is(typeof(COLL_WEIGHTS_MAX))) {
        private enum enumMixinStr_COLL_WEIGHTS_MAX = `enum COLL_WEIGHTS_MAX = 255;`;
        static if(is(typeof({ mixin(enumMixinStr_COLL_WEIGHTS_MAX); }))) {
            mixin(enumMixinStr_COLL_WEIGHTS_MAX);
        }
    }




    static if(!is(typeof(EXPR_NEST_MAX))) {
        private enum enumMixinStr_EXPR_NEST_MAX = `enum EXPR_NEST_MAX = 32;`;
        static if(is(typeof({ mixin(enumMixinStr_EXPR_NEST_MAX); }))) {
            mixin(enumMixinStr_EXPR_NEST_MAX);
        }
    }




    static if(!is(typeof(LINE_MAX))) {
        private enum enumMixinStr_LINE_MAX = `enum LINE_MAX = 2048;`;
        static if(is(typeof({ mixin(enumMixinStr_LINE_MAX); }))) {
            mixin(enumMixinStr_LINE_MAX);
        }
    }




    static if(!is(typeof(CHARCLASS_NAME_MAX))) {
        private enum enumMixinStr_CHARCLASS_NAME_MAX = `enum CHARCLASS_NAME_MAX = 2048;`;
        static if(is(typeof({ mixin(enumMixinStr_CHARCLASS_NAME_MAX); }))) {
            mixin(enumMixinStr_CHARCLASS_NAME_MAX);
        }
    }




    static if(!is(typeof(RE_DUP_MAX))) {
        private enum enumMixinStr_RE_DUP_MAX = `enum RE_DUP_MAX = ( 0x7fff );`;
        static if(is(typeof({ mixin(enumMixinStr_RE_DUP_MAX); }))) {
            mixin(enumMixinStr_RE_DUP_MAX);
        }
    }




    static if(!is(typeof(_BITS_TYPES_H))) {
        private enum enumMixinStr__BITS_TYPES_H = `enum _BITS_TYPES_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__BITS_TYPES_H); }))) {
            mixin(enumMixinStr__BITS_TYPES_H);
        }
    }




    static if(!is(typeof(_STDC_PREDEF_H))) {
        private enum enumMixinStr__STDC_PREDEF_H = `enum _STDC_PREDEF_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__STDC_PREDEF_H); }))) {
            mixin(enumMixinStr__STDC_PREDEF_H);
        }
    }
    static if(!is(typeof(M_SQRT1_2))) {
        private enum enumMixinStr_M_SQRT1_2 = `enum M_SQRT1_2 = 0.70710678118654752440;`;
        static if(is(typeof({ mixin(enumMixinStr_M_SQRT1_2); }))) {
            mixin(enumMixinStr_M_SQRT1_2);
        }
    }




    static if(!is(typeof(M_SQRT2))) {
        private enum enumMixinStr_M_SQRT2 = `enum M_SQRT2 = 1.41421356237309504880;`;
        static if(is(typeof({ mixin(enumMixinStr_M_SQRT2); }))) {
            mixin(enumMixinStr_M_SQRT2);
        }
    }




    static if(!is(typeof(M_2_SQRTPI))) {
        private enum enumMixinStr_M_2_SQRTPI = `enum M_2_SQRTPI = 1.12837916709551257390;`;
        static if(is(typeof({ mixin(enumMixinStr_M_2_SQRTPI); }))) {
            mixin(enumMixinStr_M_2_SQRTPI);
        }
    }




    static if(!is(typeof(M_2_PI))) {
        private enum enumMixinStr_M_2_PI = `enum M_2_PI = 0.63661977236758134308;`;
        static if(is(typeof({ mixin(enumMixinStr_M_2_PI); }))) {
            mixin(enumMixinStr_M_2_PI);
        }
    }




    static if(!is(typeof(M_1_PI))) {
        private enum enumMixinStr_M_1_PI = `enum M_1_PI = 0.31830988618379067154;`;
        static if(is(typeof({ mixin(enumMixinStr_M_1_PI); }))) {
            mixin(enumMixinStr_M_1_PI);
        }
    }




    static if(!is(typeof(M_PI_4))) {
        private enum enumMixinStr_M_PI_4 = `enum M_PI_4 = 0.78539816339744830962;`;
        static if(is(typeof({ mixin(enumMixinStr_M_PI_4); }))) {
            mixin(enumMixinStr_M_PI_4);
        }
    }




    static if(!is(typeof(M_PI_2))) {
        private enum enumMixinStr_M_PI_2 = `enum M_PI_2 = 1.57079632679489661923;`;
        static if(is(typeof({ mixin(enumMixinStr_M_PI_2); }))) {
            mixin(enumMixinStr_M_PI_2);
        }
    }




    static if(!is(typeof(M_PI))) {
        private enum enumMixinStr_M_PI = `enum M_PI = 3.14159265358979323846;`;
        static if(is(typeof({ mixin(enumMixinStr_M_PI); }))) {
            mixin(enumMixinStr_M_PI);
        }
    }




    static if(!is(typeof(M_LN10))) {
        private enum enumMixinStr_M_LN10 = `enum M_LN10 = 2.30258509299404568402;`;
        static if(is(typeof({ mixin(enumMixinStr_M_LN10); }))) {
            mixin(enumMixinStr_M_LN10);
        }
    }




    static if(!is(typeof(M_LN2))) {
        private enum enumMixinStr_M_LN2 = `enum M_LN2 = 0.69314718055994530942;`;
        static if(is(typeof({ mixin(enumMixinStr_M_LN2); }))) {
            mixin(enumMixinStr_M_LN2);
        }
    }




    static if(!is(typeof(M_LOG10E))) {
        private enum enumMixinStr_M_LOG10E = `enum M_LOG10E = 0.43429448190325182765;`;
        static if(is(typeof({ mixin(enumMixinStr_M_LOG10E); }))) {
            mixin(enumMixinStr_M_LOG10E);
        }
    }




    static if(!is(typeof(M_LOG2E))) {
        private enum enumMixinStr_M_LOG2E = `enum M_LOG2E = 1.4426950408889634074;`;
        static if(is(typeof({ mixin(enumMixinStr_M_LOG2E); }))) {
            mixin(enumMixinStr_M_LOG2E);
        }
    }




    static if(!is(typeof(M_E))) {
        private enum enumMixinStr_M_E = `enum M_E = 2.7182818284590452354;`;
        static if(is(typeof({ mixin(enumMixinStr_M_E); }))) {
            mixin(enumMixinStr_M_E);
        }
    }




    static if(!is(typeof(math_errhandling))) {
        private enum enumMixinStr_math_errhandling = `enum math_errhandling = ( MATH_ERRNO | MATH_ERREXCEPT );`;
        static if(is(typeof({ mixin(enumMixinStr_math_errhandling); }))) {
            mixin(enumMixinStr_math_errhandling);
        }
    }




    static if(!is(typeof(__S16_TYPE))) {
        private enum enumMixinStr___S16_TYPE = `enum __S16_TYPE = short int;`;
        static if(is(typeof({ mixin(enumMixinStr___S16_TYPE); }))) {
            mixin(enumMixinStr___S16_TYPE);
        }
    }




    static if(!is(typeof(__U16_TYPE))) {
        private enum enumMixinStr___U16_TYPE = `enum __U16_TYPE = unsigned short int;`;
        static if(is(typeof({ mixin(enumMixinStr___U16_TYPE); }))) {
            mixin(enumMixinStr___U16_TYPE);
        }
    }




    static if(!is(typeof(__S32_TYPE))) {
        private enum enumMixinStr___S32_TYPE = `enum __S32_TYPE = int;`;
        static if(is(typeof({ mixin(enumMixinStr___S32_TYPE); }))) {
            mixin(enumMixinStr___S32_TYPE);
        }
    }




    static if(!is(typeof(__U32_TYPE))) {
        private enum enumMixinStr___U32_TYPE = `enum __U32_TYPE = unsigned int;`;
        static if(is(typeof({ mixin(enumMixinStr___U32_TYPE); }))) {
            mixin(enumMixinStr___U32_TYPE);
        }
    }




    static if(!is(typeof(__SLONGWORD_TYPE))) {
        private enum enumMixinStr___SLONGWORD_TYPE = `enum __SLONGWORD_TYPE = long int;`;
        static if(is(typeof({ mixin(enumMixinStr___SLONGWORD_TYPE); }))) {
            mixin(enumMixinStr___SLONGWORD_TYPE);
        }
    }




    static if(!is(typeof(__ULONGWORD_TYPE))) {
        private enum enumMixinStr___ULONGWORD_TYPE = `enum __ULONGWORD_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___ULONGWORD_TYPE); }))) {
            mixin(enumMixinStr___ULONGWORD_TYPE);
        }
    }




    static if(!is(typeof(MATH_ERREXCEPT))) {
        private enum enumMixinStr_MATH_ERREXCEPT = `enum MATH_ERREXCEPT = 2;`;
        static if(is(typeof({ mixin(enumMixinStr_MATH_ERREXCEPT); }))) {
            mixin(enumMixinStr_MATH_ERREXCEPT);
        }
    }




    static if(!is(typeof(MATH_ERRNO))) {
        private enum enumMixinStr_MATH_ERRNO = `enum MATH_ERRNO = 1;`;
        static if(is(typeof({ mixin(enumMixinStr_MATH_ERRNO); }))) {
            mixin(enumMixinStr_MATH_ERRNO);
        }
    }




    static if(!is(typeof(__SQUAD_TYPE))) {
        private enum enumMixinStr___SQUAD_TYPE = `enum __SQUAD_TYPE = long int;`;
        static if(is(typeof({ mixin(enumMixinStr___SQUAD_TYPE); }))) {
            mixin(enumMixinStr___SQUAD_TYPE);
        }
    }




    static if(!is(typeof(__UQUAD_TYPE))) {
        private enum enumMixinStr___UQUAD_TYPE = `enum __UQUAD_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___UQUAD_TYPE); }))) {
            mixin(enumMixinStr___UQUAD_TYPE);
        }
    }




    static if(!is(typeof(__SWORD_TYPE))) {
        private enum enumMixinStr___SWORD_TYPE = `enum __SWORD_TYPE = long int;`;
        static if(is(typeof({ mixin(enumMixinStr___SWORD_TYPE); }))) {
            mixin(enumMixinStr___SWORD_TYPE);
        }
    }




    static if(!is(typeof(__UWORD_TYPE))) {
        private enum enumMixinStr___UWORD_TYPE = `enum __UWORD_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___UWORD_TYPE); }))) {
            mixin(enumMixinStr___UWORD_TYPE);
        }
    }




    static if(!is(typeof(__SLONG32_TYPE))) {
        private enum enumMixinStr___SLONG32_TYPE = `enum __SLONG32_TYPE = int;`;
        static if(is(typeof({ mixin(enumMixinStr___SLONG32_TYPE); }))) {
            mixin(enumMixinStr___SLONG32_TYPE);
        }
    }




    static if(!is(typeof(__ULONG32_TYPE))) {
        private enum enumMixinStr___ULONG32_TYPE = `enum __ULONG32_TYPE = unsigned int;`;
        static if(is(typeof({ mixin(enumMixinStr___ULONG32_TYPE); }))) {
            mixin(enumMixinStr___ULONG32_TYPE);
        }
    }




    static if(!is(typeof(__S64_TYPE))) {
        private enum enumMixinStr___S64_TYPE = `enum __S64_TYPE = long int;`;
        static if(is(typeof({ mixin(enumMixinStr___S64_TYPE); }))) {
            mixin(enumMixinStr___S64_TYPE);
        }
    }




    static if(!is(typeof(__U64_TYPE))) {
        private enum enumMixinStr___U64_TYPE = `enum __U64_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___U64_TYPE); }))) {
            mixin(enumMixinStr___U64_TYPE);
        }
    }




    static if(!is(typeof(__STD_TYPE))) {
        private enum enumMixinStr___STD_TYPE = `enum __STD_TYPE = typedef;`;
        static if(is(typeof({ mixin(enumMixinStr___STD_TYPE); }))) {
            mixin(enumMixinStr___STD_TYPE);
        }
    }
    static if(!is(typeof(FP_NORMAL))) {
        private enum enumMixinStr_FP_NORMAL = `enum FP_NORMAL = 4;`;
        static if(is(typeof({ mixin(enumMixinStr_FP_NORMAL); }))) {
            mixin(enumMixinStr_FP_NORMAL);
        }
    }




    static if(!is(typeof(FP_SUBNORMAL))) {
        private enum enumMixinStr_FP_SUBNORMAL = `enum FP_SUBNORMAL = 3;`;
        static if(is(typeof({ mixin(enumMixinStr_FP_SUBNORMAL); }))) {
            mixin(enumMixinStr_FP_SUBNORMAL);
        }
    }




    static if(!is(typeof(FP_ZERO))) {
        private enum enumMixinStr_FP_ZERO = `enum FP_ZERO = 2;`;
        static if(is(typeof({ mixin(enumMixinStr_FP_ZERO); }))) {
            mixin(enumMixinStr_FP_ZERO);
        }
    }




    static if(!is(typeof(FP_INFINITE))) {
        private enum enumMixinStr_FP_INFINITE = `enum FP_INFINITE = 1;`;
        static if(is(typeof({ mixin(enumMixinStr_FP_INFINITE); }))) {
            mixin(enumMixinStr_FP_INFINITE);
        }
    }




    static if(!is(typeof(FP_NAN))) {
        private enum enumMixinStr_FP_NAN = `enum FP_NAN = 0;`;
        static if(is(typeof({ mixin(enumMixinStr_FP_NAN); }))) {
            mixin(enumMixinStr_FP_NAN);
        }
    }






    static if(!is(typeof(__MATH_DECLARING_FLOATN))) {
        private enum enumMixinStr___MATH_DECLARING_FLOATN = `enum __MATH_DECLARING_FLOATN = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___MATH_DECLARING_FLOATN); }))) {
            mixin(enumMixinStr___MATH_DECLARING_FLOATN);
        }
    }




    static if(!is(typeof(__MATH_DECLARING_DOUBLE))) {
        private enum enumMixinStr___MATH_DECLARING_DOUBLE = `enum __MATH_DECLARING_DOUBLE = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___MATH_DECLARING_DOUBLE); }))) {
            mixin(enumMixinStr___MATH_DECLARING_DOUBLE);
        }
    }




    static if(!is(typeof(__MATH_PRECNAME))) {
        private enum enumMixinStr___MATH_PRECNAME = `enum __MATH_PRECNAME = ( name , r ) name ## f64x ## r;`;
        static if(is(typeof({ mixin(enumMixinStr___MATH_PRECNAME); }))) {
            mixin(enumMixinStr___MATH_PRECNAME);
        }
    }




    static if(!is(typeof(_Mdouble_))) {
        private enum enumMixinStr__Mdouble_ = `enum _Mdouble_ = _Float64x;`;
        static if(is(typeof({ mixin(enumMixinStr__Mdouble_); }))) {
            mixin(enumMixinStr__Mdouble_);
        }
    }




    static if(!is(typeof(__MATH_DECLARE_LDOUBLE))) {
        private enum enumMixinStr___MATH_DECLARE_LDOUBLE = `enum __MATH_DECLARE_LDOUBLE = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___MATH_DECLARE_LDOUBLE); }))) {
            mixin(enumMixinStr___MATH_DECLARE_LDOUBLE);
        }
    }




    static if(!is(typeof(__MATHDECL_1))) {
        private enum enumMixinStr___MATHDECL_1 = `enum __MATHDECL_1 = ( type , function , suffix , args ) extern type ( name , r ) namef64xr ( function , suffix ) args __THROW;`;
        static if(is(typeof({ mixin(enumMixinStr___MATHDECL_1); }))) {
            mixin(enumMixinStr___MATHDECL_1);
        }
    }
    static if(!is(typeof(__MATHDECL))) {
        private enum enumMixinStr___MATHDECL = `enum __MATHDECL = ( type , function , suffix , args ) ( type , function , suffix , args ) extern type ( name , r ) namef64xr ( function , suffix ) args __THROW ( type , function , suffix , args ) ; ( type , function , suffix , args ) extern type ( name , r ) namef64xr ( function , suffix ) args __THROW ( type , __CONCAT ( __ , function ) , suffix , args );`;
        static if(is(typeof({ mixin(enumMixinStr___MATHDECL); }))) {
            mixin(enumMixinStr___MATHDECL);
        }
    }




    static if(!is(typeof(__MATHCALL))) {
        private enum enumMixinStr___MATHCALL = `enum __MATHCALL = ( function , suffix , args ) ( type , function , suffix , args ) ( type , function , suffix , args ) extern type ( name , r ) namef64xr ( function , suffix ) args __THROW ( type , function , suffix , args ) ; ( type , function , suffix , args ) extern type ( name , r ) namef64xr ( function , suffix ) args __THROW ( type , __CONCAT ( __ , function ) , suffix , args ) ( _Float64x , function , suffix , args );`;
        static if(is(typeof({ mixin(enumMixinStr___MATHCALL); }))) {
            mixin(enumMixinStr___MATHCALL);
        }
    }
    static if(!is(typeof(FP_ILOGBNAN))) {
        private enum enumMixinStr_FP_ILOGBNAN = `enum FP_ILOGBNAN = ( - 2147483647 - 1 );`;
        static if(is(typeof({ mixin(enumMixinStr_FP_ILOGBNAN); }))) {
            mixin(enumMixinStr_FP_ILOGBNAN);
        }
    }




    static if(!is(typeof(FP_ILOGB0))) {
        private enum enumMixinStr_FP_ILOGB0 = `enum FP_ILOGB0 = ( - 2147483647 - 1 );`;
        static if(is(typeof({ mixin(enumMixinStr_FP_ILOGB0); }))) {
            mixin(enumMixinStr_FP_ILOGB0);
        }
    }




    static if(!is(typeof(NAN))) {
        private enum enumMixinStr_NAN = `enum NAN = ( __builtin_nanf ( "" ) );`;
        static if(is(typeof({ mixin(enumMixinStr_NAN); }))) {
            mixin(enumMixinStr_NAN);
        }
    }




    static if(!is(typeof(INFINITY))) {
        private enum enumMixinStr_INFINITY = `enum INFINITY = ( __builtin_inff ( ) );`;
        static if(is(typeof({ mixin(enumMixinStr_INFINITY); }))) {
            mixin(enumMixinStr_INFINITY);
        }
    }




    static if(!is(typeof(HUGE_VALL))) {
        private enum enumMixinStr_HUGE_VALL = `enum HUGE_VALL = ( __builtin_huge_vall ( ) );`;
        static if(is(typeof({ mixin(enumMixinStr_HUGE_VALL); }))) {
            mixin(enumMixinStr_HUGE_VALL);
        }
    }




    static if(!is(typeof(HUGE_VALF))) {
        private enum enumMixinStr_HUGE_VALF = `enum HUGE_VALF = ( __builtin_huge_valf ( ) );`;
        static if(is(typeof({ mixin(enumMixinStr_HUGE_VALF); }))) {
            mixin(enumMixinStr_HUGE_VALF);
        }
    }




    static if(!is(typeof(HUGE_VAL))) {
        private enum enumMixinStr_HUGE_VAL = `enum HUGE_VAL = ( __builtin_huge_val ( ) );`;
        static if(is(typeof({ mixin(enumMixinStr_HUGE_VAL); }))) {
            mixin(enumMixinStr_HUGE_VAL);
        }
    }






    static if(!is(typeof(_MATH_H))) {
        private enum enumMixinStr__MATH_H = `enum _MATH_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__MATH_H); }))) {
            mixin(enumMixinStr__MATH_H);
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




    static if(!is(typeof(cublasZtrmm))) {
        private enum enumMixinStr_cublasZtrmm = `enum cublasZtrmm = cublasZtrmm_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZtrmm); }))) {
            mixin(enumMixinStr_cublasZtrmm);
        }
    }




    static if(!is(typeof(cublasCtrmm))) {
        private enum enumMixinStr_cublasCtrmm = `enum cublasCtrmm = cublasCtrmm_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasCtrmm); }))) {
            mixin(enumMixinStr_cublasCtrmm);
        }
    }




    static if(!is(typeof(cublasDtrmm))) {
        private enum enumMixinStr_cublasDtrmm = `enum cublasDtrmm = cublasDtrmm_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDtrmm); }))) {
            mixin(enumMixinStr_cublasDtrmm);
        }
    }




    static if(!is(typeof(cublasStrmm))) {
        private enum enumMixinStr_cublasStrmm = `enum cublasStrmm = cublasStrmm_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasStrmm); }))) {
            mixin(enumMixinStr_cublasStrmm);
        }
    }




    static if(!is(typeof(_BITS_TYPESIZES_H))) {
        private enum enumMixinStr__BITS_TYPESIZES_H = `enum _BITS_TYPESIZES_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__BITS_TYPESIZES_H); }))) {
            mixin(enumMixinStr__BITS_TYPESIZES_H);
        }
    }




    static if(!is(typeof(cublasZtrsm))) {
        private enum enumMixinStr_cublasZtrsm = `enum cublasZtrsm = cublasZtrsm_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZtrsm); }))) {
            mixin(enumMixinStr_cublasZtrsm);
        }
    }




    static if(!is(typeof(__SYSCALL_SLONG_TYPE))) {
        private enum enumMixinStr___SYSCALL_SLONG_TYPE = `enum __SYSCALL_SLONG_TYPE = long int;`;
        static if(is(typeof({ mixin(enumMixinStr___SYSCALL_SLONG_TYPE); }))) {
            mixin(enumMixinStr___SYSCALL_SLONG_TYPE);
        }
    }




    static if(!is(typeof(__SYSCALL_ULONG_TYPE))) {
        private enum enumMixinStr___SYSCALL_ULONG_TYPE = `enum __SYSCALL_ULONG_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___SYSCALL_ULONG_TYPE); }))) {
            mixin(enumMixinStr___SYSCALL_ULONG_TYPE);
        }
    }




    static if(!is(typeof(__DEV_T_TYPE))) {
        private enum enumMixinStr___DEV_T_TYPE = `enum __DEV_T_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___DEV_T_TYPE); }))) {
            mixin(enumMixinStr___DEV_T_TYPE);
        }
    }




    static if(!is(typeof(__UID_T_TYPE))) {
        private enum enumMixinStr___UID_T_TYPE = `enum __UID_T_TYPE = unsigned int;`;
        static if(is(typeof({ mixin(enumMixinStr___UID_T_TYPE); }))) {
            mixin(enumMixinStr___UID_T_TYPE);
        }
    }




    static if(!is(typeof(__GID_T_TYPE))) {
        private enum enumMixinStr___GID_T_TYPE = `enum __GID_T_TYPE = unsigned int;`;
        static if(is(typeof({ mixin(enumMixinStr___GID_T_TYPE); }))) {
            mixin(enumMixinStr___GID_T_TYPE);
        }
    }




    static if(!is(typeof(__INO_T_TYPE))) {
        private enum enumMixinStr___INO_T_TYPE = `enum __INO_T_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___INO_T_TYPE); }))) {
            mixin(enumMixinStr___INO_T_TYPE);
        }
    }




    static if(!is(typeof(__INO64_T_TYPE))) {
        private enum enumMixinStr___INO64_T_TYPE = `enum __INO64_T_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___INO64_T_TYPE); }))) {
            mixin(enumMixinStr___INO64_T_TYPE);
        }
    }




    static if(!is(typeof(__MODE_T_TYPE))) {
        private enum enumMixinStr___MODE_T_TYPE = `enum __MODE_T_TYPE = unsigned int;`;
        static if(is(typeof({ mixin(enumMixinStr___MODE_T_TYPE); }))) {
            mixin(enumMixinStr___MODE_T_TYPE);
        }
    }




    static if(!is(typeof(cublasCtrsm))) {
        private enum enumMixinStr_cublasCtrsm = `enum cublasCtrsm = cublasCtrsm_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasCtrsm); }))) {
            mixin(enumMixinStr_cublasCtrsm);
        }
    }




    static if(!is(typeof(__NLINK_T_TYPE))) {
        private enum enumMixinStr___NLINK_T_TYPE = `enum __NLINK_T_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___NLINK_T_TYPE); }))) {
            mixin(enumMixinStr___NLINK_T_TYPE);
        }
    }




    static if(!is(typeof(__FSWORD_T_TYPE))) {
        private enum enumMixinStr___FSWORD_T_TYPE = `enum __FSWORD_T_TYPE = long int;`;
        static if(is(typeof({ mixin(enumMixinStr___FSWORD_T_TYPE); }))) {
            mixin(enumMixinStr___FSWORD_T_TYPE);
        }
    }




    static if(!is(typeof(__OFF_T_TYPE))) {
        private enum enumMixinStr___OFF_T_TYPE = `enum __OFF_T_TYPE = long int;`;
        static if(is(typeof({ mixin(enumMixinStr___OFF_T_TYPE); }))) {
            mixin(enumMixinStr___OFF_T_TYPE);
        }
    }




    static if(!is(typeof(__OFF64_T_TYPE))) {
        private enum enumMixinStr___OFF64_T_TYPE = `enum __OFF64_T_TYPE = long int;`;
        static if(is(typeof({ mixin(enumMixinStr___OFF64_T_TYPE); }))) {
            mixin(enumMixinStr___OFF64_T_TYPE);
        }
    }




    static if(!is(typeof(__PID_T_TYPE))) {
        private enum enumMixinStr___PID_T_TYPE = `enum __PID_T_TYPE = int;`;
        static if(is(typeof({ mixin(enumMixinStr___PID_T_TYPE); }))) {
            mixin(enumMixinStr___PID_T_TYPE);
        }
    }




    static if(!is(typeof(__RLIM_T_TYPE))) {
        private enum enumMixinStr___RLIM_T_TYPE = `enum __RLIM_T_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___RLIM_T_TYPE); }))) {
            mixin(enumMixinStr___RLIM_T_TYPE);
        }
    }




    static if(!is(typeof(__RLIM64_T_TYPE))) {
        private enum enumMixinStr___RLIM64_T_TYPE = `enum __RLIM64_T_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___RLIM64_T_TYPE); }))) {
            mixin(enumMixinStr___RLIM64_T_TYPE);
        }
    }




    static if(!is(typeof(__BLKCNT_T_TYPE))) {
        private enum enumMixinStr___BLKCNT_T_TYPE = `enum __BLKCNT_T_TYPE = long int;`;
        static if(is(typeof({ mixin(enumMixinStr___BLKCNT_T_TYPE); }))) {
            mixin(enumMixinStr___BLKCNT_T_TYPE);
        }
    }




    static if(!is(typeof(__BLKCNT64_T_TYPE))) {
        private enum enumMixinStr___BLKCNT64_T_TYPE = `enum __BLKCNT64_T_TYPE = long int;`;
        static if(is(typeof({ mixin(enumMixinStr___BLKCNT64_T_TYPE); }))) {
            mixin(enumMixinStr___BLKCNT64_T_TYPE);
        }
    }




    static if(!is(typeof(__FSBLKCNT_T_TYPE))) {
        private enum enumMixinStr___FSBLKCNT_T_TYPE = `enum __FSBLKCNT_T_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___FSBLKCNT_T_TYPE); }))) {
            mixin(enumMixinStr___FSBLKCNT_T_TYPE);
        }
    }




    static if(!is(typeof(__FSBLKCNT64_T_TYPE))) {
        private enum enumMixinStr___FSBLKCNT64_T_TYPE = `enum __FSBLKCNT64_T_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___FSBLKCNT64_T_TYPE); }))) {
            mixin(enumMixinStr___FSBLKCNT64_T_TYPE);
        }
    }




    static if(!is(typeof(__FSFILCNT_T_TYPE))) {
        private enum enumMixinStr___FSFILCNT_T_TYPE = `enum __FSFILCNT_T_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___FSFILCNT_T_TYPE); }))) {
            mixin(enumMixinStr___FSFILCNT_T_TYPE);
        }
    }




    static if(!is(typeof(__FSFILCNT64_T_TYPE))) {
        private enum enumMixinStr___FSFILCNT64_T_TYPE = `enum __FSFILCNT64_T_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___FSFILCNT64_T_TYPE); }))) {
            mixin(enumMixinStr___FSFILCNT64_T_TYPE);
        }
    }




    static if(!is(typeof(__ID_T_TYPE))) {
        private enum enumMixinStr___ID_T_TYPE = `enum __ID_T_TYPE = unsigned int;`;
        static if(is(typeof({ mixin(enumMixinStr___ID_T_TYPE); }))) {
            mixin(enumMixinStr___ID_T_TYPE);
        }
    }




    static if(!is(typeof(__CLOCK_T_TYPE))) {
        private enum enumMixinStr___CLOCK_T_TYPE = `enum __CLOCK_T_TYPE = long int;`;
        static if(is(typeof({ mixin(enumMixinStr___CLOCK_T_TYPE); }))) {
            mixin(enumMixinStr___CLOCK_T_TYPE);
        }
    }




    static if(!is(typeof(__TIME_T_TYPE))) {
        private enum enumMixinStr___TIME_T_TYPE = `enum __TIME_T_TYPE = long int;`;
        static if(is(typeof({ mixin(enumMixinStr___TIME_T_TYPE); }))) {
            mixin(enumMixinStr___TIME_T_TYPE);
        }
    }




    static if(!is(typeof(__USECONDS_T_TYPE))) {
        private enum enumMixinStr___USECONDS_T_TYPE = `enum __USECONDS_T_TYPE = unsigned int;`;
        static if(is(typeof({ mixin(enumMixinStr___USECONDS_T_TYPE); }))) {
            mixin(enumMixinStr___USECONDS_T_TYPE);
        }
    }




    static if(!is(typeof(__SUSECONDS_T_TYPE))) {
        private enum enumMixinStr___SUSECONDS_T_TYPE = `enum __SUSECONDS_T_TYPE = long int;`;
        static if(is(typeof({ mixin(enumMixinStr___SUSECONDS_T_TYPE); }))) {
            mixin(enumMixinStr___SUSECONDS_T_TYPE);
        }
    }




    static if(!is(typeof(__DADDR_T_TYPE))) {
        private enum enumMixinStr___DADDR_T_TYPE = `enum __DADDR_T_TYPE = int;`;
        static if(is(typeof({ mixin(enumMixinStr___DADDR_T_TYPE); }))) {
            mixin(enumMixinStr___DADDR_T_TYPE);
        }
    }




    static if(!is(typeof(__KEY_T_TYPE))) {
        private enum enumMixinStr___KEY_T_TYPE = `enum __KEY_T_TYPE = int;`;
        static if(is(typeof({ mixin(enumMixinStr___KEY_T_TYPE); }))) {
            mixin(enumMixinStr___KEY_T_TYPE);
        }
    }




    static if(!is(typeof(__CLOCKID_T_TYPE))) {
        private enum enumMixinStr___CLOCKID_T_TYPE = `enum __CLOCKID_T_TYPE = int;`;
        static if(is(typeof({ mixin(enumMixinStr___CLOCKID_T_TYPE); }))) {
            mixin(enumMixinStr___CLOCKID_T_TYPE);
        }
    }




    static if(!is(typeof(__TIMER_T_TYPE))) {
        private enum enumMixinStr___TIMER_T_TYPE = `enum __TIMER_T_TYPE = void *;`;
        static if(is(typeof({ mixin(enumMixinStr___TIMER_T_TYPE); }))) {
            mixin(enumMixinStr___TIMER_T_TYPE);
        }
    }




    static if(!is(typeof(__BLKSIZE_T_TYPE))) {
        private enum enumMixinStr___BLKSIZE_T_TYPE = `enum __BLKSIZE_T_TYPE = long int;`;
        static if(is(typeof({ mixin(enumMixinStr___BLKSIZE_T_TYPE); }))) {
            mixin(enumMixinStr___BLKSIZE_T_TYPE);
        }
    }




    static if(!is(typeof(__FSID_T_TYPE))) {
        private enum enumMixinStr___FSID_T_TYPE = `enum __FSID_T_TYPE = { int __val [ 2 ] ; };`;
        static if(is(typeof({ mixin(enumMixinStr___FSID_T_TYPE); }))) {
            mixin(enumMixinStr___FSID_T_TYPE);
        }
    }




    static if(!is(typeof(__SSIZE_T_TYPE))) {
        private enum enumMixinStr___SSIZE_T_TYPE = `enum __SSIZE_T_TYPE = long int;`;
        static if(is(typeof({ mixin(enumMixinStr___SSIZE_T_TYPE); }))) {
            mixin(enumMixinStr___SSIZE_T_TYPE);
        }
    }




    static if(!is(typeof(__CPU_MASK_TYPE))) {
        private enum enumMixinStr___CPU_MASK_TYPE = `enum __CPU_MASK_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___CPU_MASK_TYPE); }))) {
            mixin(enumMixinStr___CPU_MASK_TYPE);
        }
    }




    static if(!is(typeof(cublasDtrsm))) {
        private enum enumMixinStr_cublasDtrsm = `enum cublasDtrsm = cublasDtrsm_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDtrsm); }))) {
            mixin(enumMixinStr_cublasDtrsm);
        }
    }




    static if(!is(typeof(__OFF_T_MATCHES_OFF64_T))) {
        private enum enumMixinStr___OFF_T_MATCHES_OFF64_T = `enum __OFF_T_MATCHES_OFF64_T = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___OFF_T_MATCHES_OFF64_T); }))) {
            mixin(enumMixinStr___OFF_T_MATCHES_OFF64_T);
        }
    }




    static if(!is(typeof(__INO_T_MATCHES_INO64_T))) {
        private enum enumMixinStr___INO_T_MATCHES_INO64_T = `enum __INO_T_MATCHES_INO64_T = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___INO_T_MATCHES_INO64_T); }))) {
            mixin(enumMixinStr___INO_T_MATCHES_INO64_T);
        }
    }




    static if(!is(typeof(__RLIM_T_MATCHES_RLIM64_T))) {
        private enum enumMixinStr___RLIM_T_MATCHES_RLIM64_T = `enum __RLIM_T_MATCHES_RLIM64_T = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___RLIM_T_MATCHES_RLIM64_T); }))) {
            mixin(enumMixinStr___RLIM_T_MATCHES_RLIM64_T);
        }
    }




    static if(!is(typeof(__FD_SETSIZE))) {
        private enum enumMixinStr___FD_SETSIZE = `enum __FD_SETSIZE = 1024;`;
        static if(is(typeof({ mixin(enumMixinStr___FD_SETSIZE); }))) {
            mixin(enumMixinStr___FD_SETSIZE);
        }
    }




    static if(!is(typeof(cublasStrsm))) {
        private enum enumMixinStr_cublasStrsm = `enum cublasStrsm = cublasStrsm_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasStrsm); }))) {
            mixin(enumMixinStr_cublasStrsm);
        }
    }




    static if(!is(typeof(cublasZhemm))) {
        private enum enumMixinStr_cublasZhemm = `enum cublasZhemm = cublasZhemm_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZhemm); }))) {
            mixin(enumMixinStr_cublasZhemm);
        }
    }




    static if(!is(typeof(__WORDSIZE))) {
        private enum enumMixinStr___WORDSIZE = `enum __WORDSIZE = 64;`;
        static if(is(typeof({ mixin(enumMixinStr___WORDSIZE); }))) {
            mixin(enumMixinStr___WORDSIZE);
        }
    }




    static if(!is(typeof(cublasChemm))) {
        private enum enumMixinStr_cublasChemm = `enum cublasChemm = cublasChemm_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasChemm); }))) {
            mixin(enumMixinStr_cublasChemm);
        }
    }




    static if(!is(typeof(cublasZsymm))) {
        private enum enumMixinStr_cublasZsymm = `enum cublasZsymm = cublasZsymm_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZsymm); }))) {
            mixin(enumMixinStr_cublasZsymm);
        }
    }




    static if(!is(typeof(__WORDSIZE_TIME64_COMPAT32))) {
        private enum enumMixinStr___WORDSIZE_TIME64_COMPAT32 = `enum __WORDSIZE_TIME64_COMPAT32 = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___WORDSIZE_TIME64_COMPAT32); }))) {
            mixin(enumMixinStr___WORDSIZE_TIME64_COMPAT32);
        }
    }




    static if(!is(typeof(__SYSCALL_WORDSIZE))) {
        private enum enumMixinStr___SYSCALL_WORDSIZE = `enum __SYSCALL_WORDSIZE = 64;`;
        static if(is(typeof({ mixin(enumMixinStr___SYSCALL_WORDSIZE); }))) {
            mixin(enumMixinStr___SYSCALL_WORDSIZE);
        }
    }
    static if(!is(typeof(cublasCsymm))) {
        private enum enumMixinStr_cublasCsymm = `enum cublasCsymm = cublasCsymm_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasCsymm); }))) {
            mixin(enumMixinStr_cublasCsymm);
        }
    }




    static if(!is(typeof(cublasDsymm))) {
        private enum enumMixinStr_cublasDsymm = `enum cublasDsymm = cublasDsymm_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDsymm); }))) {
            mixin(enumMixinStr_cublasDsymm);
        }
    }




    static if(!is(typeof(cublasSsymm))) {
        private enum enumMixinStr_cublasSsymm = `enum cublasSsymm = cublasSsymm_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasSsymm); }))) {
            mixin(enumMixinStr_cublasSsymm);
        }
    }




    static if(!is(typeof(cublasZher2k))) {
        private enum enumMixinStr_cublasZher2k = `enum cublasZher2k = cublasZher2k_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZher2k); }))) {
            mixin(enumMixinStr_cublasZher2k);
        }
    }




    static if(!is(typeof(cublasCher2k))) {
        private enum enumMixinStr_cublasCher2k = `enum cublasCher2k = cublasCher2k_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasCher2k); }))) {
            mixin(enumMixinStr_cublasCher2k);
        }
    }




    static if(!is(typeof(_SYS_CDEFS_H))) {
        private enum enumMixinStr__SYS_CDEFS_H = `enum _SYS_CDEFS_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__SYS_CDEFS_H); }))) {
            mixin(enumMixinStr__SYS_CDEFS_H);
        }
    }




    static if(!is(typeof(cublasZsyr2k))) {
        private enum enumMixinStr_cublasZsyr2k = `enum cublasZsyr2k = cublasZsyr2k_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZsyr2k); }))) {
            mixin(enumMixinStr_cublasZsyr2k);
        }
    }




    static if(!is(typeof(cublasCsyr2k))) {
        private enum enumMixinStr_cublasCsyr2k = `enum cublasCsyr2k = cublasCsyr2k_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasCsyr2k); }))) {
            mixin(enumMixinStr_cublasCsyr2k);
        }
    }




    static if(!is(typeof(cublasDsyr2k))) {
        private enum enumMixinStr_cublasDsyr2k = `enum cublasDsyr2k = cublasDsyr2k_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDsyr2k); }))) {
            mixin(enumMixinStr_cublasDsyr2k);
        }
    }




    static if(!is(typeof(cublasSsyr2k))) {
        private enum enumMixinStr_cublasSsyr2k = `enum cublasSsyr2k = cublasSsyr2k_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasSsyr2k); }))) {
            mixin(enumMixinStr_cublasSsyr2k);
        }
    }




    static if(!is(typeof(cublasZherk))) {
        private enum enumMixinStr_cublasZherk = `enum cublasZherk = cublasZherk_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZherk); }))) {
            mixin(enumMixinStr_cublasZherk);
        }
    }
    static if(!is(typeof(cublasCherk))) {
        private enum enumMixinStr_cublasCherk = `enum cublasCherk = cublasCherk_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasCherk); }))) {
            mixin(enumMixinStr_cublasCherk);
        }
    }




    static if(!is(typeof(__THROW))) {
        private enum enumMixinStr___THROW = `enum __THROW = __attribute__ ( ( __nothrow__ ) );`;
        static if(is(typeof({ mixin(enumMixinStr___THROW); }))) {
            mixin(enumMixinStr___THROW);
        }
    }




    static if(!is(typeof(__THROWNL))) {
        private enum enumMixinStr___THROWNL = `enum __THROWNL = __attribute__ ( ( __nothrow__ ) );`;
        static if(is(typeof({ mixin(enumMixinStr___THROWNL); }))) {
            mixin(enumMixinStr___THROWNL);
        }
    }
    static if(!is(typeof(cublasZsyrk))) {
        private enum enumMixinStr_cublasZsyrk = `enum cublasZsyrk = cublasZsyrk_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZsyrk); }))) {
            mixin(enumMixinStr_cublasZsyrk);
        }
    }




    static if(!is(typeof(cublasCsyrk))) {
        private enum enumMixinStr_cublasCsyrk = `enum cublasCsyrk = cublasCsyrk_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasCsyrk); }))) {
            mixin(enumMixinStr_cublasCsyrk);
        }
    }
    static if(!is(typeof(__ptr_t))) {
        private enum enumMixinStr___ptr_t = `enum __ptr_t = void *;`;
        static if(is(typeof({ mixin(enumMixinStr___ptr_t); }))) {
            mixin(enumMixinStr___ptr_t);
        }
    }
    static if(!is(typeof(cublasDsyrk))) {
        private enum enumMixinStr_cublasDsyrk = `enum cublasDsyrk = cublasDsyrk_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDsyrk); }))) {
            mixin(enumMixinStr_cublasDsyrk);
        }
    }
    static if(!is(typeof(cublasSsyrk))) {
        private enum enumMixinStr_cublasSsyrk = `enum cublasSsyrk = cublasSsyrk_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasSsyrk); }))) {
            mixin(enumMixinStr_cublasSsyrk);
        }
    }




    static if(!is(typeof(cublasZgemm))) {
        private enum enumMixinStr_cublasZgemm = `enum cublasZgemm = cublasZgemm_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZgemm); }))) {
            mixin(enumMixinStr_cublasZgemm);
        }
    }




    static if(!is(typeof(__flexarr))) {
        private enum enumMixinStr___flexarr = `enum __flexarr = [ ];`;
        static if(is(typeof({ mixin(enumMixinStr___flexarr); }))) {
            mixin(enumMixinStr___flexarr);
        }
    }




    static if(!is(typeof(__glibc_c99_flexarr_available))) {
        private enum enumMixinStr___glibc_c99_flexarr_available = `enum __glibc_c99_flexarr_available = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___glibc_c99_flexarr_available); }))) {
            mixin(enumMixinStr___glibc_c99_flexarr_available);
        }
    }




    static if(!is(typeof(cublasCgemm))) {
        private enum enumMixinStr_cublasCgemm = `enum cublasCgemm = cublasCgemm_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasCgemm); }))) {
            mixin(enumMixinStr_cublasCgemm);
        }
    }




    static if(!is(typeof(cublasDgemm))) {
        private enum enumMixinStr_cublasDgemm = `enum cublasDgemm = cublasDgemm_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDgemm); }))) {
            mixin(enumMixinStr_cublasDgemm);
        }
    }
    static if(!is(typeof(cublasSgemm))) {
        private enum enumMixinStr_cublasSgemm = `enum cublasSgemm = cublasSgemm_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasSgemm); }))) {
            mixin(enumMixinStr_cublasSgemm);
        }
    }




    static if(!is(typeof(cublasZhpr2))) {
        private enum enumMixinStr_cublasZhpr2 = `enum cublasZhpr2 = cublasZhpr2_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZhpr2); }))) {
            mixin(enumMixinStr_cublasZhpr2);
        }
    }




    static if(!is(typeof(cublasChpr2))) {
        private enum enumMixinStr_cublasChpr2 = `enum cublasChpr2 = cublasChpr2_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasChpr2); }))) {
            mixin(enumMixinStr_cublasChpr2);
        }
    }




    static if(!is(typeof(__attribute_malloc__))) {
        private enum enumMixinStr___attribute_malloc__ = `enum __attribute_malloc__ = __attribute__ ( ( __malloc__ ) );`;
        static if(is(typeof({ mixin(enumMixinStr___attribute_malloc__); }))) {
            mixin(enumMixinStr___attribute_malloc__);
        }
    }




    static if(!is(typeof(cublasDspr2))) {
        private enum enumMixinStr_cublasDspr2 = `enum cublasDspr2 = cublasDspr2_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDspr2); }))) {
            mixin(enumMixinStr_cublasDspr2);
        }
    }






    static if(!is(typeof(cublasSspr2))) {
        private enum enumMixinStr_cublasSspr2 = `enum cublasSspr2 = cublasSspr2_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasSspr2); }))) {
            mixin(enumMixinStr_cublasSspr2);
        }
    }




    static if(!is(typeof(__attribute_pure__))) {
        private enum enumMixinStr___attribute_pure__ = `enum __attribute_pure__ = __attribute__ ( ( __pure__ ) );`;
        static if(is(typeof({ mixin(enumMixinStr___attribute_pure__); }))) {
            mixin(enumMixinStr___attribute_pure__);
        }
    }




    static if(!is(typeof(cublasZher2))) {
        private enum enumMixinStr_cublasZher2 = `enum cublasZher2 = cublasZher2_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZher2); }))) {
            mixin(enumMixinStr_cublasZher2);
        }
    }




    static if(!is(typeof(__attribute_const__))) {
        private enum enumMixinStr___attribute_const__ = `enum __attribute_const__ = __attribute__ ( cast( __const__ ) );`;
        static if(is(typeof({ mixin(enumMixinStr___attribute_const__); }))) {
            mixin(enumMixinStr___attribute_const__);
        }
    }




    static if(!is(typeof(cublasCher2))) {
        private enum enumMixinStr_cublasCher2 = `enum cublasCher2 = cublasCher2_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasCher2); }))) {
            mixin(enumMixinStr_cublasCher2);
        }
    }




    static if(!is(typeof(__attribute_used__))) {
        private enum enumMixinStr___attribute_used__ = `enum __attribute_used__ = __attribute__ ( ( __used__ ) );`;
        static if(is(typeof({ mixin(enumMixinStr___attribute_used__); }))) {
            mixin(enumMixinStr___attribute_used__);
        }
    }




    static if(!is(typeof(__attribute_noinline__))) {
        private enum enumMixinStr___attribute_noinline__ = `enum __attribute_noinline__ = __attribute__ ( ( __noinline__ ) );`;
        static if(is(typeof({ mixin(enumMixinStr___attribute_noinline__); }))) {
            mixin(enumMixinStr___attribute_noinline__);
        }
    }




    static if(!is(typeof(cublasZsyr2))) {
        private enum enumMixinStr_cublasZsyr2 = `enum cublasZsyr2 = cublasZsyr2_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZsyr2); }))) {
            mixin(enumMixinStr_cublasZsyr2);
        }
    }




    static if(!is(typeof(__attribute_deprecated__))) {
        private enum enumMixinStr___attribute_deprecated__ = `enum __attribute_deprecated__ = __attribute__ ( ( __deprecated__ ) );`;
        static if(is(typeof({ mixin(enumMixinStr___attribute_deprecated__); }))) {
            mixin(enumMixinStr___attribute_deprecated__);
        }
    }




    static if(!is(typeof(cublasCsyr2))) {
        private enum enumMixinStr_cublasCsyr2 = `enum cublasCsyr2 = cublasCsyr2_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasCsyr2); }))) {
            mixin(enumMixinStr_cublasCsyr2);
        }
    }




    static if(!is(typeof(cublasDsyr2))) {
        private enum enumMixinStr_cublasDsyr2 = `enum cublasDsyr2 = cublasDsyr2_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDsyr2); }))) {
            mixin(enumMixinStr_cublasDsyr2);
        }
    }






    static if(!is(typeof(cublasSsyr2))) {
        private enum enumMixinStr_cublasSsyr2 = `enum cublasSsyr2 = cublasSsyr2_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasSsyr2); }))) {
            mixin(enumMixinStr_cublasSsyr2);
        }
    }






    static if(!is(typeof(cublasZhpr))) {
        private enum enumMixinStr_cublasZhpr = `enum cublasZhpr = cublasZhpr_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZhpr); }))) {
            mixin(enumMixinStr_cublasZhpr);
        }
    }






    static if(!is(typeof(cublasChpr))) {
        private enum enumMixinStr_cublasChpr = `enum cublasChpr = cublasChpr_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasChpr); }))) {
            mixin(enumMixinStr_cublasChpr);
        }
    }






    static if(!is(typeof(cublasDspr))) {
        private enum enumMixinStr_cublasDspr = `enum cublasDspr = cublasDspr_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDspr); }))) {
            mixin(enumMixinStr_cublasDspr);
        }
    }




    static if(!is(typeof(__attribute_warn_unused_result__))) {
        private enum enumMixinStr___attribute_warn_unused_result__ = `enum __attribute_warn_unused_result__ = __attribute__ ( ( __warn_unused_result__ ) );`;
        static if(is(typeof({ mixin(enumMixinStr___attribute_warn_unused_result__); }))) {
            mixin(enumMixinStr___attribute_warn_unused_result__);
        }
    }




    static if(!is(typeof(cublasSspr))) {
        private enum enumMixinStr_cublasSspr = `enum cublasSspr = cublasSspr_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasSspr); }))) {
            mixin(enumMixinStr_cublasSspr);
        }
    }






    static if(!is(typeof(cublasZher))) {
        private enum enumMixinStr_cublasZher = `enum cublasZher = cublasZher_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZher); }))) {
            mixin(enumMixinStr_cublasZher);
        }
    }




    static if(!is(typeof(__always_inline))) {
        private enum enumMixinStr___always_inline = `enum __always_inline = __inline __attribute__ ( ( __always_inline__ ) );`;
        static if(is(typeof({ mixin(enumMixinStr___always_inline); }))) {
            mixin(enumMixinStr___always_inline);
        }
    }




    static if(!is(typeof(cublasCher))) {
        private enum enumMixinStr_cublasCher = `enum cublasCher = cublasCher_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasCher); }))) {
            mixin(enumMixinStr_cublasCher);
        }
    }






    static if(!is(typeof(cublasZsyr))) {
        private enum enumMixinStr_cublasZsyr = `enum cublasZsyr = cublasZsyr_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZsyr); }))) {
            mixin(enumMixinStr_cublasZsyr);
        }
    }




    static if(!is(typeof(cublasCsyr))) {
        private enum enumMixinStr_cublasCsyr = `enum cublasCsyr = cublasCsyr_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasCsyr); }))) {
            mixin(enumMixinStr_cublasCsyr);
        }
    }




    static if(!is(typeof(cublasDsyr))) {
        private enum enumMixinStr_cublasDsyr = `enum cublasDsyr = cublasDsyr_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDsyr); }))) {
            mixin(enumMixinStr_cublasDsyr);
        }
    }




    static if(!is(typeof(cublasSsyr))) {
        private enum enumMixinStr_cublasSsyr = `enum cublasSsyr = cublasSsyr_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasSsyr); }))) {
            mixin(enumMixinStr_cublasSsyr);
        }
    }




    static if(!is(typeof(__extern_inline))) {
        private enum enumMixinStr___extern_inline = `enum __extern_inline = extern __inline __attribute__ ( ( __gnu_inline__ ) );`;
        static if(is(typeof({ mixin(enumMixinStr___extern_inline); }))) {
            mixin(enumMixinStr___extern_inline);
        }
    }




    static if(!is(typeof(__extern_always_inline))) {
        private enum enumMixinStr___extern_always_inline = `enum __extern_always_inline = extern __inline __attribute__ ( ( __always_inline__ ) ) __attribute__ ( ( __gnu_inline__ ) );`;
        static if(is(typeof({ mixin(enumMixinStr___extern_always_inline); }))) {
            mixin(enumMixinStr___extern_always_inline);
        }
    }




    static if(!is(typeof(cublasZgerc))) {
        private enum enumMixinStr_cublasZgerc = `enum cublasZgerc = cublasZgerc_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZgerc); }))) {
            mixin(enumMixinStr_cublasZgerc);
        }
    }




    static if(!is(typeof(__fortify_function))) {
        private enum enumMixinStr___fortify_function = `enum __fortify_function = extern __inline __attribute__ ( ( __always_inline__ ) ) __attribute__ ( ( __gnu_inline__ ) ) ;`;
        static if(is(typeof({ mixin(enumMixinStr___fortify_function); }))) {
            mixin(enumMixinStr___fortify_function);
        }
    }




    static if(!is(typeof(cublasZgeru))) {
        private enum enumMixinStr_cublasZgeru = `enum cublasZgeru = cublasZgeru_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZgeru); }))) {
            mixin(enumMixinStr_cublasZgeru);
        }
    }




    static if(!is(typeof(cublasCgerc))) {
        private enum enumMixinStr_cublasCgerc = `enum cublasCgerc = cublasCgerc_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasCgerc); }))) {
            mixin(enumMixinStr_cublasCgerc);
        }
    }




    static if(!is(typeof(cublasCgeru))) {
        private enum enumMixinStr_cublasCgeru = `enum cublasCgeru = cublasCgeru_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasCgeru); }))) {
            mixin(enumMixinStr_cublasCgeru);
        }
    }




    static if(!is(typeof(cublasDger))) {
        private enum enumMixinStr_cublasDger = `enum cublasDger = cublasDger_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDger); }))) {
            mixin(enumMixinStr_cublasDger);
        }
    }




    static if(!is(typeof(__restrict_arr))) {
        private enum enumMixinStr___restrict_arr = `enum __restrict_arr = __restrict;`;
        static if(is(typeof({ mixin(enumMixinStr___restrict_arr); }))) {
            mixin(enumMixinStr___restrict_arr);
        }
    }




    static if(!is(typeof(cublasSger))) {
        private enum enumMixinStr_cublasSger = `enum cublasSger = cublasSger_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasSger); }))) {
            mixin(enumMixinStr_cublasSger);
        }
    }
    static if(!is(typeof(cublasZhpmv))) {
        private enum enumMixinStr_cublasZhpmv = `enum cublasZhpmv = cublasZhpmv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZhpmv); }))) {
            mixin(enumMixinStr_cublasZhpmv);
        }
    }




    static if(!is(typeof(cublasChpmv))) {
        private enum enumMixinStr_cublasChpmv = `enum cublasChpmv = cublasChpmv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasChpmv); }))) {
            mixin(enumMixinStr_cublasChpmv);
        }
    }




    static if(!is(typeof(cublasDspmv))) {
        private enum enumMixinStr_cublasDspmv = `enum cublasDspmv = cublasDspmv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDspmv); }))) {
            mixin(enumMixinStr_cublasDspmv);
        }
    }




    static if(!is(typeof(cublasSspmv))) {
        private enum enumMixinStr_cublasSspmv = `enum cublasSspmv = cublasSspmv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasSspmv); }))) {
            mixin(enumMixinStr_cublasSspmv);
        }
    }






    static if(!is(typeof(cublasZhbmv))) {
        private enum enumMixinStr_cublasZhbmv = `enum cublasZhbmv = cublasZhbmv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZhbmv); }))) {
            mixin(enumMixinStr_cublasZhbmv);
        }
    }




    static if(!is(typeof(cublasChbmv))) {
        private enum enumMixinStr_cublasChbmv = `enum cublasChbmv = cublasChbmv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasChbmv); }))) {
            mixin(enumMixinStr_cublasChbmv);
        }
    }




    static if(!is(typeof(cublasDsbmv))) {
        private enum enumMixinStr_cublasDsbmv = `enum cublasDsbmv = cublasDsbmv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDsbmv); }))) {
            mixin(enumMixinStr_cublasDsbmv);
        }
    }




    static if(!is(typeof(cublasSsbmv))) {
        private enum enumMixinStr_cublasSsbmv = `enum cublasSsbmv = cublasSsbmv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasSsbmv); }))) {
            mixin(enumMixinStr_cublasSsbmv);
        }
    }




    static if(!is(typeof(cublasZhemv))) {
        private enum enumMixinStr_cublasZhemv = `enum cublasZhemv = cublasZhemv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZhemv); }))) {
            mixin(enumMixinStr_cublasZhemv);
        }
    }




    static if(!is(typeof(cublasChemv))) {
        private enum enumMixinStr_cublasChemv = `enum cublasChemv = cublasChemv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasChemv); }))) {
            mixin(enumMixinStr_cublasChemv);
        }
    }
    static if(!is(typeof(cublasZsymv))) {
        private enum enumMixinStr_cublasZsymv = `enum cublasZsymv = cublasZsymv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZsymv); }))) {
            mixin(enumMixinStr_cublasZsymv);
        }
    }
    static if(!is(typeof(cublasCsymv))) {
        private enum enumMixinStr_cublasCsymv = `enum cublasCsymv = cublasCsymv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasCsymv); }))) {
            mixin(enumMixinStr_cublasCsymv);
        }
    }




    static if(!is(typeof(cublasDsymv))) {
        private enum enumMixinStr_cublasDsymv = `enum cublasDsymv = cublasDsymv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDsymv); }))) {
            mixin(enumMixinStr_cublasDsymv);
        }
    }
    static if(!is(typeof(cublasSsymv))) {
        private enum enumMixinStr_cublasSsymv = `enum cublasSsymv = cublasSsymv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasSsymv); }))) {
            mixin(enumMixinStr_cublasSsymv);
        }
    }




    static if(!is(typeof(cublasZtbsv))) {
        private enum enumMixinStr_cublasZtbsv = `enum cublasZtbsv = cublasZtbsv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZtbsv); }))) {
            mixin(enumMixinStr_cublasZtbsv);
        }
    }




    static if(!is(typeof(cublasCtbsv))) {
        private enum enumMixinStr_cublasCtbsv = `enum cublasCtbsv = cublasCtbsv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasCtbsv); }))) {
            mixin(enumMixinStr_cublasCtbsv);
        }
    }




    static if(!is(typeof(cublasDtbsv))) {
        private enum enumMixinStr_cublasDtbsv = `enum cublasDtbsv = cublasDtbsv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDtbsv); }))) {
            mixin(enumMixinStr_cublasDtbsv);
        }
    }




    static if(!is(typeof(cublasStbsv))) {
        private enum enumMixinStr_cublasStbsv = `enum cublasStbsv = cublasStbsv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasStbsv); }))) {
            mixin(enumMixinStr_cublasStbsv);
        }
    }




    static if(!is(typeof(__HAVE_GENERIC_SELECTION))) {
        private enum enumMixinStr___HAVE_GENERIC_SELECTION = `enum __HAVE_GENERIC_SELECTION = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_GENERIC_SELECTION); }))) {
            mixin(enumMixinStr___HAVE_GENERIC_SELECTION);
        }
    }




    static if(!is(typeof(cublasZtpsv))) {
        private enum enumMixinStr_cublasZtpsv = `enum cublasZtpsv = cublasZtpsv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZtpsv); }))) {
            mixin(enumMixinStr_cublasZtpsv);
        }
    }




    static if(!is(typeof(cublasCtpsv))) {
        private enum enumMixinStr_cublasCtpsv = `enum cublasCtpsv = cublasCtpsv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasCtpsv); }))) {
            mixin(enumMixinStr_cublasCtpsv);
        }
    }




    static if(!is(typeof(cublasDtpsv))) {
        private enum enumMixinStr_cublasDtpsv = `enum cublasDtpsv = cublasDtpsv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDtpsv); }))) {
            mixin(enumMixinStr_cublasDtpsv);
        }
    }






    static if(!is(typeof(cublasStpsv))) {
        private enum enumMixinStr_cublasStpsv = `enum cublasStpsv = cublasStpsv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasStpsv); }))) {
            mixin(enumMixinStr_cublasStpsv);
        }
    }




    static if(!is(typeof(cublasZtrsv))) {
        private enum enumMixinStr_cublasZtrsv = `enum cublasZtrsv = cublasZtrsv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZtrsv); }))) {
            mixin(enumMixinStr_cublasZtrsv);
        }
    }




    static if(!is(typeof(__no_return__))) {
        private enum enumMixinStr___no_return__ = `enum __no_return__ = __attribute__ ( ( noreturn ) );`;
        static if(is(typeof({ mixin(enumMixinStr___no_return__); }))) {
            mixin(enumMixinStr___no_return__);
        }
    }




    static if(!is(typeof(__forceinline__))) {
        private enum enumMixinStr___forceinline__ = `enum __forceinline__ = __inline__ __attribute__ ( ( always_inline ) );`;
        static if(is(typeof({ mixin(enumMixinStr___forceinline__); }))) {
            mixin(enumMixinStr___forceinline__);
        }
    }






    static if(!is(typeof(__thread__))) {
        private enum enumMixinStr___thread__ = `enum __thread__ = __thread;`;
        static if(is(typeof({ mixin(enumMixinStr___thread__); }))) {
            mixin(enumMixinStr___thread__);
        }
    }
    static if(!is(typeof(cublasCtrsv))) {
        private enum enumMixinStr_cublasCtrsv = `enum cublasCtrsv = cublasCtrsv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasCtrsv); }))) {
            mixin(enumMixinStr_cublasCtrsv);
        }
    }




    static if(!is(typeof(cublasDtrsv))) {
        private enum enumMixinStr_cublasDtrsv = `enum cublasDtrsv = cublasDtrsv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDtrsv); }))) {
            mixin(enumMixinStr_cublasDtrsv);
        }
    }




    static if(!is(typeof(cublasStrsv))) {
        private enum enumMixinStr_cublasStrsv = `enum cublasStrsv = cublasStrsv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasStrsv); }))) {
            mixin(enumMixinStr_cublasStrsv);
        }
    }




    static if(!is(typeof(cublasZtpmv))) {
        private enum enumMixinStr_cublasZtpmv = `enum cublasZtpmv = cublasZtpmv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZtpmv); }))) {
            mixin(enumMixinStr_cublasZtpmv);
        }
    }




    static if(!is(typeof(cublasCtpmv))) {
        private enum enumMixinStr_cublasCtpmv = `enum cublasCtpmv = cublasCtpmv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasCtpmv); }))) {
            mixin(enumMixinStr_cublasCtpmv);
        }
    }




    static if(!is(typeof(cublasDtpmv))) {
        private enum enumMixinStr_cublasDtpmv = `enum cublasDtpmv = cublasDtpmv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDtpmv); }))) {
            mixin(enumMixinStr_cublasDtpmv);
        }
    }






    static if(!is(typeof(cublasStpmv))) {
        private enum enumMixinStr_cublasStpmv = `enum cublasStpmv = cublasStpmv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasStpmv); }))) {
            mixin(enumMixinStr_cublasStpmv);
        }
    }






    static if(!is(typeof(__host__))) {
        private enum enumMixinStr___host__ = `enum __host__ = __attribute__ ( ( host ) );`;
        static if(is(typeof({ mixin(enumMixinStr___host__); }))) {
            mixin(enumMixinStr___host__);
        }
    }




    static if(!is(typeof(__device__))) {
        private enum enumMixinStr___device__ = `enum __device__ = __attribute__ ( ( device ) );`;
        static if(is(typeof({ mixin(enumMixinStr___device__); }))) {
            mixin(enumMixinStr___device__);
        }
    }




    static if(!is(typeof(__global__))) {
        private enum enumMixinStr___global__ = `enum __global__ = __attribute__ ( ( global ) );`;
        static if(is(typeof({ mixin(enumMixinStr___global__); }))) {
            mixin(enumMixinStr___global__);
        }
    }




    static if(!is(typeof(__shared__))) {
        private enum enumMixinStr___shared__ = `enum __shared__ = __attribute__ ( ( shared ) );`;
        static if(is(typeof({ mixin(enumMixinStr___shared__); }))) {
            mixin(enumMixinStr___shared__);
        }
    }




    static if(!is(typeof(__constant__))) {
        private enum enumMixinStr___constant__ = `enum __constant__ = __attribute__ ( ( constant ) );`;
        static if(is(typeof({ mixin(enumMixinStr___constant__); }))) {
            mixin(enumMixinStr___constant__);
        }
    }




    static if(!is(typeof(__managed__))) {
        private enum enumMixinStr___managed__ = `enum __managed__ = __attribute__ ( ( managed ) );`;
        static if(is(typeof({ mixin(enumMixinStr___managed__); }))) {
            mixin(enumMixinStr___managed__);
        }
    }
    static if(!is(typeof(cublasZtbmv))) {
        private enum enumMixinStr_cublasZtbmv = `enum cublasZtbmv = cublasZtbmv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZtbmv); }))) {
            mixin(enumMixinStr_cublasZtbmv);
        }
    }




    static if(!is(typeof(cublasCtbmv))) {
        private enum enumMixinStr_cublasCtbmv = `enum cublasCtbmv = cublasCtbmv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasCtbmv); }))) {
            mixin(enumMixinStr_cublasCtbmv);
        }
    }




    static if(!is(typeof(cublasDtbmv))) {
        private enum enumMixinStr_cublasDtbmv = `enum cublasDtbmv = cublasDtbmv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDtbmv); }))) {
            mixin(enumMixinStr_cublasDtbmv);
        }
    }




    static if(!is(typeof(cublasStbmv))) {
        private enum enumMixinStr_cublasStbmv = `enum cublasStbmv = cublasStbmv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasStbmv); }))) {
            mixin(enumMixinStr_cublasStbmv);
        }
    }




    static if(!is(typeof(cublasZtrmv))) {
        private enum enumMixinStr_cublasZtrmv = `enum cublasZtrmv = cublasZtrmv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZtrmv); }))) {
            mixin(enumMixinStr_cublasZtrmv);
        }
    }




    static if(!is(typeof(cublasCtrmv))) {
        private enum enumMixinStr_cublasCtrmv = `enum cublasCtrmv = cublasCtrmv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasCtrmv); }))) {
            mixin(enumMixinStr_cublasCtrmv);
        }
    }




    static if(!is(typeof(cublasDtrmv))) {
        private enum enumMixinStr_cublasDtrmv = `enum cublasDtrmv = cublasDtrmv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDtrmv); }))) {
            mixin(enumMixinStr_cublasDtrmv);
        }
    }




    static if(!is(typeof(cublasStrmv))) {
        private enum enumMixinStr_cublasStrmv = `enum cublasStrmv = cublasStrmv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasStrmv); }))) {
            mixin(enumMixinStr_cublasStrmv);
        }
    }




    static if(!is(typeof(cublasZgbmv))) {
        private enum enumMixinStr_cublasZgbmv = `enum cublasZgbmv = cublasZgbmv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZgbmv); }))) {
            mixin(enumMixinStr_cublasZgbmv);
        }
    }




    static if(!is(typeof(cublasCgbmv))) {
        private enum enumMixinStr_cublasCgbmv = `enum cublasCgbmv = cublasCgbmv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasCgbmv); }))) {
            mixin(enumMixinStr_cublasCgbmv);
        }
    }




    static if(!is(typeof(cublasDgbmv))) {
        private enum enumMixinStr_cublasDgbmv = `enum cublasDgbmv = cublasDgbmv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDgbmv); }))) {
            mixin(enumMixinStr_cublasDgbmv);
        }
    }




    static if(!is(typeof(cublasSgbmv))) {
        private enum enumMixinStr_cublasSgbmv = `enum cublasSgbmv = cublasSgbmv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasSgbmv); }))) {
            mixin(enumMixinStr_cublasSgbmv);
        }
    }




    static if(!is(typeof(cublasZgemv))) {
        private enum enumMixinStr_cublasZgemv = `enum cublasZgemv = cublasZgemv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZgemv); }))) {
            mixin(enumMixinStr_cublasZgemv);
        }
    }




    static if(!is(typeof(cublasCgemv))) {
        private enum enumMixinStr_cublasCgemv = `enum cublasCgemv = cublasCgemv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasCgemv); }))) {
            mixin(enumMixinStr_cublasCgemv);
        }
    }




    static if(!is(typeof(cublasDgemv))) {
        private enum enumMixinStr_cublasDgemv = `enum cublasDgemv = cublasDgemv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDgemv); }))) {
            mixin(enumMixinStr_cublasDgemv);
        }
    }




    static if(!is(typeof(cublasSgemv))) {
        private enum enumMixinStr_cublasSgemv = `enum cublasSgemv = cublasSgemv_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasSgemv); }))) {
            mixin(enumMixinStr_cublasSgemv);
        }
    }




    static if(!is(typeof(cublasDrotmg))) {
        private enum enumMixinStr_cublasDrotmg = `enum cublasDrotmg = cublasDrotmg_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDrotmg); }))) {
            mixin(enumMixinStr_cublasDrotmg);
        }
    }




    static if(!is(typeof(cublasSrotmg))) {
        private enum enumMixinStr_cublasSrotmg = `enum cublasSrotmg = cublasSrotmg_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasSrotmg); }))) {
            mixin(enumMixinStr_cublasSrotmg);
        }
    }




    static if(!is(typeof(cublasDrotm))) {
        private enum enumMixinStr_cublasDrotm = `enum cublasDrotm = cublasDrotm_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDrotm); }))) {
            mixin(enumMixinStr_cublasDrotm);
        }
    }




    static if(!is(typeof(cublasSrotm))) {
        private enum enumMixinStr_cublasSrotm = `enum cublasSrotm = cublasSrotm_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasSrotm); }))) {
            mixin(enumMixinStr_cublasSrotm);
        }
    }




    static if(!is(typeof(cublasZrotg))) {
        private enum enumMixinStr_cublasZrotg = `enum cublasZrotg = cublasZrotg_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZrotg); }))) {
            mixin(enumMixinStr_cublasZrotg);
        }
    }




    static if(!is(typeof(cublasCrotg))) {
        private enum enumMixinStr_cublasCrotg = `enum cublasCrotg = cublasCrotg_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasCrotg); }))) {
            mixin(enumMixinStr_cublasCrotg);
        }
    }




    static if(!is(typeof(cublasDrotg))) {
        private enum enumMixinStr_cublasDrotg = `enum cublasDrotg = cublasDrotg_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDrotg); }))) {
            mixin(enumMixinStr_cublasDrotg);
        }
    }




    static if(!is(typeof(cublasSrotg))) {
        private enum enumMixinStr_cublasSrotg = `enum cublasSrotg = cublasSrotg_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasSrotg); }))) {
            mixin(enumMixinStr_cublasSrotg);
        }
    }




    static if(!is(typeof(cublasZdrot))) {
        private enum enumMixinStr_cublasZdrot = `enum cublasZdrot = cublasZdrot_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZdrot); }))) {
            mixin(enumMixinStr_cublasZdrot);
        }
    }




    static if(!is(typeof(cublasZrot))) {
        private enum enumMixinStr_cublasZrot = `enum cublasZrot = cublasZrot_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZrot); }))) {
            mixin(enumMixinStr_cublasZrot);
        }
    }




    static if(!is(typeof(cublasCsrot))) {
        private enum enumMixinStr_cublasCsrot = `enum cublasCsrot = cublasCsrot_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasCsrot); }))) {
            mixin(enumMixinStr_cublasCsrot);
        }
    }




    static if(!is(typeof(cublasCrot))) {
        private enum enumMixinStr_cublasCrot = `enum cublasCrot = cublasCrot_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasCrot); }))) {
            mixin(enumMixinStr_cublasCrot);
        }
    }




    static if(!is(typeof(cublasDrot))) {
        private enum enumMixinStr_cublasDrot = `enum cublasDrot = cublasDrot_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDrot); }))) {
            mixin(enumMixinStr_cublasDrot);
        }
    }




    static if(!is(typeof(cublasSrot))) {
        private enum enumMixinStr_cublasSrot = `enum cublasSrot = cublasSrot_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasSrot); }))) {
            mixin(enumMixinStr_cublasSrot);
        }
    }




    static if(!is(typeof(cublasDzasum))) {
        private enum enumMixinStr_cublasDzasum = `enum cublasDzasum = cublasDzasum_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDzasum); }))) {
            mixin(enumMixinStr_cublasDzasum);
        }
    }




    static if(!is(typeof(cublasScasum))) {
        private enum enumMixinStr_cublasScasum = `enum cublasScasum = cublasScasum_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasScasum); }))) {
            mixin(enumMixinStr_cublasScasum);
        }
    }




    static if(!is(typeof(cublasDasum))) {
        private enum enumMixinStr_cublasDasum = `enum cublasDasum = cublasDasum_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDasum); }))) {
            mixin(enumMixinStr_cublasDasum);
        }
    }




    static if(!is(typeof(cublasSasum))) {
        private enum enumMixinStr_cublasSasum = `enum cublasSasum = cublasSasum_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasSasum); }))) {
            mixin(enumMixinStr_cublasSasum);
        }
    }




    static if(!is(typeof(cublasIzamin))) {
        private enum enumMixinStr_cublasIzamin = `enum cublasIzamin = cublasIzamin_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasIzamin); }))) {
            mixin(enumMixinStr_cublasIzamin);
        }
    }




    static if(!is(typeof(cublasIcamin))) {
        private enum enumMixinStr_cublasIcamin = `enum cublasIcamin = cublasIcamin_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasIcamin); }))) {
            mixin(enumMixinStr_cublasIcamin);
        }
    }




    static if(!is(typeof(cublasIdamin))) {
        private enum enumMixinStr_cublasIdamin = `enum cublasIdamin = cublasIdamin_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasIdamin); }))) {
            mixin(enumMixinStr_cublasIdamin);
        }
    }




    static if(!is(typeof(cublasIsamin))) {
        private enum enumMixinStr_cublasIsamin = `enum cublasIsamin = cublasIsamin_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasIsamin); }))) {
            mixin(enumMixinStr_cublasIsamin);
        }
    }




    static if(!is(typeof(cublasIzamax))) {
        private enum enumMixinStr_cublasIzamax = `enum cublasIzamax = cublasIzamax_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasIzamax); }))) {
            mixin(enumMixinStr_cublasIzamax);
        }
    }




    static if(!is(typeof(cublasIcamax))) {
        private enum enumMixinStr_cublasIcamax = `enum cublasIcamax = cublasIcamax_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasIcamax); }))) {
            mixin(enumMixinStr_cublasIcamax);
        }
    }




    static if(!is(typeof(cublasIdamax))) {
        private enum enumMixinStr_cublasIdamax = `enum cublasIdamax = cublasIdamax_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasIdamax); }))) {
            mixin(enumMixinStr_cublasIdamax);
        }
    }




    static if(!is(typeof(cublasIsamax))) {
        private enum enumMixinStr_cublasIsamax = `enum cublasIsamax = cublasIsamax_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasIsamax); }))) {
            mixin(enumMixinStr_cublasIsamax);
        }
    }




    static if(!is(typeof(cublasZswap))) {
        private enum enumMixinStr_cublasZswap = `enum cublasZswap = cublasZswap_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZswap); }))) {
            mixin(enumMixinStr_cublasZswap);
        }
    }




    static if(!is(typeof(cublasCswap))) {
        private enum enumMixinStr_cublasCswap = `enum cublasCswap = cublasCswap_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasCswap); }))) {
            mixin(enumMixinStr_cublasCswap);
        }
    }




    static if(!is(typeof(cublasDswap))) {
        private enum enumMixinStr_cublasDswap = `enum cublasDswap = cublasDswap_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDswap); }))) {
            mixin(enumMixinStr_cublasDswap);
        }
    }




    static if(!is(typeof(cublasSswap))) {
        private enum enumMixinStr_cublasSswap = `enum cublasSswap = cublasSswap_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasSswap); }))) {
            mixin(enumMixinStr_cublasSswap);
        }
    }




    static if(!is(typeof(cublasZcopy))) {
        private enum enumMixinStr_cublasZcopy = `enum cublasZcopy = cublasZcopy_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZcopy); }))) {
            mixin(enumMixinStr_cublasZcopy);
        }
    }




    static if(!is(typeof(cublasCcopy))) {
        private enum enumMixinStr_cublasCcopy = `enum cublasCcopy = cublasCcopy_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasCcopy); }))) {
            mixin(enumMixinStr_cublasCcopy);
        }
    }




    static if(!is(typeof(cublasDcopy))) {
        private enum enumMixinStr_cublasDcopy = `enum cublasDcopy = cublasDcopy_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDcopy); }))) {
            mixin(enumMixinStr_cublasDcopy);
        }
    }




    static if(!is(typeof(cublasScopy))) {
        private enum enumMixinStr_cublasScopy = `enum cublasScopy = cublasScopy_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasScopy); }))) {
            mixin(enumMixinStr_cublasScopy);
        }
    }




    static if(!is(typeof(cublasZaxpy))) {
        private enum enumMixinStr_cublasZaxpy = `enum cublasZaxpy = cublasZaxpy_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZaxpy); }))) {
            mixin(enumMixinStr_cublasZaxpy);
        }
    }




    static if(!is(typeof(cublasCaxpy))) {
        private enum enumMixinStr_cublasCaxpy = `enum cublasCaxpy = cublasCaxpy_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasCaxpy); }))) {
            mixin(enumMixinStr_cublasCaxpy);
        }
    }




    static if(!is(typeof(cublasDaxpy))) {
        private enum enumMixinStr_cublasDaxpy = `enum cublasDaxpy = cublasDaxpy_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDaxpy); }))) {
            mixin(enumMixinStr_cublasDaxpy);
        }
    }




    static if(!is(typeof(cublasSaxpy))) {
        private enum enumMixinStr_cublasSaxpy = `enum cublasSaxpy = cublasSaxpy_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasSaxpy); }))) {
            mixin(enumMixinStr_cublasSaxpy);
        }
    }




    static if(!is(typeof(cublasZdscal))) {
        private enum enumMixinStr_cublasZdscal = `enum cublasZdscal = cublasZdscal_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZdscal); }))) {
            mixin(enumMixinStr_cublasZdscal);
        }
    }




    static if(!is(typeof(cublasZscal))) {
        private enum enumMixinStr_cublasZscal = `enum cublasZscal = cublasZscal_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZscal); }))) {
            mixin(enumMixinStr_cublasZscal);
        }
    }




    static if(!is(typeof(cublasCsscal))) {
        private enum enumMixinStr_cublasCsscal = `enum cublasCsscal = cublasCsscal_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasCsscal); }))) {
            mixin(enumMixinStr_cublasCsscal);
        }
    }




    static if(!is(typeof(cublasCscal))) {
        private enum enumMixinStr_cublasCscal = `enum cublasCscal = cublasCscal_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasCscal); }))) {
            mixin(enumMixinStr_cublasCscal);
        }
    }




    static if(!is(typeof(cublasDscal))) {
        private enum enumMixinStr_cublasDscal = `enum cublasDscal = cublasDscal_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDscal); }))) {
            mixin(enumMixinStr_cublasDscal);
        }
    }




    static if(!is(typeof(cublasSscal))) {
        private enum enumMixinStr_cublasSscal = `enum cublasSscal = cublasSscal_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasSscal); }))) {
            mixin(enumMixinStr_cublasSscal);
        }
    }




    static if(!is(typeof(cublasZdotc))) {
        private enum enumMixinStr_cublasZdotc = `enum cublasZdotc = cublasZdotc_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZdotc); }))) {
            mixin(enumMixinStr_cublasZdotc);
        }
    }




    static if(!is(typeof(cublasZdotu))) {
        private enum enumMixinStr_cublasZdotu = `enum cublasZdotu = cublasZdotu_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasZdotu); }))) {
            mixin(enumMixinStr_cublasZdotu);
        }
    }




    static if(!is(typeof(cublasCdotc))) {
        private enum enumMixinStr_cublasCdotc = `enum cublasCdotc = cublasCdotc_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasCdotc); }))) {
            mixin(enumMixinStr_cublasCdotc);
        }
    }




    static if(!is(typeof(cublasCdotu))) {
        private enum enumMixinStr_cublasCdotu = `enum cublasCdotu = cublasCdotu_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasCdotu); }))) {
            mixin(enumMixinStr_cublasCdotu);
        }
    }




    static if(!is(typeof(cublasDdot))) {
        private enum enumMixinStr_cublasDdot = `enum cublasDdot = cublasDdot_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDdot); }))) {
            mixin(enumMixinStr_cublasDdot);
        }
    }




    static if(!is(typeof(cublasSdot))) {
        private enum enumMixinStr_cublasSdot = `enum cublasSdot = cublasSdot_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasSdot); }))) {
            mixin(enumMixinStr_cublasSdot);
        }
    }




    static if(!is(typeof(cublasDznrm2))) {
        private enum enumMixinStr_cublasDznrm2 = `enum cublasDznrm2 = cublasDznrm2_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDznrm2); }))) {
            mixin(enumMixinStr_cublasDznrm2);
        }
    }




    static if(!is(typeof(cublasScnrm2))) {
        private enum enumMixinStr_cublasScnrm2 = `enum cublasScnrm2 = cublasScnrm2_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasScnrm2); }))) {
            mixin(enumMixinStr_cublasScnrm2);
        }
    }




    static if(!is(typeof(cublasDnrm2))) {
        private enum enumMixinStr_cublasDnrm2 = `enum cublasDnrm2 = cublasDnrm2_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDnrm2); }))) {
            mixin(enumMixinStr_cublasDnrm2);
        }
    }




    static if(!is(typeof(cublasSnrm2))) {
        private enum enumMixinStr_cublasSnrm2 = `enum cublasSnrm2 = cublasSnrm2_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasSnrm2); }))) {
            mixin(enumMixinStr_cublasSnrm2);
        }
    }




    static if(!is(typeof(cublasSetPointerMode))) {
        private enum enumMixinStr_cublasSetPointerMode = `enum cublasSetPointerMode = cublasSetPointerMode_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasSetPointerMode); }))) {
            mixin(enumMixinStr_cublasSetPointerMode);
        }
    }




    static if(!is(typeof(cublasGetPointerMode))) {
        private enum enumMixinStr_cublasGetPointerMode = `enum cublasGetPointerMode = cublasGetPointerMode_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasGetPointerMode); }))) {
            mixin(enumMixinStr_cublasGetPointerMode);
        }
    }




    static if(!is(typeof(cublasGetStream))) {
        private enum enumMixinStr_cublasGetStream = `enum cublasGetStream = cublasGetStream_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasGetStream); }))) {
            mixin(enumMixinStr_cublasGetStream);
        }
    }




    static if(!is(typeof(cublasSetStream))) {
        private enum enumMixinStr_cublasSetStream = `enum cublasSetStream = cublasSetStream_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasSetStream); }))) {
            mixin(enumMixinStr_cublasSetStream);
        }
    }




    static if(!is(typeof(cublasGetVersion))) {
        private enum enumMixinStr_cublasGetVersion = `enum cublasGetVersion = cublasGetVersion_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasGetVersion); }))) {
            mixin(enumMixinStr_cublasGetVersion);
        }
    }




    static if(!is(typeof(cublasDestroy))) {
        private enum enumMixinStr_cublasDestroy = `enum cublasDestroy = cublasDestroy_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasDestroy); }))) {
            mixin(enumMixinStr_cublasDestroy);
        }
    }




    static if(!is(typeof(cublasCreate))) {
        private enum enumMixinStr_cublasCreate = `enum cublasCreate = cublasCreate_v2;`;
        static if(is(typeof({ mixin(enumMixinStr_cublasCreate); }))) {
            mixin(enumMixinStr_cublasCreate);
        }
    }
    static if(!is(typeof(CUBLAS_VERSION))) {
        private enum enumMixinStr_CUBLAS_VERSION = `enum CUBLAS_VERSION = ( CUBLAS_VER_MAJOR * 1000 + CUBLAS_VER_MINOR * 100 + CUBLAS_VER_PATCH );`;
        static if(is(typeof({ mixin(enumMixinStr_CUBLAS_VERSION); }))) {
            mixin(enumMixinStr_CUBLAS_VERSION);
        }
    }






    static if(!is(typeof(CUBLAS_VER_BUILD))) {
        private enum enumMixinStr_CUBLAS_VER_BUILD = `enum CUBLAS_VER_BUILD = 89;`;
        static if(is(typeof({ mixin(enumMixinStr_CUBLAS_VER_BUILD); }))) {
            mixin(enumMixinStr_CUBLAS_VER_BUILD);
        }
    }




    static if(!is(typeof(__CUDA_HOSTDEVICE_FP16_DECL__))) {
        private enum enumMixinStr___CUDA_HOSTDEVICE_FP16_DECL__ = `enum __CUDA_HOSTDEVICE_FP16_DECL__ = static __attribute__ ( ( unused ) );`;
        static if(is(typeof({ mixin(enumMixinStr___CUDA_HOSTDEVICE_FP16_DECL__); }))) {
            mixin(enumMixinStr___CUDA_HOSTDEVICE_FP16_DECL__);
        }
    }






    static if(!is(typeof(CUBLAS_VER_PATCH))) {
        private enum enumMixinStr_CUBLAS_VER_PATCH = `enum CUBLAS_VER_PATCH = 2;`;
        static if(is(typeof({ mixin(enumMixinStr_CUBLAS_VER_PATCH); }))) {
            mixin(enumMixinStr_CUBLAS_VER_PATCH);
        }
    }
    static if(!is(typeof(CUBLAS_VER_MINOR))) {
        private enum enumMixinStr_CUBLAS_VER_MINOR = `enum CUBLAS_VER_MINOR = 2;`;
        static if(is(typeof({ mixin(enumMixinStr_CUBLAS_VER_MINOR); }))) {
            mixin(enumMixinStr_CUBLAS_VER_MINOR);
        }
    }




    static if(!is(typeof(CUBLAS_VER_MAJOR))) {
        private enum enumMixinStr_CUBLAS_VER_MAJOR = `enum CUBLAS_VER_MAJOR = 10;`;
        static if(is(typeof({ mixin(enumMixinStr_CUBLAS_VER_MAJOR); }))) {
            mixin(enumMixinStr_CUBLAS_VER_MAJOR);
        }
    }
    static if(!is(typeof(NULL))) {
        private enum enumMixinStr_NULL = `enum NULL = ( cast( void * ) 0 );`;
        static if(is(typeof({ mixin(enumMixinStr_NULL); }))) {
            mixin(enumMixinStr_NULL);
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






    static if(!is(typeof(__CUDA_DEPRECATED))) {
        private enum enumMixinStr___CUDA_DEPRECATED = `enum __CUDA_DEPRECATED = __attribute__ ( ( deprecated ) );`;
        static if(is(typeof({ mixin(enumMixinStr___CUDA_DEPRECATED); }))) {
            mixin(enumMixinStr___CUDA_DEPRECATED);
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
    static if(!is(typeof(__cuda_builtin_vector_align8))) {
        private enum enumMixinStr___cuda_builtin_vector_align8 = `enum __cuda_builtin_vector_align8 = ( tag , members ) __attribute__ ( ( aligned ( 8 ) ) ) tag
 { members
 };`;
        static if(is(typeof({ mixin(enumMixinStr___cuda_builtin_vector_align8); }))) {
            mixin(enumMixinStr___cuda_builtin_vector_align8);
        }
    }



}
