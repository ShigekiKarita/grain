module grain.dpp.cudnn;


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
    static double4 make_double4(double, double, double, double) @nogc nothrow;
    static double3 make_double3(double, double, double) @nogc nothrow;
    static double2 make_double2(double, double) @nogc nothrow;
    static double1 make_double1(double) @nogc nothrow;
    static ulonglong4 make_ulonglong4(ulong, ulong, ulong, ulong) @nogc nothrow;
    static longlong4 make_longlong4(long, long, long, long) @nogc nothrow;
    static ulonglong3 make_ulonglong3(ulong, ulong, ulong) @nogc nothrow;
    static longlong3 make_longlong3(long, long, long) @nogc nothrow;
    static ulonglong2 make_ulonglong2(ulong, ulong) @nogc nothrow;
    static longlong2 make_longlong2(long, long) @nogc nothrow;
    static ulonglong1 make_ulonglong1(ulong) @nogc nothrow;
    static longlong1 make_longlong1(long) @nogc nothrow;
    static float4 make_float4(float, float, float, float) @nogc nothrow;
    static float3 make_float3(float, float, float) @nogc nothrow;
    static float2 make_float2(float, float) @nogc nothrow;
    static float1 make_float1(float) @nogc nothrow;
    static ulong4 make_ulong4(c_ulong, c_ulong, c_ulong, c_ulong) @nogc nothrow;
    static long4 make_long4(c_long, c_long, c_long, c_long) @nogc nothrow;
    static ulong3 make_ulong3(c_ulong, c_ulong, c_ulong) @nogc nothrow;
    static long3 make_long3(c_long, c_long, c_long) @nogc nothrow;
    static ulong2 make_ulong2(c_ulong, c_ulong) @nogc nothrow;
    static long2 make_long2(c_long, c_long) @nogc nothrow;
    static ulong1 make_ulong1(c_ulong) @nogc nothrow;
    static long1 make_long1(c_long) @nogc nothrow;
    static uint4 make_uint4(uint, uint, uint, uint) @nogc nothrow;
    static int4 make_int4(int, int, int, int) @nogc nothrow;
    static uint3 make_uint3(uint, uint, uint) @nogc nothrow;
    static int3 make_int3(int, int, int) @nogc nothrow;
    static uint2 make_uint2(uint, uint) @nogc nothrow;
    static int2 make_int2(int, int) @nogc nothrow;
    static uint1 make_uint1(uint) @nogc nothrow;
    static int1 make_int1(int) @nogc nothrow;
    static ushort4 make_ushort4(ushort, ushort, ushort, ushort) @nogc nothrow;
    struct max_align_t
    {
        long __clang_max_align_nonce1;
        real __clang_max_align_nonce2;
    }
    static short4 make_short4(short, short, short, short) @nogc nothrow;
    static ushort3 make_ushort3(ushort, ushort, ushort) @nogc nothrow;
    static short3 make_short3(short, short, short) @nogc nothrow;
    static ushort2 make_ushort2(ushort, ushort) @nogc nothrow;
    static short2 make_short2(short, short) @nogc nothrow;
    static ushort1 make_ushort1(ushort) @nogc nothrow;
    static short1 make_short1(short) @nogc nothrow;
    static uchar4 make_uchar4(ubyte, ubyte, ubyte, ubyte) @nogc nothrow;
    static char4 make_char4(byte, byte, byte, byte) @nogc nothrow;
    static uchar3 make_uchar3(ubyte, ubyte, ubyte) @nogc nothrow;
    static char3 make_char3(byte, byte, byte) @nogc nothrow;
    static uchar2 make_uchar2(ubyte, ubyte) @nogc nothrow;
    alias ptrdiff_t = c_long;
    static char2 make_char2(byte, byte) @nogc nothrow;
    alias size_t = c_ulong;
    alias wchar_t = int;
    static uchar1 make_uchar1(ubyte) @nogc nothrow;
    static char1 make_char1(byte) @nogc nothrow;
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
    alias cudaOutputMode_t = cudaOutputMode;
    alias uintptr_t = c_ulong;
    alias intmax_t = c_long;
    alias uintmax_t = c_ulong;
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
    static cudaExtent make_cudaExtent(c_ulong, c_ulong, c_ulong) @nogc nothrow;
    static cudaPos make_cudaPos(c_ulong, c_ulong, c_ulong) @nogc nothrow;
    static cudaPitchedPtr make_cudaPitchedPtr(void*, c_ulong, c_ulong, c_ulong) @nogc nothrow;
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
    alias int8_t = byte;
    alias int16_t = short;
    alias int32_t = int;
    alias int64_t = c_long;
    alias uint8_t = ubyte;
    alias uint16_t = ushort;
    alias uint32_t = uint;
    alias uint64_t = ulong;
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
    cudnnStatus_t cudnnSetRNNDescriptor_v5(cudnnRNNStruct*, int, int, cudnnDropoutStruct*, cudnnRNNInputMode_t, cudnnDirectionMode_t, cudnnRNNMode_t, cudnnDataType_t) @nogc nothrow;
    cudnnStatus_t cudnnSetRNNDescriptor_v6(cudnnContext*, cudnnRNNStruct*, const(int), const(int), cudnnDropoutStruct*, cudnnRNNInputMode_t, cudnnDirectionMode_t, cudnnRNNMode_t, cudnnRNNAlgo_t, cudnnDataType_t) @nogc nothrow;
    cudnnStatus_t cudnnFusedOpsExecute(cudnnContext*, const(cudnnFusedOpsPlanStruct*), cudnnFusedOpsVariantParamStruct*) @nogc nothrow;
    cudnnStatus_t cudnnMakeFusedOpsPlan(cudnnContext*, cudnnFusedOpsPlanStruct*, const(cudnnFusedOpsConstParamStruct*), c_ulong*) @nogc nothrow;
    cudnnStatus_t cudnnDestroyFusedOpsPlan(cudnnFusedOpsPlanStruct*) @nogc nothrow;
    cudnnStatus_t cudnnCreateFusedOpsPlan(cudnnFusedOpsPlanStruct**, cudnnFusedOps_t) @nogc nothrow;
    cudnnStatus_t cudnnGetFusedOpsVariantParamPackAttribute(const(cudnnFusedOpsVariantParamStruct*), cudnnFusedOpsVariantParamLabel_t, void*) @nogc nothrow;
    cudnnStatus_t cudnnSetFusedOpsVariantParamPackAttribute(cudnnFusedOpsVariantParamStruct*, cudnnFusedOpsVariantParamLabel_t, void*) @nogc nothrow;
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
    cudnnStatus_t cudnnDestroyFusedOpsVariantParamPack(cudnnFusedOpsVariantParamStruct*) @nogc nothrow;
    cudnnStatus_t cudnnCreateFusedOpsVariantParamPack(cudnnFusedOpsVariantParamStruct**, cudnnFusedOps_t) @nogc nothrow;
    cudnnStatus_t cudnnGetFusedOpsConstParamPackAttribute(const(cudnnFusedOpsConstParamStruct*), cudnnFusedOpsConstParamLabel_t, void*, int*) @nogc nothrow;
    cudnnStatus_t cudnnSetFusedOpsConstParamPackAttribute(cudnnFusedOpsConstParamStruct*, cudnnFusedOpsConstParamLabel_t, const(void)*) @nogc nothrow;
    cudnnStatus_t cudnnDestroyFusedOpsConstParamPack(cudnnFusedOpsConstParamStruct*) @nogc nothrow;
    cudnnStatus_t cudnnCreateFusedOpsConstParamPack(cudnnFusedOpsConstParamStruct**, cudnnFusedOps_t) @nogc nothrow;
    enum _Anonymous_17
    {
        CUDNN_PTR_XDATA = 0,
        CUDNN_PTR_BN_EQSCALE = 1,
        CUDNN_PTR_BN_EQBIAS = 2,
        CUDNN_PTR_WDATA = 3,
        CUDNN_PTR_DWDATA = 4,
        CUDNN_PTR_YDATA = 5,
        CUDNN_PTR_DYDATA = 6,
        CUDNN_PTR_YSUM = 7,
        CUDNN_PTR_YSQSUM = 8,
        CUDNN_PTR_WORKSPACE = 9,
        CUDNN_PTR_BN_SCALE = 10,
        CUDNN_PTR_BN_BIAS = 11,
        CUDNN_PTR_BN_SAVED_MEAN = 12,
        CUDNN_PTR_BN_SAVED_INVSTD = 13,
        CUDNN_PTR_BN_RUNNING_MEAN = 14,
        CUDNN_PTR_BN_RUNNING_VAR = 15,
        CUDNN_PTR_ZDATA = 16,
        CUDNN_PTR_BN_Z_EQSCALE = 17,
        CUDNN_PTR_BN_Z_EQBIAS = 18,
        CUDNN_PTR_ACTIVATION_BITMASK = 19,
        CUDNN_PTR_DXDATA = 20,
        CUDNN_PTR_DZDATA = 21,
        CUDNN_PTR_BN_DSCALE = 22,
        CUDNN_PTR_BN_DBIAS = 23,
        CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES = 100,
        CUDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT = 101,
        CUDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR = 102,
        CUDNN_SCALAR_DOUBLE_BN_EPSILON = 103,
    }
    enum CUDNN_PTR_XDATA = _Anonymous_17.CUDNN_PTR_XDATA;
    enum CUDNN_PTR_BN_EQSCALE = _Anonymous_17.CUDNN_PTR_BN_EQSCALE;
    enum CUDNN_PTR_BN_EQBIAS = _Anonymous_17.CUDNN_PTR_BN_EQBIAS;
    enum CUDNN_PTR_WDATA = _Anonymous_17.CUDNN_PTR_WDATA;
    enum CUDNN_PTR_DWDATA = _Anonymous_17.CUDNN_PTR_DWDATA;
    enum CUDNN_PTR_YDATA = _Anonymous_17.CUDNN_PTR_YDATA;
    enum CUDNN_PTR_DYDATA = _Anonymous_17.CUDNN_PTR_DYDATA;
    enum CUDNN_PTR_YSUM = _Anonymous_17.CUDNN_PTR_YSUM;
    enum CUDNN_PTR_YSQSUM = _Anonymous_17.CUDNN_PTR_YSQSUM;
    enum CUDNN_PTR_WORKSPACE = _Anonymous_17.CUDNN_PTR_WORKSPACE;
    enum CUDNN_PTR_BN_SCALE = _Anonymous_17.CUDNN_PTR_BN_SCALE;
    enum CUDNN_PTR_BN_BIAS = _Anonymous_17.CUDNN_PTR_BN_BIAS;
    enum CUDNN_PTR_BN_SAVED_MEAN = _Anonymous_17.CUDNN_PTR_BN_SAVED_MEAN;
    enum CUDNN_PTR_BN_SAVED_INVSTD = _Anonymous_17.CUDNN_PTR_BN_SAVED_INVSTD;
    enum CUDNN_PTR_BN_RUNNING_MEAN = _Anonymous_17.CUDNN_PTR_BN_RUNNING_MEAN;
    enum CUDNN_PTR_BN_RUNNING_VAR = _Anonymous_17.CUDNN_PTR_BN_RUNNING_VAR;
    enum CUDNN_PTR_ZDATA = _Anonymous_17.CUDNN_PTR_ZDATA;
    enum CUDNN_PTR_BN_Z_EQSCALE = _Anonymous_17.CUDNN_PTR_BN_Z_EQSCALE;
    enum CUDNN_PTR_BN_Z_EQBIAS = _Anonymous_17.CUDNN_PTR_BN_Z_EQBIAS;
    enum CUDNN_PTR_ACTIVATION_BITMASK = _Anonymous_17.CUDNN_PTR_ACTIVATION_BITMASK;
    enum CUDNN_PTR_DXDATA = _Anonymous_17.CUDNN_PTR_DXDATA;
    enum CUDNN_PTR_DZDATA = _Anonymous_17.CUDNN_PTR_DZDATA;
    enum CUDNN_PTR_BN_DSCALE = _Anonymous_17.CUDNN_PTR_BN_DSCALE;
    enum CUDNN_PTR_BN_DBIAS = _Anonymous_17.CUDNN_PTR_BN_DBIAS;
    enum CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES = _Anonymous_17.CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES;
    enum CUDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT = _Anonymous_17.CUDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT;
    enum CUDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR = _Anonymous_17.CUDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR;
    enum CUDNN_SCALAR_DOUBLE_BN_EPSILON = _Anonymous_17.CUDNN_SCALAR_DOUBLE_BN_EPSILON;
    alias cudnnFusedOpsVariantParamLabel_t = _Anonymous_17;
    enum _Anonymous_18
    {
        CUDNN_PTR_NULL = 0,
        CUDNN_PTR_ELEM_ALIGNED = 1,
        CUDNN_PTR_16B_ALIGNED = 2,
    }
    enum CUDNN_PTR_NULL = _Anonymous_18.CUDNN_PTR_NULL;
    enum CUDNN_PTR_ELEM_ALIGNED = _Anonymous_18.CUDNN_PTR_ELEM_ALIGNED;
    enum CUDNN_PTR_16B_ALIGNED = _Anonymous_18.CUDNN_PTR_16B_ALIGNED;
    alias cudnnFusedOpsPointerPlaceHolder_t = _Anonymous_18;
    enum _Anonymous_19
    {
        CUDNN_PARAM_XDESC = 0,
        CUDNN_PARAM_XDATA_PLACEHOLDER = 1,
        CUDNN_PARAM_BN_MODE = 2,
        CUDNN_PARAM_BN_EQSCALEBIAS_DESC = 3,
        CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER = 4,
        CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER = 5,
        CUDNN_PARAM_ACTIVATION_DESC = 6,
        CUDNN_PARAM_CONV_DESC = 7,
        CUDNN_PARAM_WDESC = 8,
        CUDNN_PARAM_WDATA_PLACEHOLDER = 9,
        CUDNN_PARAM_DWDESC = 10,
        CUDNN_PARAM_DWDATA_PLACEHOLDER = 11,
        CUDNN_PARAM_YDESC = 12,
        CUDNN_PARAM_YDATA_PLACEHOLDER = 13,
        CUDNN_PARAM_DYDESC = 14,
        CUDNN_PARAM_DYDATA_PLACEHOLDER = 15,
        CUDNN_PARAM_YSTATS_DESC = 16,
        CUDNN_PARAM_YSUM_PLACEHOLDER = 17,
        CUDNN_PARAM_YSQSUM_PLACEHOLDER = 18,
        CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC = 19,
        CUDNN_PARAM_BN_SCALE_PLACEHOLDER = 20,
        CUDNN_PARAM_BN_BIAS_PLACEHOLDER = 21,
        CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER = 22,
        CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER = 23,
        CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER = 24,
        CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER = 25,
        CUDNN_PARAM_ZDESC = 26,
        CUDNN_PARAM_ZDATA_PLACEHOLDER = 27,
        CUDNN_PARAM_BN_Z_EQSCALEBIAS_DESC = 28,
        CUDNN_PARAM_BN_Z_EQSCALE_PLACEHOLDER = 29,
        CUDNN_PARAM_BN_Z_EQBIAS_PLACEHOLDER = 30,
        CUDNN_PARAM_ACTIVATION_BITMASK_DESC = 31,
        CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER = 32,
        CUDNN_PARAM_DXDESC = 33,
        CUDNN_PARAM_DXDATA_PLACEHOLDER = 34,
        CUDNN_PARAM_DZDESC = 35,
        CUDNN_PARAM_DZDATA_PLACEHOLDER = 36,
        CUDNN_PARAM_BN_DSCALE_PLACEHOLDER = 37,
        CUDNN_PARAM_BN_DBIAS_PLACEHOLDER = 38,
    }
    enum CUDNN_PARAM_XDESC = _Anonymous_19.CUDNN_PARAM_XDESC;
    enum CUDNN_PARAM_XDATA_PLACEHOLDER = _Anonymous_19.CUDNN_PARAM_XDATA_PLACEHOLDER;
    enum CUDNN_PARAM_BN_MODE = _Anonymous_19.CUDNN_PARAM_BN_MODE;
    enum CUDNN_PARAM_BN_EQSCALEBIAS_DESC = _Anonymous_19.CUDNN_PARAM_BN_EQSCALEBIAS_DESC;
    enum CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER = _Anonymous_19.CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER;
    enum CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER = _Anonymous_19.CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER;
    enum CUDNN_PARAM_ACTIVATION_DESC = _Anonymous_19.CUDNN_PARAM_ACTIVATION_DESC;
    enum CUDNN_PARAM_CONV_DESC = _Anonymous_19.CUDNN_PARAM_CONV_DESC;
    enum CUDNN_PARAM_WDESC = _Anonymous_19.CUDNN_PARAM_WDESC;
    enum CUDNN_PARAM_WDATA_PLACEHOLDER = _Anonymous_19.CUDNN_PARAM_WDATA_PLACEHOLDER;
    enum CUDNN_PARAM_DWDESC = _Anonymous_19.CUDNN_PARAM_DWDESC;
    enum CUDNN_PARAM_DWDATA_PLACEHOLDER = _Anonymous_19.CUDNN_PARAM_DWDATA_PLACEHOLDER;
    enum CUDNN_PARAM_YDESC = _Anonymous_19.CUDNN_PARAM_YDESC;
    enum CUDNN_PARAM_YDATA_PLACEHOLDER = _Anonymous_19.CUDNN_PARAM_YDATA_PLACEHOLDER;
    enum CUDNN_PARAM_DYDESC = _Anonymous_19.CUDNN_PARAM_DYDESC;
    enum CUDNN_PARAM_DYDATA_PLACEHOLDER = _Anonymous_19.CUDNN_PARAM_DYDATA_PLACEHOLDER;
    enum CUDNN_PARAM_YSTATS_DESC = _Anonymous_19.CUDNN_PARAM_YSTATS_DESC;
    enum CUDNN_PARAM_YSUM_PLACEHOLDER = _Anonymous_19.CUDNN_PARAM_YSUM_PLACEHOLDER;
    enum CUDNN_PARAM_YSQSUM_PLACEHOLDER = _Anonymous_19.CUDNN_PARAM_YSQSUM_PLACEHOLDER;
    enum CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC = _Anonymous_19.CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC;
    enum CUDNN_PARAM_BN_SCALE_PLACEHOLDER = _Anonymous_19.CUDNN_PARAM_BN_SCALE_PLACEHOLDER;
    enum CUDNN_PARAM_BN_BIAS_PLACEHOLDER = _Anonymous_19.CUDNN_PARAM_BN_BIAS_PLACEHOLDER;
    enum CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER = _Anonymous_19.CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER;
    enum CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER = _Anonymous_19.CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER;
    enum CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER = _Anonymous_19.CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER;
    enum CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER = _Anonymous_19.CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER;
    enum CUDNN_PARAM_ZDESC = _Anonymous_19.CUDNN_PARAM_ZDESC;
    enum CUDNN_PARAM_ZDATA_PLACEHOLDER = _Anonymous_19.CUDNN_PARAM_ZDATA_PLACEHOLDER;
    enum CUDNN_PARAM_BN_Z_EQSCALEBIAS_DESC = _Anonymous_19.CUDNN_PARAM_BN_Z_EQSCALEBIAS_DESC;
    enum CUDNN_PARAM_BN_Z_EQSCALE_PLACEHOLDER = _Anonymous_19.CUDNN_PARAM_BN_Z_EQSCALE_PLACEHOLDER;
    enum CUDNN_PARAM_BN_Z_EQBIAS_PLACEHOLDER = _Anonymous_19.CUDNN_PARAM_BN_Z_EQBIAS_PLACEHOLDER;
    enum CUDNN_PARAM_ACTIVATION_BITMASK_DESC = _Anonymous_19.CUDNN_PARAM_ACTIVATION_BITMASK_DESC;
    enum CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER = _Anonymous_19.CUDNN_PARAM_ACTIVATION_BITMASK_PLACEHOLDER;
    enum CUDNN_PARAM_DXDESC = _Anonymous_19.CUDNN_PARAM_DXDESC;
    enum CUDNN_PARAM_DXDATA_PLACEHOLDER = _Anonymous_19.CUDNN_PARAM_DXDATA_PLACEHOLDER;
    enum CUDNN_PARAM_DZDESC = _Anonymous_19.CUDNN_PARAM_DZDESC;
    enum CUDNN_PARAM_DZDATA_PLACEHOLDER = _Anonymous_19.CUDNN_PARAM_DZDATA_PLACEHOLDER;
    enum CUDNN_PARAM_BN_DSCALE_PLACEHOLDER = _Anonymous_19.CUDNN_PARAM_BN_DSCALE_PLACEHOLDER;
    enum CUDNN_PARAM_BN_DBIAS_PLACEHOLDER = _Anonymous_19.CUDNN_PARAM_BN_DBIAS_PLACEHOLDER;
    alias cudnnFusedOpsConstParamLabel_t = _Anonymous_19;
    enum _Anonymous_20
    {
        CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS = 0,
        CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD = 1,
        CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING = 2,
        CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE = 3,
        CUDNN_FUSED_CONV_SCALE_BIAS_ADD_ACTIVATION = 4,
        CUDNN_FUSED_SCALE_BIAS_ADD_ACTIVATION_GEN_BITMASK = 5,
        CUDNN_FUSED_DACTIVATION_FORK_DBATCHNORM = 6,
    }
    enum CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS = _Anonymous_20.CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS;
    enum CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD = _Anonymous_20.CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD;
    enum CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING = _Anonymous_20.CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING;
    enum CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE = _Anonymous_20.CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE;
    enum CUDNN_FUSED_CONV_SCALE_BIAS_ADD_ACTIVATION = _Anonymous_20.CUDNN_FUSED_CONV_SCALE_BIAS_ADD_ACTIVATION;
    enum CUDNN_FUSED_SCALE_BIAS_ADD_ACTIVATION_GEN_BITMASK = _Anonymous_20.CUDNN_FUSED_SCALE_BIAS_ADD_ACTIVATION_GEN_BITMASK;
    enum CUDNN_FUSED_DACTIVATION_FORK_DBATCHNORM = _Anonymous_20.CUDNN_FUSED_DACTIVATION_FORK_DBATCHNORM;
    alias cudnnFusedOps_t = _Anonymous_20;
    alias cudnnFusedOpsPlan_t = cudnnFusedOpsPlanStruct*;
    struct cudnnFusedOpsPlanStruct;
    alias cudnnFusedOpsVariantParamPack_t = cudnnFusedOpsVariantParamStruct*;
    struct cudnnFusedOpsVariantParamStruct;
    alias cudnnFusedOpsConstParamPack_t = cudnnFusedOpsConstParamStruct*;
    struct cudnnFusedOpsConstParamStruct;
    cudnnStatus_t cudnnGetCallback(uint*, void**, void function(cudnnSeverity_t, void*, const(cudnnDebug_t)*, const(char)*)*) @nogc nothrow;
    cudnnStatus_t cudnnSetCallback(uint, void*, void function(cudnnSeverity_t, void*, const(cudnnDebug_t)*, const(char)*)) @nogc nothrow;
    alias cudnnCallback_t = void function(cudnnSeverity_t, void*, const(cudnnDebug_t)*, const(char)*);
    struct cudnnDebug_t
    {
        uint cudnn_version;
        cudnnStatus_t cudnnStatus;
        uint time_sec;
        uint time_usec;
        uint time_delta;
        cudnnContext* handle;
        CUstream_st* stream;
        ulong pid;
        ulong tid;
        int cudaDeviceId;
        int[15] reserved;
    }
    enum _Anonymous_21
    {
        CUDNN_SEV_FATAL = 0,
        CUDNN_SEV_ERROR = 1,
        CUDNN_SEV_WARNING = 2,
        CUDNN_SEV_INFO = 3,
    }
    enum CUDNN_SEV_FATAL = _Anonymous_21.CUDNN_SEV_FATAL;
    enum CUDNN_SEV_ERROR = _Anonymous_21.CUDNN_SEV_ERROR;
    enum CUDNN_SEV_WARNING = _Anonymous_21.CUDNN_SEV_WARNING;
    enum CUDNN_SEV_INFO = _Anonymous_21.CUDNN_SEV_INFO;
    alias cudnnSeverity_t = _Anonymous_21;
    cudnnStatus_t cudnnRestoreAlgorithm(cudnnContext*, void*, c_ulong, cudnnAlgorithmStruct*) @nogc nothrow;
    cudnnStatus_t cudnnSaveAlgorithm(cudnnContext*, cudnnAlgorithmStruct*, void*, c_ulong) @nogc nothrow;
    cudnnStatus_t cudnnGetAlgorithmSpaceSize(cudnnContext*, cudnnAlgorithmStruct*, c_ulong*) @nogc nothrow;
    cudnnStatus_t cudnnDestroyAlgorithmPerformance(cudnnAlgorithmPerformanceStruct**, int) @nogc nothrow;
    cudnnStatus_t cudnnGetAlgorithmPerformance(const(cudnnAlgorithmPerformanceStruct*), cudnnAlgorithmStruct**, cudnnStatus_t*, float*, c_ulong*) @nogc nothrow;
    cudnnStatus_t cudnnSetAlgorithmPerformance(cudnnAlgorithmPerformanceStruct*, cudnnAlgorithmStruct*, cudnnStatus_t, float, c_ulong) @nogc nothrow;
    cudnnStatus_t cudnnCreateAlgorithmPerformance(cudnnAlgorithmPerformanceStruct**, int) @nogc nothrow;
    cudnnStatus_t cudnnDestroyAlgorithmDescriptor(cudnnAlgorithmStruct*) @nogc nothrow;
    cudnnStatus_t cudnnCopyAlgorithmDescriptor(const(cudnnAlgorithmStruct*), cudnnAlgorithmStruct*) @nogc nothrow;
    cudnnStatus_t cudnnGetAlgorithmDescriptor(const(cudnnAlgorithmStruct*), cudnnAlgorithm_t*) @nogc nothrow;
    cudnnStatus_t cudnnSetAlgorithmDescriptor(cudnnAlgorithmStruct*, cudnnAlgorithm_t) @nogc nothrow;
    cudnnStatus_t cudnnCreateAlgorithmDescriptor(cudnnAlgorithmStruct**) @nogc nothrow;
    struct cudnnAlgorithm_t
    {
        union Algorithm
        {
            cudnnConvolutionFwdAlgo_t convFwdAlgo;
            cudnnConvolutionBwdFilterAlgo_t convBwdFilterAlgo;
            cudnnConvolutionBwdDataAlgo_t convBwdDataAlgo;
            cudnnRNNAlgo_t RNNAlgo;
            cudnnCTCLossAlgo_t CTCLossAlgo;
        }
        cudnnAlgorithm_t.Algorithm algo;
    }
    cudnnStatus_t cudnnGetCTCLossWorkspaceSize(cudnnContext*, const(cudnnTensorStruct*), const(cudnnTensorStruct*), const(int)*, const(int)*, const(int)*, cudnnCTCLossAlgo_t, cudnnCTCLossStruct*, c_ulong*) @nogc nothrow;
    cudnnStatus_t cudnnCTCLoss(cudnnContext*, const(cudnnTensorStruct*), const(void)*, const(int)*, const(int)*, const(int)*, void*, const(cudnnTensorStruct*), const(void)*, cudnnCTCLossAlgo_t, cudnnCTCLossStruct*, void*, c_ulong) @nogc nothrow;
    cudnnStatus_t cudnnDestroyCTCLossDescriptor(cudnnCTCLossStruct*) @nogc nothrow;
    cudnnStatus_t cudnnGetCTCLossDescriptorEx(cudnnCTCLossStruct*, cudnnDataType_t*, cudnnLossNormalizationMode_t*, cudnnNanPropagation_t*) @nogc nothrow;
    cudnnStatus_t cudnnGetCTCLossDescriptor(cudnnCTCLossStruct*, cudnnDataType_t*) @nogc nothrow;
    cudnnStatus_t cudnnSetCTCLossDescriptorEx(cudnnCTCLossStruct*, cudnnDataType_t, cudnnLossNormalizationMode_t, cudnnNanPropagation_t) @nogc nothrow;
    cudnnStatus_t cudnnSetCTCLossDescriptor(cudnnCTCLossStruct*, cudnnDataType_t) @nogc nothrow;
    cudnnStatus_t cudnnCreateCTCLossDescriptor(cudnnCTCLossStruct**) @nogc nothrow;
    enum _Anonymous_22
    {
        CUDNN_LOSS_NORMALIZATION_NONE = 0,
        CUDNN_LOSS_NORMALIZATION_SOFTMAX = 1,
    }
    enum CUDNN_LOSS_NORMALIZATION_NONE = _Anonymous_22.CUDNN_LOSS_NORMALIZATION_NONE;
    enum CUDNN_LOSS_NORMALIZATION_SOFTMAX = _Anonymous_22.CUDNN_LOSS_NORMALIZATION_SOFTMAX;
    alias cudnnLossNormalizationMode_t = _Anonymous_22;
    enum _Anonymous_23
    {
        CUDNN_CTC_LOSS_ALGO_DETERMINISTIC = 0,
        CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC = 1,
    }
    enum CUDNN_CTC_LOSS_ALGO_DETERMINISTIC = _Anonymous_23.CUDNN_CTC_LOSS_ALGO_DETERMINISTIC;
    enum CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC = _Anonymous_23.CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC;
    alias cudnnCTCLossAlgo_t = _Anonymous_23;
    cudnnStatus_t cudnnMultiHeadAttnBackwardWeights(cudnnContext*, const(cudnnAttnStruct*), cudnnWgradMode_t, const(cudnnSeqDataStruct*), const(void)*, const(cudnnSeqDataStruct*), const(void)*, const(cudnnSeqDataStruct*), const(void)*, const(cudnnSeqDataStruct*), const(void)*, c_ulong, const(void)*, void*, c_ulong, void*, c_ulong, void*) @nogc nothrow;
    enum _Anonymous_24
    {
        CUDNN_WGRAD_MODE_ADD = 0,
        CUDNN_WGRAD_MODE_SET = 1,
    }
    enum CUDNN_WGRAD_MODE_ADD = _Anonymous_24.CUDNN_WGRAD_MODE_ADD;
    enum CUDNN_WGRAD_MODE_SET = _Anonymous_24.CUDNN_WGRAD_MODE_SET;
    alias cudnnWgradMode_t = _Anonymous_24;
    cudnnStatus_t cudnnMultiHeadAttnBackwardData(cudnnContext*, const(cudnnAttnStruct*), const(int)*, const(int)*, const(int)*, const(int)*, const(cudnnSeqDataStruct*), const(void)*, const(cudnnSeqDataStruct*), void*, const(void)*, const(cudnnSeqDataStruct*), void*, const(void)*, const(cudnnSeqDataStruct*), void*, const(void)*, c_ulong, const(void)*, c_ulong, void*, c_ulong, void*) @nogc nothrow;
    cudnnStatus_t cudnnMultiHeadAttnForward(cudnnContext*, const(cudnnAttnStruct*), int, const(int)*, const(int)*, const(int)*, const(int)*, const(cudnnSeqDataStruct*), const(void)*, const(void)*, const(cudnnSeqDataStruct*), const(void)*, const(cudnnSeqDataStruct*), const(void)*, const(cudnnSeqDataStruct*), void*, c_ulong, const(void)*, c_ulong, void*, c_ulong, void*) @nogc nothrow;
    cudnnStatus_t cudnnGetMultiHeadAttnWeights(cudnnContext*, const(cudnnAttnStruct*), cudnnMultiHeadAttnWeightKind_t, c_ulong, const(void)*, cudnnTensorStruct*, void**) @nogc nothrow;
    enum _Anonymous_25
    {
        CUDNN_MH_ATTN_Q_WEIGHTS = 0,
        CUDNN_MH_ATTN_K_WEIGHTS = 1,
        CUDNN_MH_ATTN_V_WEIGHTS = 2,
        CUDNN_MH_ATTN_O_WEIGHTS = 3,
        CUDNN_MH_ATTN_Q_BIASES = 4,
        CUDNN_MH_ATTN_K_BIASES = 5,
        CUDNN_MH_ATTN_V_BIASES = 6,
        CUDNN_MH_ATTN_O_BIASES = 7,
    }
    enum CUDNN_MH_ATTN_Q_WEIGHTS = _Anonymous_25.CUDNN_MH_ATTN_Q_WEIGHTS;
    enum CUDNN_MH_ATTN_K_WEIGHTS = _Anonymous_25.CUDNN_MH_ATTN_K_WEIGHTS;
    enum CUDNN_MH_ATTN_V_WEIGHTS = _Anonymous_25.CUDNN_MH_ATTN_V_WEIGHTS;
    enum CUDNN_MH_ATTN_O_WEIGHTS = _Anonymous_25.CUDNN_MH_ATTN_O_WEIGHTS;
    enum CUDNN_MH_ATTN_Q_BIASES = _Anonymous_25.CUDNN_MH_ATTN_Q_BIASES;
    enum CUDNN_MH_ATTN_K_BIASES = _Anonymous_25.CUDNN_MH_ATTN_K_BIASES;
    enum CUDNN_MH_ATTN_V_BIASES = _Anonymous_25.CUDNN_MH_ATTN_V_BIASES;
    enum CUDNN_MH_ATTN_O_BIASES = _Anonymous_25.CUDNN_MH_ATTN_O_BIASES;
    alias cudnnMultiHeadAttnWeightKind_t = _Anonymous_25;
    cudnnStatus_t cudnnGetMultiHeadAttnBuffers(cudnnContext*, const(cudnnAttnStruct*), c_ulong*, c_ulong*, c_ulong*) @nogc nothrow;
    cudnnStatus_t cudnnGetAttnDescriptor(cudnnAttnStruct*, uint*, int*, double*, cudnnDataType_t*, cudnnDataType_t*, cudnnMathType_t*, cudnnDropoutStruct**, cudnnDropoutStruct**, int*, int*, int*, int*, int*, int*, int*, int*, int*, int*, int*) @nogc nothrow;
    cudnnStatus_t cudnnSetAttnDescriptor(cudnnAttnStruct*, uint, int, double, cudnnDataType_t, cudnnDataType_t, cudnnMathType_t, cudnnDropoutStruct*, cudnnDropoutStruct*, int, int, int, int, int, int, int, int, int, int, int) @nogc nothrow;
    cudnnStatus_t cudnnDestroyAttnDescriptor(cudnnAttnStruct*) @nogc nothrow;
    cudnnStatus_t cudnnCreateAttnDescriptor(cudnnAttnStruct**) @nogc nothrow;
    alias cudnnAttnDescriptor_t = cudnnAttnStruct*;
    struct cudnnAttnStruct;
    alias cudnnAttnQueryMap_t = uint;
    cudnnStatus_t cudnnGetSeqDataDescriptor(const(cudnnSeqDataStruct*), cudnnDataType_t*, int*, int, int*, cudnnSeqDataAxis_t*, c_ulong*, c_ulong, int*, void*) @nogc nothrow;
    cudnnStatus_t cudnnSetSeqDataDescriptor(cudnnSeqDataStruct*, cudnnDataType_t, int, const(int)*, const(cudnnSeqDataAxis_t)*, c_ulong, const(int)*, void*) @nogc nothrow;
    cudnnStatus_t cudnnDestroySeqDataDescriptor(cudnnSeqDataStruct*) @nogc nothrow;
    cudnnStatus_t cudnnCreateSeqDataDescriptor(cudnnSeqDataStruct**) @nogc nothrow;
    alias cudnnSeqDataDescriptor_t = cudnnSeqDataStruct*;
    struct cudnnSeqDataStruct;
    enum _Anonymous_26
    {
        CUDNN_SEQDATA_TIME_DIM = 0,
        CUDNN_SEQDATA_BATCH_DIM = 1,
        CUDNN_SEQDATA_BEAM_DIM = 2,
        CUDNN_SEQDATA_VECT_DIM = 3,
    }
    enum CUDNN_SEQDATA_TIME_DIM = _Anonymous_26.CUDNN_SEQDATA_TIME_DIM;
    enum CUDNN_SEQDATA_BATCH_DIM = _Anonymous_26.CUDNN_SEQDATA_BATCH_DIM;
    enum CUDNN_SEQDATA_BEAM_DIM = _Anonymous_26.CUDNN_SEQDATA_BEAM_DIM;
    enum CUDNN_SEQDATA_VECT_DIM = _Anonymous_26.CUDNN_SEQDATA_VECT_DIM;
    alias cudnnSeqDataAxis_t = _Anonymous_26;
    cudnnStatus_t cudnnFindRNNBackwardWeightsAlgorithmEx(cudnnContext*, const(cudnnRNNStruct*), const(int), const(cudnnTensorStruct*)*, const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*)*, const(void)*, const(float), const(int), int*, cudnnAlgorithmPerformanceStruct**, const(void)*, c_ulong, const(cudnnFilterStruct*), void*, const(void)*, c_ulong) @nogc nothrow;
    cudnnStatus_t cudnnGetRNNBackwardWeightsAlgorithmMaxCount(cudnnContext*, const(cudnnRNNStruct*), int*) @nogc nothrow;
    cudnnStatus_t cudnnFindRNNBackwardDataAlgorithmEx(cudnnContext*, const(cudnnRNNStruct*), const(int), const(cudnnTensorStruct*)*, const(void)*, const(cudnnTensorStruct*)*, const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnFilterStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*)*, void*, const(cudnnTensorStruct*), void*, const(cudnnTensorStruct*), void*, const(float), const(int), int*, cudnnAlgorithmPerformanceStruct**, void*, c_ulong, void*, c_ulong) @nogc nothrow;
    cudnnStatus_t cudnnGetRNNBackwardDataAlgorithmMaxCount(cudnnContext*, const(cudnnRNNStruct*), int*) @nogc nothrow;
    cudnnStatus_t cudnnFindRNNForwardTrainingAlgorithmEx(cudnnContext*, const(cudnnRNNStruct*), const(int), const(cudnnTensorStruct*)*, const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnFilterStruct*), const(void)*, const(cudnnTensorStruct*)*, void*, const(cudnnTensorStruct*), void*, const(cudnnTensorStruct*), void*, const(float), const(int), int*, cudnnAlgorithmPerformanceStruct**, void*, c_ulong, void*, c_ulong) @nogc nothrow;
    cudnnStatus_t cudnnGetRNNForwardTrainingAlgorithmMaxCount(cudnnContext*, const(cudnnRNNStruct*), int*) @nogc nothrow;
    cudnnStatus_t cudnnFindRNNForwardInferenceAlgorithmEx(cudnnContext*, const(cudnnRNNStruct*), const(int), const(cudnnTensorStruct*)*, const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnFilterStruct*), const(void)*, const(cudnnTensorStruct*)*, void*, const(cudnnTensorStruct*), void*, const(cudnnTensorStruct*), void*, const(float), const(int), int*, cudnnAlgorithmPerformanceStruct**, void*, c_ulong) @nogc nothrow;
    cudnnStatus_t cudnnGetRNNForwardInferenceAlgorithmMaxCount(cudnnContext*, const(cudnnRNNStruct*), int*) @nogc nothrow;
    cudnnStatus_t cudnnSetRNNAlgorithmDescriptor(cudnnContext*, cudnnRNNStruct*, cudnnAlgorithmStruct*) @nogc nothrow;
    struct cudnnAlgorithmPerformanceStruct;
    alias cudnnAlgorithmPerformance_t = cudnnAlgorithmPerformanceStruct*;
    struct cudnnAlgorithmStruct;
    alias cudnnAlgorithmDescriptor_t = cudnnAlgorithmStruct*;
    cudnnStatus_t cudnnRNNBackwardWeightsEx(cudnnContext*, const(cudnnRNNStruct*), const(cudnnRNNDataStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnRNNDataStruct*), const(void)*, void*, c_ulong, const(cudnnFilterStruct*), void*, void*, c_ulong) @nogc nothrow;
    cudnnStatus_t cudnnRNNBackwardDataEx(cudnnContext*, const(cudnnRNNStruct*), const(cudnnRNNDataStruct*), const(void)*, const(cudnnRNNDataStruct*), const(void)*, const(cudnnRNNDataStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnFilterStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnRNNDataStruct*), void*, const(cudnnTensorStruct*), void*, const(cudnnTensorStruct*), void*, const(cudnnRNNDataStruct*), void*, void*, c_ulong, void*, c_ulong) @nogc nothrow;
    cudnnStatus_t cudnnRNNForwardInferenceEx(cudnnContext*, const(cudnnRNNStruct*), const(cudnnRNNDataStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnFilterStruct*), const(void)*, const(cudnnRNNDataStruct*), void*, const(cudnnTensorStruct*), void*, const(cudnnTensorStruct*), void*, const(cudnnRNNDataStruct*), const(void)*, const(cudnnRNNDataStruct*), void*, const(cudnnRNNDataStruct*), void*, const(cudnnRNNDataStruct*), void*, void*, c_ulong) @nogc nothrow;
    cudnnStatus_t cudnnRNNForwardTrainingEx(cudnnContext*, const(cudnnRNNStruct*), const(cudnnRNNDataStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnFilterStruct*), const(void)*, const(cudnnRNNDataStruct*), void*, const(cudnnTensorStruct*), void*, const(cudnnTensorStruct*), void*, const(cudnnRNNDataStruct*), const(void)*, const(cudnnRNNDataStruct*), void*, const(cudnnRNNDataStruct*), void*, const(cudnnRNNDataStruct*), void*, void*, c_ulong, void*, c_ulong) @nogc nothrow;
    cudnnStatus_t cudnnGetRNNDataDescriptor(cudnnRNNDataStruct*, cudnnDataType_t*, cudnnRNNDataLayout_t*, int*, int*, int*, int, int*, void*) @nogc nothrow;
    cudnnStatus_t cudnnSetRNNDataDescriptor(cudnnRNNDataStruct*, cudnnDataType_t, cudnnRNNDataLayout_t, int, int, int, const(int)*, void*) @nogc nothrow;
    cudnnStatus_t cudnnDestroyRNNDataDescriptor(cudnnRNNDataStruct*) @nogc nothrow;
    cudnnStatus_t cudnnCreateRNNDataDescriptor(cudnnRNNDataStruct**) @nogc nothrow;
    cudnnStatus_t cudnnGetRNNPaddingMode(cudnnRNNStruct*, cudnnRNNPaddingMode_t*) @nogc nothrow;
    cudnnStatus_t cudnnSetRNNPaddingMode(cudnnRNNStruct*, cudnnRNNPaddingMode_t) @nogc nothrow;
    cudnnStatus_t cudnnRNNBackwardWeights(cudnnContext*, const(cudnnRNNStruct*), const(int), const(cudnnTensorStruct*)*, const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*)*, const(void)*, const(void)*, c_ulong, const(cudnnFilterStruct*), void*, const(void)*, c_ulong) @nogc nothrow;
    cudnnStatus_t cudnnRNNBackwardData(cudnnContext*, const(cudnnRNNStruct*), const(int), const(cudnnTensorStruct*)*, const(void)*, const(cudnnTensorStruct*)*, const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnFilterStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*)*, void*, const(cudnnTensorStruct*), void*, const(cudnnTensorStruct*), void*, void*, c_ulong, void*, c_ulong) @nogc nothrow;
    cudnnStatus_t cudnnRNNForwardTraining(cudnnContext*, const(cudnnRNNStruct*), const(int), const(cudnnTensorStruct*)*, const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnFilterStruct*), const(void)*, const(cudnnTensorStruct*)*, void*, const(cudnnTensorStruct*), void*, const(cudnnTensorStruct*), void*, void*, c_ulong, void*, c_ulong) @nogc nothrow;
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
    cudaError cudaMemcpyFromArrayAsync(void*, const(cudaArray)*, c_ulong, c_ulong, c_ulong, cudaMemcpyKind, CUstream_st*) @nogc nothrow;
    cudaError cudaPointerGetAttributes(cudaPointerAttributes*, const(void)*) @nogc nothrow;
    cudaError cudaDeviceCanAccessPeer(int*, int, int) @nogc nothrow;
    cudaError cudaDeviceEnablePeerAccess(int, uint) @nogc nothrow;
    cudaError cudaDeviceDisablePeerAccess(int) @nogc nothrow;
    cudaError cudaGraphicsUnregisterResource(cudaGraphicsResource*) @nogc nothrow;
    cudaError cudaGraphicsResourceSetMapFlags(cudaGraphicsResource*, uint) @nogc nothrow;
    cudaError cudaGraphicsMapResources(int, cudaGraphicsResource**, CUstream_st*) @nogc nothrow;
    cudaError cudaGraphicsUnmapResources(int, cudaGraphicsResource**, CUstream_st*) @nogc nothrow;
    cudaError cudaGraphicsResourceGetMappedPointer(void**, c_ulong*, cudaGraphicsResource*) @nogc nothrow;
    cudaError cudaGraphicsSubResourceGetMappedArray(cudaArray**, cudaGraphicsResource*, uint, uint) @nogc nothrow;
    cudaError cudaGraphicsResourceGetMappedMipmappedArray(cudaMipmappedArray**, cudaGraphicsResource*) @nogc nothrow;
    cudaError cudaBindTexture(c_ulong*, const(textureReference)*, const(void)*, const(cudaChannelFormatDesc)*, c_ulong) @nogc nothrow;
    cudaError cudaBindTexture2D(c_ulong*, const(textureReference)*, const(void)*, const(cudaChannelFormatDesc)*, c_ulong, c_ulong, c_ulong) @nogc nothrow;
    cudaError cudaBindTextureToArray(const(textureReference)*, const(cudaArray)*, const(cudaChannelFormatDesc)*) @nogc nothrow;
    cudaError cudaBindTextureToMipmappedArray(const(textureReference)*, const(cudaMipmappedArray)*, const(cudaChannelFormatDesc)*) @nogc nothrow;
    cudaError cudaUnbindTexture(const(textureReference)*) @nogc nothrow;
    cudaError cudaGetTextureAlignmentOffset(c_ulong*, const(textureReference)*) @nogc nothrow;
    cudaError cudaGetTextureReference(const(textureReference)**, const(void)*) @nogc nothrow;
    cudaError cudaBindSurfaceToArray(const(surfaceReference)*, const(cudaArray)*, const(cudaChannelFormatDesc)*) @nogc nothrow;
    cudaError cudaGetSurfaceReference(const(surfaceReference)**, const(void)*) @nogc nothrow;
    cudaError cudaGetChannelDesc(cudaChannelFormatDesc*, const(cudaArray)*) @nogc nothrow;
    cudaChannelFormatDesc cudaCreateChannelDesc(int, int, int, int, cudaChannelFormatKind) @nogc nothrow;
    cudaError cudaCreateTextureObject(ulong*, const(cudaResourceDesc)*, const(cudaTextureDesc)*, const(cudaResourceViewDesc)*) @nogc nothrow;
    cudaError cudaDestroyTextureObject(ulong) @nogc nothrow;
    cudaError cudaGetTextureObjectResourceDesc(cudaResourceDesc*, ulong) @nogc nothrow;
    cudaError cudaGetTextureObjectTextureDesc(cudaTextureDesc*, ulong) @nogc nothrow;
    cudaError cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc*, ulong) @nogc nothrow;
    cudaError cudaCreateSurfaceObject(ulong*, const(cudaResourceDesc)*) @nogc nothrow;
    cudaError cudaDestroySurfaceObject(ulong) @nogc nothrow;
    cudaError cudaGetSurfaceObjectResourceDesc(cudaResourceDesc*, ulong) @nogc nothrow;
    cudaError cudaDriverGetVersion(int*) @nogc nothrow;
    cudaError cudaRuntimeGetVersion(int*) @nogc nothrow;
    cudaError cudaGraphCreate(CUgraph_st**, uint) @nogc nothrow;
    cudaError cudaGraphAddKernelNode(CUgraphNode_st**, CUgraph_st*, const(CUgraphNode_st*)*, c_ulong, const(cudaKernelNodeParams)*) @nogc nothrow;
    cudaError cudaGraphKernelNodeGetParams(CUgraphNode_st*, cudaKernelNodeParams*) @nogc nothrow;
    cudaError cudaGraphKernelNodeSetParams(CUgraphNode_st*, const(cudaKernelNodeParams)*) @nogc nothrow;
    cudaError cudaGraphAddMemcpyNode(CUgraphNode_st**, CUgraph_st*, const(CUgraphNode_st*)*, c_ulong, const(cudaMemcpy3DParms)*) @nogc nothrow;
    cudaError cudaGraphMemcpyNodeGetParams(CUgraphNode_st*, cudaMemcpy3DParms*) @nogc nothrow;
    cudaError cudaGraphMemcpyNodeSetParams(CUgraphNode_st*, const(cudaMemcpy3DParms)*) @nogc nothrow;
    cudaError cudaGraphAddMemsetNode(CUgraphNode_st**, CUgraph_st*, const(CUgraphNode_st*)*, c_ulong, const(cudaMemsetParams)*) @nogc nothrow;
    cudaError cudaGraphMemsetNodeGetParams(CUgraphNode_st*, cudaMemsetParams*) @nogc nothrow;
    cudaError cudaGraphMemsetNodeSetParams(CUgraphNode_st*, const(cudaMemsetParams)*) @nogc nothrow;
    cudaError cudaGraphAddHostNode(CUgraphNode_st**, CUgraph_st*, const(CUgraphNode_st*)*, c_ulong, const(cudaHostNodeParams)*) @nogc nothrow;
    cudaError cudaGraphHostNodeGetParams(CUgraphNode_st*, cudaHostNodeParams*) @nogc nothrow;
    cudaError cudaGraphHostNodeSetParams(CUgraphNode_st*, const(cudaHostNodeParams)*) @nogc nothrow;
    cudaError cudaGraphAddChildGraphNode(CUgraphNode_st**, CUgraph_st*, const(CUgraphNode_st*)*, c_ulong, CUgraph_st*) @nogc nothrow;
    cudaError cudaGraphChildGraphNodeGetGraph(CUgraphNode_st*, CUgraph_st**) @nogc nothrow;
    cudaError cudaGraphAddEmptyNode(CUgraphNode_st**, CUgraph_st*, const(CUgraphNode_st*)*, c_ulong) @nogc nothrow;
    cudaError cudaGraphClone(CUgraph_st**, CUgraph_st*) @nogc nothrow;
    cudaError cudaGraphNodeFindInClone(CUgraphNode_st**, CUgraphNode_st*, CUgraph_st*) @nogc nothrow;
    cudaError cudaGraphNodeGetType(CUgraphNode_st*, cudaGraphNodeType*) @nogc nothrow;
    cudaError cudaGraphGetNodes(CUgraph_st*, CUgraphNode_st**, c_ulong*) @nogc nothrow;
    cudaError cudaGraphGetRootNodes(CUgraph_st*, CUgraphNode_st**, c_ulong*) @nogc nothrow;
    cudaError cudaGraphGetEdges(CUgraph_st*, CUgraphNode_st**, CUgraphNode_st**, c_ulong*) @nogc nothrow;
    cudaError cudaGraphNodeGetDependencies(CUgraphNode_st*, CUgraphNode_st**, c_ulong*) @nogc nothrow;
    cudaError cudaGraphNodeGetDependentNodes(CUgraphNode_st*, CUgraphNode_st**, c_ulong*) @nogc nothrow;
    cudaError cudaGraphAddDependencies(CUgraph_st*, const(CUgraphNode_st*)*, const(CUgraphNode_st*)*, c_ulong) @nogc nothrow;
    cudaError cudaGraphRemoveDependencies(CUgraph_st*, const(CUgraphNode_st*)*, const(CUgraphNode_st*)*, c_ulong) @nogc nothrow;
    cudaError cudaGraphDestroyNode(CUgraphNode_st*) @nogc nothrow;
    cudaError cudaGraphInstantiate(CUgraphExec_st**, CUgraph_st*, CUgraphNode_st**, char*, c_ulong) @nogc nothrow;
    cudaError cudaGraphExecKernelNodeSetParams(CUgraphExec_st*, CUgraphNode_st*, const(cudaKernelNodeParams)*) @nogc nothrow;
    cudaError cudaGraphExecMemcpyNodeSetParams(CUgraphExec_st*, CUgraphNode_st*, const(cudaMemcpy3DParms)*) @nogc nothrow;
    cudaError cudaGraphExecMemsetNodeSetParams(CUgraphExec_st*, CUgraphNode_st*, const(cudaMemsetParams)*) @nogc nothrow;
    cudaError cudaGraphExecHostNodeSetParams(CUgraphExec_st*, CUgraphNode_st*, const(cudaHostNodeParams)*) @nogc nothrow;
    cudaError cudaGraphExecUpdate(CUgraphExec_st*, CUgraph_st*, CUgraphNode_st**, cudaGraphExecUpdateResult*) @nogc nothrow;
    cudaError cudaGraphLaunch(CUgraphExec_st*, CUstream_st*) @nogc nothrow;
    cudaError cudaGraphExecDestroy(CUgraphExec_st*) @nogc nothrow;
    cudaError cudaGraphDestroy(CUgraph_st*) @nogc nothrow;
    cudaError cudaGetExportTable(const(void)**, const(CUuuid_st)*) @nogc nothrow;
    cudnnStatus_t cudnnRNNForwardInference(cudnnContext*, const(cudnnRNNStruct*), const(int), const(cudnnTensorStruct*)*, const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnFilterStruct*), const(void)*, const(cudnnTensorStruct*)*, void*, const(cudnnTensorStruct*), void*, const(cudnnTensorStruct*), void*, void*, c_ulong) @nogc nothrow;
    cudnnStatus_t cudnnGetRNNLinLayerBiasParams(cudnnContext*, const(cudnnRNNStruct*), const(int), const(cudnnTensorStruct*), const(cudnnFilterStruct*), const(void)*, const(int), cudnnFilterStruct*, void**) @nogc nothrow;
    cudnnStatus_t cudnnGetRNNLinLayerMatrixParams(cudnnContext*, const(cudnnRNNStruct*), const(int), const(cudnnTensorStruct*), const(cudnnFilterStruct*), const(void)*, const(int), cudnnFilterStruct*, void**) @nogc nothrow;
    struct cudnnContext;
    alias cudnnHandle_t = cudnnContext*;
    c_ulong cudnnGetVersion() @nogc nothrow;
    c_ulong cudnnGetCudartVersion() @nogc nothrow;
    alias cudnnStatus_t = _Anonymous_27;
    enum _Anonymous_27
    {
        CUDNN_STATUS_SUCCESS = 0,
        CUDNN_STATUS_NOT_INITIALIZED = 1,
        CUDNN_STATUS_ALLOC_FAILED = 2,
        CUDNN_STATUS_BAD_PARAM = 3,
        CUDNN_STATUS_INTERNAL_ERROR = 4,
        CUDNN_STATUS_INVALID_VALUE = 5,
        CUDNN_STATUS_ARCH_MISMATCH = 6,
        CUDNN_STATUS_MAPPING_ERROR = 7,
        CUDNN_STATUS_EXECUTION_FAILED = 8,
        CUDNN_STATUS_NOT_SUPPORTED = 9,
        CUDNN_STATUS_LICENSE_ERROR = 10,
        CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = 11,
        CUDNN_STATUS_RUNTIME_IN_PROGRESS = 12,
        CUDNN_STATUS_RUNTIME_FP_OVERFLOW = 13,
    }
    enum CUDNN_STATUS_SUCCESS = _Anonymous_27.CUDNN_STATUS_SUCCESS;
    enum CUDNN_STATUS_NOT_INITIALIZED = _Anonymous_27.CUDNN_STATUS_NOT_INITIALIZED;
    enum CUDNN_STATUS_ALLOC_FAILED = _Anonymous_27.CUDNN_STATUS_ALLOC_FAILED;
    enum CUDNN_STATUS_BAD_PARAM = _Anonymous_27.CUDNN_STATUS_BAD_PARAM;
    enum CUDNN_STATUS_INTERNAL_ERROR = _Anonymous_27.CUDNN_STATUS_INTERNAL_ERROR;
    enum CUDNN_STATUS_INVALID_VALUE = _Anonymous_27.CUDNN_STATUS_INVALID_VALUE;
    enum CUDNN_STATUS_ARCH_MISMATCH = _Anonymous_27.CUDNN_STATUS_ARCH_MISMATCH;
    enum CUDNN_STATUS_MAPPING_ERROR = _Anonymous_27.CUDNN_STATUS_MAPPING_ERROR;
    enum CUDNN_STATUS_EXECUTION_FAILED = _Anonymous_27.CUDNN_STATUS_EXECUTION_FAILED;
    enum CUDNN_STATUS_NOT_SUPPORTED = _Anonymous_27.CUDNN_STATUS_NOT_SUPPORTED;
    enum CUDNN_STATUS_LICENSE_ERROR = _Anonymous_27.CUDNN_STATUS_LICENSE_ERROR;
    enum CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = _Anonymous_27.CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING;
    enum CUDNN_STATUS_RUNTIME_IN_PROGRESS = _Anonymous_27.CUDNN_STATUS_RUNTIME_IN_PROGRESS;
    enum CUDNN_STATUS_RUNTIME_FP_OVERFLOW = _Anonymous_27.CUDNN_STATUS_RUNTIME_FP_OVERFLOW;
    const(char)* cudnnGetErrorString(cudnnStatus_t) @nogc nothrow;
    struct cudnnRuntimeTag_t;
    alias cudnnErrQueryMode_t = _Anonymous_28;
    enum _Anonymous_28
    {
        CUDNN_ERRQUERY_RAWCODE = 0,
        CUDNN_ERRQUERY_NONBLOCKING = 1,
        CUDNN_ERRQUERY_BLOCKING = 2,
    }
    enum CUDNN_ERRQUERY_RAWCODE = _Anonymous_28.CUDNN_ERRQUERY_RAWCODE;
    enum CUDNN_ERRQUERY_NONBLOCKING = _Anonymous_28.CUDNN_ERRQUERY_NONBLOCKING;
    enum CUDNN_ERRQUERY_BLOCKING = _Anonymous_28.CUDNN_ERRQUERY_BLOCKING;
    cudnnStatus_t cudnnQueryRuntimeError(cudnnContext*, cudnnStatus_t*, cudnnErrQueryMode_t, cudnnRuntimeTag_t*) @nogc nothrow;
    cudnnStatus_t cudnnGetProperty(libraryPropertyType_t, int*) @nogc nothrow;
    cudnnStatus_t cudnnCreate(cudnnContext**) @nogc nothrow;
    cudnnStatus_t cudnnDestroy(cudnnContext*) @nogc nothrow;
    cudnnStatus_t cudnnSetStream(cudnnContext*, CUstream_st*) @nogc nothrow;
    cudnnStatus_t cudnnGetStream(cudnnContext*, CUstream_st**) @nogc nothrow;
    alias cudnnTensorDescriptor_t = cudnnTensorStruct*;
    struct cudnnTensorStruct;
    alias cudnnConvolutionDescriptor_t = cudnnConvolutionStruct*;
    struct cudnnConvolutionStruct;
    alias cudnnPoolingDescriptor_t = cudnnPoolingStruct*;
    struct cudnnPoolingStruct;
    alias cudnnFilterDescriptor_t = cudnnFilterStruct*;
    struct cudnnFilterStruct;
    alias cudnnLRNDescriptor_t = cudnnLRNStruct*;
    struct cudnnLRNStruct;
    alias cudnnActivationDescriptor_t = cudnnActivationStruct*;
    struct cudnnActivationStruct;
    alias cudnnSpatialTransformerDescriptor_t = cudnnSpatialTransformerStruct*;
    struct cudnnSpatialTransformerStruct;
    alias cudnnOpTensorDescriptor_t = cudnnOpTensorStruct*;
    struct cudnnOpTensorStruct;
    alias cudnnReduceTensorDescriptor_t = cudnnReduceTensorStruct*;
    struct cudnnReduceTensorStruct;
    alias cudnnCTCLossDescriptor_t = cudnnCTCLossStruct*;
    struct cudnnCTCLossStruct;
    alias cudnnTensorTransformDescriptor_t = cudnnTensorTransformStruct*;
    struct cudnnTensorTransformStruct;
    alias cudnnDataType_t = _Anonymous_29;
    enum _Anonymous_29
    {
        CUDNN_DATA_FLOAT = 0,
        CUDNN_DATA_DOUBLE = 1,
        CUDNN_DATA_HALF = 2,
        CUDNN_DATA_INT8 = 3,
        CUDNN_DATA_INT32 = 4,
        CUDNN_DATA_INT8x4 = 5,
        CUDNN_DATA_UINT8 = 6,
        CUDNN_DATA_UINT8x4 = 7,
        CUDNN_DATA_INT8x32 = 8,
    }
    enum CUDNN_DATA_FLOAT = _Anonymous_29.CUDNN_DATA_FLOAT;
    enum CUDNN_DATA_DOUBLE = _Anonymous_29.CUDNN_DATA_DOUBLE;
    enum CUDNN_DATA_HALF = _Anonymous_29.CUDNN_DATA_HALF;
    enum CUDNN_DATA_INT8 = _Anonymous_29.CUDNN_DATA_INT8;
    enum CUDNN_DATA_INT32 = _Anonymous_29.CUDNN_DATA_INT32;
    enum CUDNN_DATA_INT8x4 = _Anonymous_29.CUDNN_DATA_INT8x4;
    enum CUDNN_DATA_UINT8 = _Anonymous_29.CUDNN_DATA_UINT8;
    enum CUDNN_DATA_UINT8x4 = _Anonymous_29.CUDNN_DATA_UINT8x4;
    enum CUDNN_DATA_INT8x32 = _Anonymous_29.CUDNN_DATA_INT8x32;
    alias cudnnMathType_t = _Anonymous_30;
    enum _Anonymous_30
    {
        CUDNN_DEFAULT_MATH = 0,
        CUDNN_TENSOR_OP_MATH = 1,
        CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION = 2,
    }
    enum CUDNN_DEFAULT_MATH = _Anonymous_30.CUDNN_DEFAULT_MATH;
    enum CUDNN_TENSOR_OP_MATH = _Anonymous_30.CUDNN_TENSOR_OP_MATH;
    enum CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION = _Anonymous_30.CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
    alias cudnnNanPropagation_t = _Anonymous_31;
    enum _Anonymous_31
    {
        CUDNN_NOT_PROPAGATE_NAN = 0,
        CUDNN_PROPAGATE_NAN = 1,
    }
    enum CUDNN_NOT_PROPAGATE_NAN = _Anonymous_31.CUDNN_NOT_PROPAGATE_NAN;
    enum CUDNN_PROPAGATE_NAN = _Anonymous_31.CUDNN_PROPAGATE_NAN;
    alias cudnnDeterminism_t = _Anonymous_32;
    enum _Anonymous_32
    {
        CUDNN_NON_DETERMINISTIC = 0,
        CUDNN_DETERMINISTIC = 1,
    }
    enum CUDNN_NON_DETERMINISTIC = _Anonymous_32.CUDNN_NON_DETERMINISTIC;
    enum CUDNN_DETERMINISTIC = _Anonymous_32.CUDNN_DETERMINISTIC;
    alias cudnnReorderType_t = _Anonymous_33;
    enum _Anonymous_33
    {
        CUDNN_DEFAULT_REORDER = 0,
        CUDNN_NO_REORDER = 1,
    }
    enum CUDNN_DEFAULT_REORDER = _Anonymous_33.CUDNN_DEFAULT_REORDER;
    enum CUDNN_NO_REORDER = _Anonymous_33.CUDNN_NO_REORDER;
    cudnnStatus_t cudnnGetRNNParamsSize(cudnnContext*, const(cudnnRNNStruct*), const(cudnnTensorStruct*), c_ulong*, cudnnDataType_t) @nogc nothrow;
    cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorStruct**) @nogc nothrow;
    alias cudnnTensorFormat_t = _Anonymous_34;
    enum _Anonymous_34
    {
        CUDNN_TENSOR_NCHW = 0,
        CUDNN_TENSOR_NHWC = 1,
        CUDNN_TENSOR_NCHW_VECT_C = 2,
    }
    enum CUDNN_TENSOR_NCHW = _Anonymous_34.CUDNN_TENSOR_NCHW;
    enum CUDNN_TENSOR_NHWC = _Anonymous_34.CUDNN_TENSOR_NHWC;
    enum CUDNN_TENSOR_NCHW_VECT_C = _Anonymous_34.CUDNN_TENSOR_NCHW_VECT_C;
    cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorStruct*, cudnnTensorFormat_t, cudnnDataType_t, int, int, int, int) @nogc nothrow;
    cudnnStatus_t cudnnSetTensor4dDescriptorEx(cudnnTensorStruct*, cudnnDataType_t, int, int, int, int, int, int, int, int) @nogc nothrow;
    cudnnStatus_t cudnnGetTensor4dDescriptor(const(cudnnTensorStruct*), cudnnDataType_t*, int*, int*, int*, int*, int*, int*, int*, int*) @nogc nothrow;
    cudnnStatus_t cudnnSetTensorNdDescriptor(cudnnTensorStruct*, cudnnDataType_t, int, const(int)*, const(int)*) @nogc nothrow;
    cudnnStatus_t cudnnSetTensorNdDescriptorEx(cudnnTensorStruct*, cudnnTensorFormat_t, cudnnDataType_t, int, const(int)*) @nogc nothrow;
    cudnnStatus_t cudnnGetTensorNdDescriptor(const(cudnnTensorStruct*), int, cudnnDataType_t*, int*, int*, int*) @nogc nothrow;
    cudnnStatus_t cudnnGetTensorSizeInBytes(const(cudnnTensorStruct*), c_ulong*) @nogc nothrow;
    cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorStruct*) @nogc nothrow;
    alias cudnnFoldingDirection_t = _Anonymous_35;
    enum _Anonymous_35
    {
        CUDNN_TRANSFORM_FOLD = 0,
        CUDNN_TRANSFORM_UNFOLD = 1,
    }
    enum CUDNN_TRANSFORM_FOLD = _Anonymous_35.CUDNN_TRANSFORM_FOLD;
    enum CUDNN_TRANSFORM_UNFOLD = _Anonymous_35.CUDNN_TRANSFORM_UNFOLD;
    cudnnStatus_t cudnnInitTransformDest(const(cudnnTensorTransformStruct*), const(cudnnTensorStruct*), cudnnTensorStruct*, c_ulong*) @nogc nothrow;
    cudnnStatus_t cudnnCreateTensorTransformDescriptor(cudnnTensorTransformStruct**) @nogc nothrow;
    cudnnStatus_t cudnnSetTensorTransformDescriptor(cudnnTensorTransformStruct*, const(uint), const(cudnnTensorFormat_t), const(int)*, const(int)*, const(uint)*, const(cudnnFoldingDirection_t)) @nogc nothrow;
    cudnnStatus_t cudnnGetTensorTransformDescriptor(cudnnTensorTransformStruct*, uint, cudnnTensorFormat_t*, int*, int*, uint*, cudnnFoldingDirection_t*) @nogc nothrow;
    cudnnStatus_t cudnnDestroyTensorTransformDescriptor(cudnnTensorTransformStruct*) @nogc nothrow;
    cudnnStatus_t cudnnTransformTensor(cudnnContext*, const(void)*, const(cudnnTensorStruct*), const(void)*, const(void)*, const(cudnnTensorStruct*), void*) @nogc nothrow;
    cudnnStatus_t cudnnTransformTensorEx(cudnnContext*, const(cudnnTensorTransformStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(void)*, const(cudnnTensorStruct*), void*) @nogc nothrow;
    cudnnStatus_t cudnnGetFoldedConvBackwardDataDescriptors(const(cudnnContext*), const(cudnnFilterStruct*), const(cudnnTensorStruct*), const(cudnnConvolutionStruct*), const(cudnnTensorStruct*), const(cudnnTensorFormat_t), cudnnFilterStruct*, cudnnTensorStruct*, cudnnConvolutionStruct*, cudnnTensorStruct*, cudnnTensorTransformStruct*, cudnnTensorTransformStruct*, cudnnTensorTransformStruct*, cudnnTensorTransformStruct*) @nogc nothrow;
    cudnnStatus_t cudnnAddTensor(cudnnContext*, const(void)*, const(cudnnTensorStruct*), const(void)*, const(void)*, const(cudnnTensorStruct*), void*) @nogc nothrow;
    alias cudnnOpTensorOp_t = _Anonymous_36;
    enum _Anonymous_36
    {
        CUDNN_OP_TENSOR_ADD = 0,
        CUDNN_OP_TENSOR_MUL = 1,
        CUDNN_OP_TENSOR_MIN = 2,
        CUDNN_OP_TENSOR_MAX = 3,
        CUDNN_OP_TENSOR_SQRT = 4,
        CUDNN_OP_TENSOR_NOT = 5,
    }
    enum CUDNN_OP_TENSOR_ADD = _Anonymous_36.CUDNN_OP_TENSOR_ADD;
    enum CUDNN_OP_TENSOR_MUL = _Anonymous_36.CUDNN_OP_TENSOR_MUL;
    enum CUDNN_OP_TENSOR_MIN = _Anonymous_36.CUDNN_OP_TENSOR_MIN;
    enum CUDNN_OP_TENSOR_MAX = _Anonymous_36.CUDNN_OP_TENSOR_MAX;
    enum CUDNN_OP_TENSOR_SQRT = _Anonymous_36.CUDNN_OP_TENSOR_SQRT;
    enum CUDNN_OP_TENSOR_NOT = _Anonymous_36.CUDNN_OP_TENSOR_NOT;
    cudnnStatus_t cudnnCreateOpTensorDescriptor(cudnnOpTensorStruct**) @nogc nothrow;
    cudnnStatus_t cudnnSetOpTensorDescriptor(cudnnOpTensorStruct*, cudnnOpTensorOp_t, cudnnDataType_t, cudnnNanPropagation_t) @nogc nothrow;
    cudnnStatus_t cudnnGetOpTensorDescriptor(const(cudnnOpTensorStruct*), cudnnOpTensorOp_t*, cudnnDataType_t*, cudnnNanPropagation_t*) @nogc nothrow;
    cudnnStatus_t cudnnDestroyOpTensorDescriptor(cudnnOpTensorStruct*) @nogc nothrow;
    cudnnStatus_t cudnnOpTensor(cudnnContext*, const(cudnnOpTensorStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(void)*, const(cudnnTensorStruct*), const(void)*, const(void)*, const(cudnnTensorStruct*), void*) @nogc nothrow;
    alias cudnnReduceTensorOp_t = _Anonymous_37;
    enum _Anonymous_37
    {
        CUDNN_REDUCE_TENSOR_ADD = 0,
        CUDNN_REDUCE_TENSOR_MUL = 1,
        CUDNN_REDUCE_TENSOR_MIN = 2,
        CUDNN_REDUCE_TENSOR_MAX = 3,
        CUDNN_REDUCE_TENSOR_AMAX = 4,
        CUDNN_REDUCE_TENSOR_AVG = 5,
        CUDNN_REDUCE_TENSOR_NORM1 = 6,
        CUDNN_REDUCE_TENSOR_NORM2 = 7,
        CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS = 8,
    }
    enum CUDNN_REDUCE_TENSOR_ADD = _Anonymous_37.CUDNN_REDUCE_TENSOR_ADD;
    enum CUDNN_REDUCE_TENSOR_MUL = _Anonymous_37.CUDNN_REDUCE_TENSOR_MUL;
    enum CUDNN_REDUCE_TENSOR_MIN = _Anonymous_37.CUDNN_REDUCE_TENSOR_MIN;
    enum CUDNN_REDUCE_TENSOR_MAX = _Anonymous_37.CUDNN_REDUCE_TENSOR_MAX;
    enum CUDNN_REDUCE_TENSOR_AMAX = _Anonymous_37.CUDNN_REDUCE_TENSOR_AMAX;
    enum CUDNN_REDUCE_TENSOR_AVG = _Anonymous_37.CUDNN_REDUCE_TENSOR_AVG;
    enum CUDNN_REDUCE_TENSOR_NORM1 = _Anonymous_37.CUDNN_REDUCE_TENSOR_NORM1;
    enum CUDNN_REDUCE_TENSOR_NORM2 = _Anonymous_37.CUDNN_REDUCE_TENSOR_NORM2;
    enum CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS = _Anonymous_37.CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS;
    alias cudnnReduceTensorIndices_t = _Anonymous_38;
    enum _Anonymous_38
    {
        CUDNN_REDUCE_TENSOR_NO_INDICES = 0,
        CUDNN_REDUCE_TENSOR_FLATTENED_INDICES = 1,
    }
    enum CUDNN_REDUCE_TENSOR_NO_INDICES = _Anonymous_38.CUDNN_REDUCE_TENSOR_NO_INDICES;
    enum CUDNN_REDUCE_TENSOR_FLATTENED_INDICES = _Anonymous_38.CUDNN_REDUCE_TENSOR_FLATTENED_INDICES;
    alias cudnnIndicesType_t = _Anonymous_39;
    enum _Anonymous_39
    {
        CUDNN_32BIT_INDICES = 0,
        CUDNN_64BIT_INDICES = 1,
        CUDNN_16BIT_INDICES = 2,
        CUDNN_8BIT_INDICES = 3,
    }
    enum CUDNN_32BIT_INDICES = _Anonymous_39.CUDNN_32BIT_INDICES;
    enum CUDNN_64BIT_INDICES = _Anonymous_39.CUDNN_64BIT_INDICES;
    enum CUDNN_16BIT_INDICES = _Anonymous_39.CUDNN_16BIT_INDICES;
    enum CUDNN_8BIT_INDICES = _Anonymous_39.CUDNN_8BIT_INDICES;
    cudnnStatus_t cudnnCreateReduceTensorDescriptor(cudnnReduceTensorStruct**) @nogc nothrow;
    cudnnStatus_t cudnnSetReduceTensorDescriptor(cudnnReduceTensorStruct*, cudnnReduceTensorOp_t, cudnnDataType_t, cudnnNanPropagation_t, cudnnReduceTensorIndices_t, cudnnIndicesType_t) @nogc nothrow;
    cudnnStatus_t cudnnGetReduceTensorDescriptor(const(cudnnReduceTensorStruct*), cudnnReduceTensorOp_t*, cudnnDataType_t*, cudnnNanPropagation_t*, cudnnReduceTensorIndices_t*, cudnnIndicesType_t*) @nogc nothrow;
    cudnnStatus_t cudnnDestroyReduceTensorDescriptor(cudnnReduceTensorStruct*) @nogc nothrow;
    cudnnStatus_t cudnnGetReductionIndicesSize(cudnnContext*, const(cudnnReduceTensorStruct*), const(cudnnTensorStruct*), const(cudnnTensorStruct*), c_ulong*) @nogc nothrow;
    cudnnStatus_t cudnnGetReductionWorkspaceSize(cudnnContext*, const(cudnnReduceTensorStruct*), const(cudnnTensorStruct*), const(cudnnTensorStruct*), c_ulong*) @nogc nothrow;
    cudnnStatus_t cudnnReduceTensor(cudnnContext*, const(cudnnReduceTensorStruct*), void*, c_ulong, void*, c_ulong, const(void)*, const(cudnnTensorStruct*), const(void)*, const(void)*, const(cudnnTensorStruct*), void*) @nogc nothrow;
    cudnnStatus_t cudnnSetTensor(cudnnContext*, const(cudnnTensorStruct*), void*, const(void)*) @nogc nothrow;
    cudnnStatus_t cudnnScaleTensor(cudnnContext*, const(cudnnTensorStruct*), void*, const(void)*) @nogc nothrow;
    alias cudnnConvolutionMode_t = _Anonymous_40;
    enum _Anonymous_40
    {
        CUDNN_CONVOLUTION = 0,
        CUDNN_CROSS_CORRELATION = 1,
    }
    enum CUDNN_CONVOLUTION = _Anonymous_40.CUDNN_CONVOLUTION;
    enum CUDNN_CROSS_CORRELATION = _Anonymous_40.CUDNN_CROSS_CORRELATION;
    cudnnStatus_t cudnnCreateFilterDescriptor(cudnnFilterStruct**) @nogc nothrow;
    cudnnStatus_t cudnnSetFilter4dDescriptor(cudnnFilterStruct*, cudnnDataType_t, cudnnTensorFormat_t, int, int, int, int) @nogc nothrow;
    cudnnStatus_t cudnnGetFilter4dDescriptor(const(cudnnFilterStruct*), cudnnDataType_t*, cudnnTensorFormat_t*, int*, int*, int*, int*) @nogc nothrow;
    cudnnStatus_t cudnnSetFilterNdDescriptor(cudnnFilterStruct*, cudnnDataType_t, cudnnTensorFormat_t, int, const(int)*) @nogc nothrow;
    cudnnStatus_t cudnnGetFilterNdDescriptor(const(cudnnFilterStruct*), int, cudnnDataType_t*, cudnnTensorFormat_t*, int*, int*) @nogc nothrow;
    cudnnStatus_t cudnnGetFilterSizeInBytes(const(cudnnFilterStruct*), c_ulong*) @nogc nothrow;
    cudnnStatus_t cudnnTransformFilter(cudnnContext*, const(cudnnTensorTransformStruct*), const(void)*, const(cudnnFilterStruct*), const(void)*, const(void)*, const(cudnnFilterStruct*), void*) @nogc nothrow;
    cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterStruct*) @nogc nothrow;
    cudnnStatus_t cudnnReorderFilterAndBias(cudnnContext*, const(cudnnFilterStruct*), cudnnReorderType_t, const(void)*, void*, int, const(void)*, void*) @nogc nothrow;
    cudnnStatus_t cudnnCreateConvolutionDescriptor(cudnnConvolutionStruct**) @nogc nothrow;
    cudnnStatus_t cudnnSetConvolutionMathType(cudnnConvolutionStruct*, cudnnMathType_t) @nogc nothrow;
    cudnnStatus_t cudnnGetConvolutionMathType(cudnnConvolutionStruct*, cudnnMathType_t*) @nogc nothrow;
    cudnnStatus_t cudnnSetConvolutionGroupCount(cudnnConvolutionStruct*, int) @nogc nothrow;
    cudnnStatus_t cudnnGetConvolutionGroupCount(cudnnConvolutionStruct*, int*) @nogc nothrow;
    cudnnStatus_t cudnnSetConvolutionReorderType(cudnnConvolutionStruct*, cudnnReorderType_t) @nogc nothrow;
    cudnnStatus_t cudnnGetConvolutionReorderType(cudnnConvolutionStruct*, cudnnReorderType_t*) @nogc nothrow;
    cudnnStatus_t cudnnSetConvolution2dDescriptor(cudnnConvolutionStruct*, int, int, int, int, int, int, cudnnConvolutionMode_t, cudnnDataType_t) @nogc nothrow;
    cudnnStatus_t cudnnGetConvolution2dDescriptor(const(cudnnConvolutionStruct*), int*, int*, int*, int*, int*, int*, cudnnConvolutionMode_t*, cudnnDataType_t*) @nogc nothrow;
    cudnnStatus_t cudnnGetConvolution2dForwardOutputDim(const(cudnnConvolutionStruct*), const(cudnnTensorStruct*), const(cudnnFilterStruct*), int*, int*, int*, int*) @nogc nothrow;
    cudnnStatus_t cudnnSetConvolutionNdDescriptor(cudnnConvolutionStruct*, int, const(int)*, const(int)*, const(int)*, cudnnConvolutionMode_t, cudnnDataType_t) @nogc nothrow;
    cudnnStatus_t cudnnGetConvolutionNdDescriptor(const(cudnnConvolutionStruct*), int, int*, int*, int*, int*, cudnnConvolutionMode_t*, cudnnDataType_t*) @nogc nothrow;
    cudnnStatus_t cudnnGetConvolutionNdForwardOutputDim(const(cudnnConvolutionStruct*), const(cudnnTensorStruct*), const(cudnnFilterStruct*), int, int*) @nogc nothrow;
    cudnnStatus_t cudnnDestroyConvolutionDescriptor(cudnnConvolutionStruct*) @nogc nothrow;
    alias cudnnConvolutionFwdPreference_t = _Anonymous_41;
    enum _Anonymous_41
    {
        CUDNN_CONVOLUTION_FWD_NO_WORKSPACE = 0,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST = 1,
        CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = 2,
    }
    enum CUDNN_CONVOLUTION_FWD_NO_WORKSPACE = _Anonymous_41.CUDNN_CONVOLUTION_FWD_NO_WORKSPACE;
    enum CUDNN_CONVOLUTION_FWD_PREFER_FASTEST = _Anonymous_41.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
    enum CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = _Anonymous_41.CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT;
    alias cudnnConvolutionFwdAlgo_t = _Anonymous_42;
    enum _Anonymous_42
    {
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
        CUDNN_CONVOLUTION_FWD_ALGO_GEMM = 2,
        CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = 3,
        CUDNN_CONVOLUTION_FWD_ALGO_FFT = 4,
        CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING = 5,
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD = 6,
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED = 7,
        CUDNN_CONVOLUTION_FWD_ALGO_COUNT = 8,
    }
    enum CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = _Anonymous_42.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    enum CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = _Anonymous_42.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    enum CUDNN_CONVOLUTION_FWD_ALGO_GEMM = _Anonymous_42.CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
    enum CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = _Anonymous_42.CUDNN_CONVOLUTION_FWD_ALGO_DIRECT;
    enum CUDNN_CONVOLUTION_FWD_ALGO_FFT = _Anonymous_42.CUDNN_CONVOLUTION_FWD_ALGO_FFT;
    enum CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING = _Anonymous_42.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING;
    enum CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD = _Anonymous_42.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
    enum CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED = _Anonymous_42.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
    enum CUDNN_CONVOLUTION_FWD_ALGO_COUNT = _Anonymous_42.CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    struct cudnnConvolutionFwdAlgoPerf_t
    {
        cudnnConvolutionFwdAlgo_t algo;
        cudnnStatus_t status;
        float time;
        c_ulong memory;
        cudnnDeterminism_t determinism;
        cudnnMathType_t mathType;
        int[3] reserved;
    }
    cudnnStatus_t cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnContext*, int*) @nogc nothrow;
    cudnnStatus_t cudnnFindConvolutionForwardAlgorithm(cudnnContext*, const(cudnnTensorStruct*), const(cudnnFilterStruct*), const(cudnnConvolutionStruct*), const(cudnnTensorStruct*), const(int), int*, cudnnConvolutionFwdAlgoPerf_t*) @nogc nothrow;
    cudnnStatus_t cudnnFindConvolutionForwardAlgorithmEx(cudnnContext*, const(cudnnTensorStruct*), const(void)*, const(cudnnFilterStruct*), const(void)*, const(cudnnConvolutionStruct*), const(cudnnTensorStruct*), void*, const(int), int*, cudnnConvolutionFwdAlgoPerf_t*, void*, c_ulong) @nogc nothrow;
    cudnnStatus_t cudnnGetConvolutionForwardAlgorithm(cudnnContext*, const(cudnnTensorStruct*), const(cudnnFilterStruct*), const(cudnnConvolutionStruct*), const(cudnnTensorStruct*), cudnnConvolutionFwdPreference_t, c_ulong, cudnnConvolutionFwdAlgo_t*) @nogc nothrow;
    cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7(cudnnContext*, const(cudnnTensorStruct*), const(cudnnFilterStruct*), const(cudnnConvolutionStruct*), const(cudnnTensorStruct*), const(int), int*, cudnnConvolutionFwdAlgoPerf_t*) @nogc nothrow;
    cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(cudnnContext*, const(cudnnTensorStruct*), const(cudnnFilterStruct*), const(cudnnConvolutionStruct*), const(cudnnTensorStruct*), cudnnConvolutionFwdAlgo_t, c_ulong*) @nogc nothrow;
    cudnnStatus_t cudnnConvolutionForward(cudnnContext*, const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnFilterStruct*), const(void)*, const(cudnnConvolutionStruct*), cudnnConvolutionFwdAlgo_t, void*, c_ulong, const(void)*, const(cudnnTensorStruct*), void*) @nogc nothrow;
    cudnnStatus_t cudnnConvolutionBiasActivationForward(cudnnContext*, const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnFilterStruct*), const(void)*, const(cudnnConvolutionStruct*), cudnnConvolutionFwdAlgo_t, void*, c_ulong, const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnActivationStruct*), const(cudnnTensorStruct*), void*) @nogc nothrow;
    cudnnStatus_t cudnnConvolutionBackwardBias(cudnnContext*, const(void)*, const(cudnnTensorStruct*), const(void)*, const(void)*, const(cudnnTensorStruct*), void*) @nogc nothrow;
    alias cudnnConvolutionBwdFilterPreference_t = _Anonymous_43;
    enum _Anonymous_43
    {
        CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE = 0,
        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST = 1,
        CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT = 2,
    }
    enum CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE = _Anonymous_43.CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE;
    enum CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST = _Anonymous_43.CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST;
    enum CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT = _Anonymous_43.CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT;
    alias cudnnConvolutionBwdFilterAlgo_t = _Anonymous_44;
    enum _Anonymous_44
    {
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = 0,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 = 1,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT = 2,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 = 3,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD = 4,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED = 5,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING = 6,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT = 7,
    }
    enum CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = _Anonymous_44.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
    enum CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 = _Anonymous_44.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
    enum CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT = _Anonymous_44.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT;
    enum CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 = _Anonymous_44.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3;
    enum CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD = _Anonymous_44.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD;
    enum CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED = _Anonymous_44.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED;
    enum CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING = _Anonymous_44.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING;
    enum CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT = _Anonymous_44.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT;
    struct cudnnConvolutionBwdFilterAlgoPerf_t
    {
        cudnnConvolutionBwdFilterAlgo_t algo;
        cudnnStatus_t status;
        float time;
        c_ulong memory;
        cudnnDeterminism_t determinism;
        cudnnMathType_t mathType;
        int[3] reserved;
    }
    cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnnContext*, int*) @nogc nothrow;
    cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithm(cudnnContext*, const(cudnnTensorStruct*), const(cudnnTensorStruct*), const(cudnnConvolutionStruct*), const(cudnnFilterStruct*), const(int), int*, cudnnConvolutionBwdFilterAlgoPerf_t*) @nogc nothrow;
    cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithmEx(cudnnContext*, const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnConvolutionStruct*), const(cudnnFilterStruct*), void*, const(int), int*, cudnnConvolutionBwdFilterAlgoPerf_t*, void*, c_ulong) @nogc nothrow;
    cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm(cudnnContext*, const(cudnnTensorStruct*), const(cudnnTensorStruct*), const(cudnnConvolutionStruct*), const(cudnnFilterStruct*), cudnnConvolutionBwdFilterPreference_t, c_ulong, cudnnConvolutionBwdFilterAlgo_t*) @nogc nothrow;
    cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudnnContext*, const(cudnnTensorStruct*), const(cudnnTensorStruct*), const(cudnnConvolutionStruct*), const(cudnnFilterStruct*), const(int), int*, cudnnConvolutionBwdFilterAlgoPerf_t*) @nogc nothrow;
    cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnContext*, const(cudnnTensorStruct*), const(cudnnTensorStruct*), const(cudnnConvolutionStruct*), const(cudnnFilterStruct*), cudnnConvolutionBwdFilterAlgo_t, c_ulong*) @nogc nothrow;
    cudnnStatus_t cudnnConvolutionBackwardFilter(cudnnContext*, const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnConvolutionStruct*), cudnnConvolutionBwdFilterAlgo_t, void*, c_ulong, const(void)*, const(cudnnFilterStruct*), void*) @nogc nothrow;
    alias cudnnConvolutionBwdDataPreference_t = _Anonymous_45;
    enum _Anonymous_45
    {
        CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE = 0,
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST = 1,
        CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT = 2,
    }
    enum CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE = _Anonymous_45.CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE;
    enum CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST = _Anonymous_45.CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST;
    enum CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT = _Anonymous_45.CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT;
    alias cudnnConvolutionBwdDataAlgo_t = _Anonymous_46;
    enum _Anonymous_46
    {
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = 0,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = 1,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT = 2,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = 3,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD = 4,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = 5,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT = 6,
    }
    enum CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = _Anonymous_46.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    enum CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = _Anonymous_46.CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    enum CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT = _Anonymous_46.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT;
    enum CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = _Anonymous_46.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING;
    enum CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD = _Anonymous_46.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD;
    enum CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = _Anonymous_46.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED;
    enum CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT = _Anonymous_46.CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
    struct cudnnConvolutionBwdDataAlgoPerf_t
    {
        cudnnConvolutionBwdDataAlgo_t algo;
        cudnnStatus_t status;
        float time;
        c_ulong memory;
        cudnnDeterminism_t determinism;
        cudnnMathType_t mathType;
        int[3] reserved;
    }
    cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnnContext*, int*) @nogc nothrow;
    cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithm(cudnnContext*, const(cudnnFilterStruct*), const(cudnnTensorStruct*), const(cudnnConvolutionStruct*), const(cudnnTensorStruct*), const(int), int*, cudnnConvolutionBwdDataAlgoPerf_t*) @nogc nothrow;
    cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithmEx(cudnnContext*, const(cudnnFilterStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnConvolutionStruct*), const(cudnnTensorStruct*), void*, const(int), int*, cudnnConvolutionBwdDataAlgoPerf_t*, void*, c_ulong) @nogc nothrow;
    cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm(cudnnContext*, const(cudnnFilterStruct*), const(cudnnTensorStruct*), const(cudnnConvolutionStruct*), const(cudnnTensorStruct*), cudnnConvolutionBwdDataPreference_t, c_ulong, cudnnConvolutionBwdDataAlgo_t*) @nogc nothrow;
    cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm_v7(cudnnContext*, const(cudnnFilterStruct*), const(cudnnTensorStruct*), const(cudnnConvolutionStruct*), const(cudnnTensorStruct*), const(int), int*, cudnnConvolutionBwdDataAlgoPerf_t*) @nogc nothrow;
    cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnContext*, const(cudnnFilterStruct*), const(cudnnTensorStruct*), const(cudnnConvolutionStruct*), const(cudnnTensorStruct*), cudnnConvolutionBwdDataAlgo_t, c_ulong*) @nogc nothrow;
    cudnnStatus_t cudnnConvolutionBackwardData(cudnnContext*, const(void)*, const(cudnnFilterStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnConvolutionStruct*), cudnnConvolutionBwdDataAlgo_t, void*, c_ulong, const(void)*, const(cudnnTensorStruct*), void*) @nogc nothrow;
    cudnnStatus_t cudnnIm2Col(cudnnContext*, const(cudnnTensorStruct*), const(void)*, const(cudnnFilterStruct*), const(cudnnConvolutionStruct*), void*) @nogc nothrow;
    alias cudnnSoftmaxAlgorithm_t = _Anonymous_47;
    enum _Anonymous_47
    {
        CUDNN_SOFTMAX_FAST = 0,
        CUDNN_SOFTMAX_ACCURATE = 1,
        CUDNN_SOFTMAX_LOG = 2,
    }
    enum CUDNN_SOFTMAX_FAST = _Anonymous_47.CUDNN_SOFTMAX_FAST;
    enum CUDNN_SOFTMAX_ACCURATE = _Anonymous_47.CUDNN_SOFTMAX_ACCURATE;
    enum CUDNN_SOFTMAX_LOG = _Anonymous_47.CUDNN_SOFTMAX_LOG;
    alias cudnnSoftmaxMode_t = _Anonymous_48;
    enum _Anonymous_48
    {
        CUDNN_SOFTMAX_MODE_INSTANCE = 0,
        CUDNN_SOFTMAX_MODE_CHANNEL = 1,
    }
    enum CUDNN_SOFTMAX_MODE_INSTANCE = _Anonymous_48.CUDNN_SOFTMAX_MODE_INSTANCE;
    enum CUDNN_SOFTMAX_MODE_CHANNEL = _Anonymous_48.CUDNN_SOFTMAX_MODE_CHANNEL;
    cudnnStatus_t cudnnSoftmaxForward(cudnnContext*, cudnnSoftmaxAlgorithm_t, cudnnSoftmaxMode_t, const(void)*, const(cudnnTensorStruct*), const(void)*, const(void)*, const(cudnnTensorStruct*), void*) @nogc nothrow;
    cudnnStatus_t cudnnSoftmaxBackward(cudnnContext*, cudnnSoftmaxAlgorithm_t, cudnnSoftmaxMode_t, const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(void)*, const(cudnnTensorStruct*), void*) @nogc nothrow;
    alias cudnnPoolingMode_t = _Anonymous_49;
    enum _Anonymous_49
    {
        CUDNN_POOLING_MAX = 0,
        CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1,
        CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2,
        CUDNN_POOLING_MAX_DETERMINISTIC = 3,
    }
    enum CUDNN_POOLING_MAX = _Anonymous_49.CUDNN_POOLING_MAX;
    enum CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = _Anonymous_49.CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    enum CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = _Anonymous_49.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
    enum CUDNN_POOLING_MAX_DETERMINISTIC = _Anonymous_49.CUDNN_POOLING_MAX_DETERMINISTIC;
    cudnnStatus_t cudnnCreatePoolingDescriptor(cudnnPoolingStruct**) @nogc nothrow;
    cudnnStatus_t cudnnSetPooling2dDescriptor(cudnnPoolingStruct*, cudnnPoolingMode_t, cudnnNanPropagation_t, int, int, int, int, int, int) @nogc nothrow;
    cudnnStatus_t cudnnGetPooling2dDescriptor(const(cudnnPoolingStruct*), cudnnPoolingMode_t*, cudnnNanPropagation_t*, int*, int*, int*, int*, int*, int*) @nogc nothrow;
    cudnnStatus_t cudnnSetPoolingNdDescriptor(cudnnPoolingStruct*, const(cudnnPoolingMode_t), const(cudnnNanPropagation_t), int, const(int)*, const(int)*, const(int)*) @nogc nothrow;
    cudnnStatus_t cudnnGetPoolingNdDescriptor(const(cudnnPoolingStruct*), int, cudnnPoolingMode_t*, cudnnNanPropagation_t*, int*, int*, int*, int*) @nogc nothrow;
    cudnnStatus_t cudnnGetPoolingNdForwardOutputDim(const(cudnnPoolingStruct*), const(cudnnTensorStruct*), int, int*) @nogc nothrow;
    cudnnStatus_t cudnnGetPooling2dForwardOutputDim(const(cudnnPoolingStruct*), const(cudnnTensorStruct*), int*, int*, int*, int*) @nogc nothrow;
    cudnnStatus_t cudnnDestroyPoolingDescriptor(cudnnPoolingStruct*) @nogc nothrow;
    cudnnStatus_t cudnnPoolingForward(cudnnContext*, const(cudnnPoolingStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(void)*, const(cudnnTensorStruct*), void*) @nogc nothrow;
    cudnnStatus_t cudnnPoolingBackward(cudnnContext*, const(cudnnPoolingStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(void)*, const(cudnnTensorStruct*), void*) @nogc nothrow;
    alias cudnnActivationMode_t = _Anonymous_50;
    enum _Anonymous_50
    {
        CUDNN_ACTIVATION_SIGMOID = 0,
        CUDNN_ACTIVATION_RELU = 1,
        CUDNN_ACTIVATION_TANH = 2,
        CUDNN_ACTIVATION_CLIPPED_RELU = 3,
        CUDNN_ACTIVATION_ELU = 4,
        CUDNN_ACTIVATION_IDENTITY = 5,
    }
    enum CUDNN_ACTIVATION_SIGMOID = _Anonymous_50.CUDNN_ACTIVATION_SIGMOID;
    enum CUDNN_ACTIVATION_RELU = _Anonymous_50.CUDNN_ACTIVATION_RELU;
    enum CUDNN_ACTIVATION_TANH = _Anonymous_50.CUDNN_ACTIVATION_TANH;
    enum CUDNN_ACTIVATION_CLIPPED_RELU = _Anonymous_50.CUDNN_ACTIVATION_CLIPPED_RELU;
    enum CUDNN_ACTIVATION_ELU = _Anonymous_50.CUDNN_ACTIVATION_ELU;
    enum CUDNN_ACTIVATION_IDENTITY = _Anonymous_50.CUDNN_ACTIVATION_IDENTITY;
    cudnnStatus_t cudnnCreateActivationDescriptor(cudnnActivationStruct**) @nogc nothrow;
    cudnnStatus_t cudnnSetActivationDescriptor(cudnnActivationStruct*, cudnnActivationMode_t, cudnnNanPropagation_t, double) @nogc nothrow;
    cudnnStatus_t cudnnGetActivationDescriptor(const(cudnnActivationStruct*), cudnnActivationMode_t*, cudnnNanPropagation_t*, double*) @nogc nothrow;
    cudnnStatus_t cudnnDestroyActivationDescriptor(cudnnActivationStruct*) @nogc nothrow;
    cudnnStatus_t cudnnActivationForward(cudnnContext*, cudnnActivationStruct*, const(void)*, const(cudnnTensorStruct*), const(void)*, const(void)*, const(cudnnTensorStruct*), void*) @nogc nothrow;
    cudnnStatus_t cudnnActivationBackward(cudnnContext*, cudnnActivationStruct*, const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(void)*, const(cudnnTensorStruct*), void*) @nogc nothrow;
    cudnnStatus_t cudnnCreateLRNDescriptor(cudnnLRNStruct**) @nogc nothrow;
    cudnnStatus_t cudnnGetRNNTrainingReserveSize(cudnnContext*, const(cudnnRNNStruct*), const(int), const(cudnnTensorStruct*)*, c_ulong*) @nogc nothrow;
    cudnnStatus_t cudnnGetRNNWorkspaceSize(cudnnContext*, const(cudnnRNNStruct*), const(int), const(cudnnTensorStruct*)*, c_ulong*) @nogc nothrow;
    alias cudnnLRNMode_t = _Anonymous_51;
    enum _Anonymous_51
    {
        CUDNN_LRN_CROSS_CHANNEL_DIM1 = 0,
    }
    enum CUDNN_LRN_CROSS_CHANNEL_DIM1 = _Anonymous_51.CUDNN_LRN_CROSS_CHANNEL_DIM1;
    cudnnStatus_t cudnnSetLRNDescriptor(cudnnLRNStruct*, uint, double, double, double) @nogc nothrow;
    cudnnStatus_t cudnnGetLRNDescriptor(cudnnLRNStruct*, uint*, double*, double*, double*) @nogc nothrow;
    cudnnStatus_t cudnnDestroyLRNDescriptor(cudnnLRNStruct*) @nogc nothrow;
    cudnnStatus_t cudnnLRNCrossChannelForward(cudnnContext*, cudnnLRNStruct*, cudnnLRNMode_t, const(void)*, const(cudnnTensorStruct*), const(void)*, const(void)*, const(cudnnTensorStruct*), void*) @nogc nothrow;
    cudnnStatus_t cudnnLRNCrossChannelBackward(cudnnContext*, cudnnLRNStruct*, cudnnLRNMode_t, const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(void)*, const(cudnnTensorStruct*), void*) @nogc nothrow;
    alias cudnnDivNormMode_t = _Anonymous_52;
    enum _Anonymous_52
    {
        CUDNN_DIVNORM_PRECOMPUTED_MEANS = 0,
    }
    enum CUDNN_DIVNORM_PRECOMPUTED_MEANS = _Anonymous_52.CUDNN_DIVNORM_PRECOMPUTED_MEANS;
    cudnnStatus_t cudnnDivisiveNormalizationForward(cudnnContext*, cudnnLRNStruct*, cudnnDivNormMode_t, const(void)*, const(cudnnTensorStruct*), const(void)*, const(void)*, void*, void*, const(void)*, const(cudnnTensorStruct*), void*) @nogc nothrow;
    cudnnStatus_t cudnnDivisiveNormalizationBackward(cudnnContext*, cudnnLRNStruct*, cudnnDivNormMode_t, const(void)*, const(cudnnTensorStruct*), const(void)*, const(void)*, const(void)*, void*, void*, const(void)*, const(cudnnTensorStruct*), void*, void*) @nogc nothrow;
    alias cudnnBatchNormMode_t = _Anonymous_53;
    enum _Anonymous_53
    {
        CUDNN_BATCHNORM_PER_ACTIVATION = 0,
        CUDNN_BATCHNORM_SPATIAL = 1,
        CUDNN_BATCHNORM_SPATIAL_PERSISTENT = 2,
    }
    enum CUDNN_BATCHNORM_PER_ACTIVATION = _Anonymous_53.CUDNN_BATCHNORM_PER_ACTIVATION;
    enum CUDNN_BATCHNORM_SPATIAL = _Anonymous_53.CUDNN_BATCHNORM_SPATIAL;
    enum CUDNN_BATCHNORM_SPATIAL_PERSISTENT = _Anonymous_53.CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
    cudnnStatus_t cudnnDeriveBNTensorDescriptor(cudnnTensorStruct*, const(cudnnTensorStruct*), cudnnBatchNormMode_t) @nogc nothrow;
    alias cudnnBatchNormOps_t = _Anonymous_54;
    enum _Anonymous_54
    {
        CUDNN_BATCHNORM_OPS_BN = 0,
        CUDNN_BATCHNORM_OPS_BN_ACTIVATION = 1,
        CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION = 2,
    }
    enum CUDNN_BATCHNORM_OPS_BN = _Anonymous_54.CUDNN_BATCHNORM_OPS_BN;
    enum CUDNN_BATCHNORM_OPS_BN_ACTIVATION = _Anonymous_54.CUDNN_BATCHNORM_OPS_BN_ACTIVATION;
    enum CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION = _Anonymous_54.CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION;
    cudnnStatus_t cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(cudnnContext*, cudnnBatchNormMode_t, cudnnBatchNormOps_t, const(cudnnTensorStruct*), const(cudnnTensorStruct*), const(cudnnTensorStruct*), const(cudnnTensorStruct*), const(cudnnActivationStruct*), c_ulong*) @nogc nothrow;
    cudnnStatus_t cudnnGetBatchNormalizationBackwardExWorkspaceSize(cudnnContext*, cudnnBatchNormMode_t, cudnnBatchNormOps_t, const(cudnnTensorStruct*), const(cudnnTensorStruct*), const(cudnnTensorStruct*), const(cudnnTensorStruct*), const(cudnnTensorStruct*), const(cudnnTensorStruct*), const(cudnnActivationStruct*), c_ulong*) @nogc nothrow;
    cudnnStatus_t cudnnGetBatchNormalizationTrainingExReserveSpaceSize(cudnnContext*, cudnnBatchNormMode_t, cudnnBatchNormOps_t, const(cudnnActivationStruct*), const(cudnnTensorStruct*), c_ulong*) @nogc nothrow;
    cudnnStatus_t cudnnBatchNormalizationForwardTraining(cudnnContext*, cudnnBatchNormMode_t, const(void)*, const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*), void*, const(cudnnTensorStruct*), const(void)*, const(void)*, double, void*, void*, double, void*, void*) @nogc nothrow;
    cudnnStatus_t cudnnBatchNormalizationForwardTrainingEx(cudnnContext*, cudnnBatchNormMode_t, cudnnBatchNormOps_t, const(void)*, const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*), void*, const(cudnnTensorStruct*), const(void)*, const(void)*, double, void*, void*, double, void*, void*, cudnnActivationStruct*, void*, c_ulong, void*, c_ulong) @nogc nothrow;
    cudnnStatus_t cudnnBatchNormalizationForwardInference(cudnnContext*, cudnnBatchNormMode_t, const(void)*, const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*), void*, const(cudnnTensorStruct*), const(void)*, const(void)*, const(void)*, const(void)*, double) @nogc nothrow;
    cudnnStatus_t cudnnBatchNormalizationBackward(cudnnContext*, cudnnBatchNormMode_t, const(void)*, const(void)*, const(void)*, const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*), void*, const(cudnnTensorStruct*), const(void)*, void*, void*, double, const(void)*, const(void)*) @nogc nothrow;
    cudnnStatus_t cudnnBatchNormalizationBackwardEx(cudnnContext*, cudnnBatchNormMode_t, cudnnBatchNormOps_t, const(void)*, const(void)*, const(void)*, const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*), void*, const(cudnnTensorStruct*), void*, const(cudnnTensorStruct*), const(void)*, const(void)*, void*, void*, double, const(void)*, const(void)*, cudnnActivationStruct*, void*, c_ulong, void*, c_ulong) @nogc nothrow;
    alias cudnnSamplerType_t = _Anonymous_55;
    enum _Anonymous_55
    {
        CUDNN_SAMPLER_BILINEAR = 0,
    }
    enum CUDNN_SAMPLER_BILINEAR = _Anonymous_55.CUDNN_SAMPLER_BILINEAR;
    cudnnStatus_t cudnnCreateSpatialTransformerDescriptor(cudnnSpatialTransformerStruct**) @nogc nothrow;
    cudnnStatus_t cudnnSetSpatialTransformerNdDescriptor(cudnnSpatialTransformerStruct*, cudnnSamplerType_t, cudnnDataType_t, const(int), const(int)*) @nogc nothrow;
    cudnnStatus_t cudnnDestroySpatialTransformerDescriptor(cudnnSpatialTransformerStruct*) @nogc nothrow;
    cudnnStatus_t cudnnSpatialTfGridGeneratorForward(cudnnContext*, const(cudnnSpatialTransformerStruct*), const(void)*, void*) @nogc nothrow;
    cudnnStatus_t cudnnSpatialTfGridGeneratorBackward(cudnnContext*, const(cudnnSpatialTransformerStruct*), const(void)*, void*) @nogc nothrow;
    cudnnStatus_t cudnnSpatialTfSamplerForward(cudnnContext*, cudnnSpatialTransformerStruct*, const(void)*, const(cudnnTensorStruct*), const(void)*, const(void)*, const(void)*, cudnnTensorStruct*, void*) @nogc nothrow;
    cudnnStatus_t cudnnSpatialTfSamplerBackward(cudnnContext*, cudnnSpatialTransformerStruct*, const(void)*, const(cudnnTensorStruct*), const(void)*, const(void)*, const(cudnnTensorStruct*), void*, const(void)*, const(cudnnTensorStruct*), const(void)*, const(void)*, const(void)*, void*) @nogc nothrow;
    alias cudnnDropoutDescriptor_t = cudnnDropoutStruct*;
    struct cudnnDropoutStruct;
    cudnnStatus_t cudnnCreateDropoutDescriptor(cudnnDropoutStruct**) @nogc nothrow;
    cudnnStatus_t cudnnDestroyDropoutDescriptor(cudnnDropoutStruct*) @nogc nothrow;
    cudnnStatus_t cudnnDropoutGetStatesSize(cudnnContext*, c_ulong*) @nogc nothrow;
    cudnnStatus_t cudnnDropoutGetReserveSpaceSize(cudnnTensorStruct*, c_ulong*) @nogc nothrow;
    cudnnStatus_t cudnnSetDropoutDescriptor(cudnnDropoutStruct*, cudnnContext*, float, void*, c_ulong, ulong) @nogc nothrow;
    cudnnStatus_t cudnnRestoreDropoutDescriptor(cudnnDropoutStruct*, cudnnContext*, float, void*, c_ulong, ulong) @nogc nothrow;
    cudnnStatus_t cudnnGetDropoutDescriptor(cudnnDropoutStruct*, cudnnContext*, float*, void**, ulong*) @nogc nothrow;
    cudnnStatus_t cudnnDropoutForward(cudnnContext*, const(cudnnDropoutStruct*), const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*), void*, void*, c_ulong) @nogc nothrow;
    cudnnStatus_t cudnnDropoutBackward(cudnnContext*, const(cudnnDropoutStruct*), const(cudnnTensorStruct*), const(void)*, const(cudnnTensorStruct*), void*, void*, c_ulong) @nogc nothrow;
    alias cudnnRNNAlgo_t = _Anonymous_56;
    enum _Anonymous_56
    {
        CUDNN_RNN_ALGO_STANDARD = 0,
        CUDNN_RNN_ALGO_PERSIST_STATIC = 1,
        CUDNN_RNN_ALGO_PERSIST_DYNAMIC = 2,
        CUDNN_RNN_ALGO_COUNT = 3,
    }
    enum CUDNN_RNN_ALGO_STANDARD = _Anonymous_56.CUDNN_RNN_ALGO_STANDARD;
    enum CUDNN_RNN_ALGO_PERSIST_STATIC = _Anonymous_56.CUDNN_RNN_ALGO_PERSIST_STATIC;
    enum CUDNN_RNN_ALGO_PERSIST_DYNAMIC = _Anonymous_56.CUDNN_RNN_ALGO_PERSIST_DYNAMIC;
    enum CUDNN_RNN_ALGO_COUNT = _Anonymous_56.CUDNN_RNN_ALGO_COUNT;
    alias cudnnRNNMode_t = _Anonymous_57;
    enum _Anonymous_57
    {
        CUDNN_RNN_RELU = 0,
        CUDNN_RNN_TANH = 1,
        CUDNN_LSTM = 2,
        CUDNN_GRU = 3,
    }
    enum CUDNN_RNN_RELU = _Anonymous_57.CUDNN_RNN_RELU;
    enum CUDNN_RNN_TANH = _Anonymous_57.CUDNN_RNN_TANH;
    enum CUDNN_LSTM = _Anonymous_57.CUDNN_LSTM;
    enum CUDNN_GRU = _Anonymous_57.CUDNN_GRU;
    alias cudnnRNNBiasMode_t = _Anonymous_58;
    enum _Anonymous_58
    {
        CUDNN_RNN_NO_BIAS = 0,
        CUDNN_RNN_SINGLE_INP_BIAS = 1,
        CUDNN_RNN_DOUBLE_BIAS = 2,
        CUDNN_RNN_SINGLE_REC_BIAS = 3,
    }
    enum CUDNN_RNN_NO_BIAS = _Anonymous_58.CUDNN_RNN_NO_BIAS;
    enum CUDNN_RNN_SINGLE_INP_BIAS = _Anonymous_58.CUDNN_RNN_SINGLE_INP_BIAS;
    enum CUDNN_RNN_DOUBLE_BIAS = _Anonymous_58.CUDNN_RNN_DOUBLE_BIAS;
    enum CUDNN_RNN_SINGLE_REC_BIAS = _Anonymous_58.CUDNN_RNN_SINGLE_REC_BIAS;
    alias cudnnDirectionMode_t = _Anonymous_59;
    enum _Anonymous_59
    {
        CUDNN_UNIDIRECTIONAL = 0,
        CUDNN_BIDIRECTIONAL = 1,
    }
    enum CUDNN_UNIDIRECTIONAL = _Anonymous_59.CUDNN_UNIDIRECTIONAL;
    enum CUDNN_BIDIRECTIONAL = _Anonymous_59.CUDNN_BIDIRECTIONAL;
    alias cudnnRNNInputMode_t = _Anonymous_60;
    enum _Anonymous_60
    {
        CUDNN_LINEAR_INPUT = 0,
        CUDNN_SKIP_INPUT = 1,
    }
    enum CUDNN_LINEAR_INPUT = _Anonymous_60.CUDNN_LINEAR_INPUT;
    enum CUDNN_SKIP_INPUT = _Anonymous_60.CUDNN_SKIP_INPUT;
    alias cudnnRNNClipMode_t = _Anonymous_61;
    enum _Anonymous_61
    {
        CUDNN_RNN_CLIP_NONE = 0,
        CUDNN_RNN_CLIP_MINMAX = 1,
    }
    enum CUDNN_RNN_CLIP_NONE = _Anonymous_61.CUDNN_RNN_CLIP_NONE;
    enum CUDNN_RNN_CLIP_MINMAX = _Anonymous_61.CUDNN_RNN_CLIP_MINMAX;
    alias cudnnRNNDataLayout_t = _Anonymous_62;
    enum _Anonymous_62
    {
        CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED = 0,
        CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED = 1,
        CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED = 2,
    }
    enum CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED = _Anonymous_62.CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED;
    enum CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED = _Anonymous_62.CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED;
    enum CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED = _Anonymous_62.CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED;
    alias cudnnRNNPaddingMode_t = _Anonymous_63;
    enum _Anonymous_63
    {
        CUDNN_RNN_PADDED_IO_DISABLED = 0,
        CUDNN_RNN_PADDED_IO_ENABLED = 1,
    }
    enum CUDNN_RNN_PADDED_IO_DISABLED = _Anonymous_63.CUDNN_RNN_PADDED_IO_DISABLED;
    enum CUDNN_RNN_PADDED_IO_ENABLED = _Anonymous_63.CUDNN_RNN_PADDED_IO_ENABLED;
    struct cudnnRNNStruct;
    alias cudnnRNNDescriptor_t = cudnnRNNStruct*;
    struct cudnnPersistentRNNPlan;
    alias cudnnPersistentRNNPlan_t = cudnnPersistentRNNPlan*;
    struct cudnnRNNDataStruct;
    alias cudnnRNNDataDescriptor_t = cudnnRNNDataStruct*;
    cudnnStatus_t cudnnCreateRNNDescriptor(cudnnRNNStruct**) @nogc nothrow;
    cudnnStatus_t cudnnDestroyRNNDescriptor(cudnnRNNStruct*) @nogc nothrow;
    cudnnStatus_t cudnnSetRNNDescriptor(cudnnContext*, cudnnRNNStruct*, const(int), const(int), cudnnDropoutStruct*, cudnnRNNInputMode_t, cudnnDirectionMode_t, cudnnRNNMode_t, cudnnRNNAlgo_t, cudnnDataType_t) @nogc nothrow;
    cudnnStatus_t cudnnGetRNNDescriptor(cudnnContext*, cudnnRNNStruct*, int*, int*, cudnnDropoutStruct**, cudnnRNNInputMode_t*, cudnnDirectionMode_t*, cudnnRNNMode_t*, cudnnRNNAlgo_t*, cudnnDataType_t*) @nogc nothrow;
    cudnnStatus_t cudnnSetRNNMatrixMathType(cudnnRNNStruct*, cudnnMathType_t) @nogc nothrow;
    cudnnStatus_t cudnnGetRNNMatrixMathType(cudnnRNNStruct*, cudnnMathType_t*) @nogc nothrow;
    cudnnStatus_t cudnnSetRNNBiasMode(cudnnRNNStruct*, cudnnRNNBiasMode_t) @nogc nothrow;
    cudnnStatus_t cudnnGetRNNBiasMode(cudnnRNNStruct*, cudnnRNNBiasMode_t*) @nogc nothrow;
    cudnnStatus_t cudnnRNNSetClip(cudnnContext*, cudnnRNNStruct*, cudnnRNNClipMode_t, cudnnNanPropagation_t, double, double) @nogc nothrow;
    cudnnStatus_t cudnnRNNGetClip(cudnnContext*, cudnnRNNStruct*, cudnnRNNClipMode_t*, cudnnNanPropagation_t*, double*, double*) @nogc nothrow;
    cudnnStatus_t cudnnSetRNNProjectionLayers(cudnnContext*, cudnnRNNStruct*, const(int), const(int)) @nogc nothrow;
    cudnnStatus_t cudnnGetRNNProjectionLayers(cudnnContext*, const(cudnnRNNStruct*), int*, int*) @nogc nothrow;
    cudnnStatus_t cudnnCreatePersistentRNNPlan(cudnnRNNStruct*, const(int), const(cudnnDataType_t), cudnnPersistentRNNPlan**) @nogc nothrow;
    cudnnStatus_t cudnnDestroyPersistentRNNPlan(cudnnPersistentRNNPlan*) @nogc nothrow;
    cudnnStatus_t cudnnSetPersistentRNNPlan(cudnnRNNStruct*, cudnnPersistentRNNPlan*) @nogc nothrow;



    static if(!is(typeof(CUDNN_BN_MIN_EPSILON))) {
        private enum enumMixinStr_CUDNN_BN_MIN_EPSILON = `enum CUDNN_BN_MIN_EPSILON = 0.0;`;
        static if(is(typeof({ mixin(enumMixinStr_CUDNN_BN_MIN_EPSILON); }))) {
            mixin(enumMixinStr_CUDNN_BN_MIN_EPSILON);
        }
    }




    static if(!is(typeof(CUDNN_LRN_MIN_BETA))) {
        private enum enumMixinStr_CUDNN_LRN_MIN_BETA = `enum CUDNN_LRN_MIN_BETA = 0.01;`;
        static if(is(typeof({ mixin(enumMixinStr_CUDNN_LRN_MIN_BETA); }))) {
            mixin(enumMixinStr_CUDNN_LRN_MIN_BETA);
        }
    }




    static if(!is(typeof(CUDNN_LRN_MIN_K))) {
        private enum enumMixinStr_CUDNN_LRN_MIN_K = `enum CUDNN_LRN_MIN_K = 1e-5;`;
        static if(is(typeof({ mixin(enumMixinStr_CUDNN_LRN_MIN_K); }))) {
            mixin(enumMixinStr_CUDNN_LRN_MIN_K);
        }
    }




    static if(!is(typeof(CUDNN_LRN_MAX_N))) {
        private enum enumMixinStr_CUDNN_LRN_MAX_N = `enum CUDNN_LRN_MAX_N = 16;`;
        static if(is(typeof({ mixin(enumMixinStr_CUDNN_LRN_MAX_N); }))) {
            mixin(enumMixinStr_CUDNN_LRN_MAX_N);
        }
    }




    static if(!is(typeof(CUDNN_LRN_MIN_N))) {
        private enum enumMixinStr_CUDNN_LRN_MIN_N = `enum CUDNN_LRN_MIN_N = 1;`;
        static if(is(typeof({ mixin(enumMixinStr_CUDNN_LRN_MIN_N); }))) {
            mixin(enumMixinStr_CUDNN_LRN_MIN_N);
        }
    }




    static if(!is(typeof(CUDNN_DIM_MAX))) {
        private enum enumMixinStr_CUDNN_DIM_MAX = `enum CUDNN_DIM_MAX = 8;`;
        static if(is(typeof({ mixin(enumMixinStr_CUDNN_DIM_MAX); }))) {
            mixin(enumMixinStr_CUDNN_DIM_MAX);
        }
    }






    static if(!is(typeof(CUDNN_VERSION))) {
        private enum enumMixinStr_CUDNN_VERSION = `enum CUDNN_VERSION = ( CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL );`;
        static if(is(typeof({ mixin(enumMixinStr_CUDNN_VERSION); }))) {
            mixin(enumMixinStr_CUDNN_VERSION);
        }
    }




    static if(!is(typeof(CUDNN_PATCHLEVEL))) {
        private enum enumMixinStr_CUDNN_PATCHLEVEL = `enum CUDNN_PATCHLEVEL = 5;`;
        static if(is(typeof({ mixin(enumMixinStr_CUDNN_PATCHLEVEL); }))) {
            mixin(enumMixinStr_CUDNN_PATCHLEVEL);
        }
    }




    static if(!is(typeof(CUDNN_MINOR))) {
        private enum enumMixinStr_CUDNN_MINOR = `enum CUDNN_MINOR = 6;`;
        static if(is(typeof({ mixin(enumMixinStr_CUDNN_MINOR); }))) {
            mixin(enumMixinStr_CUDNN_MINOR);
        }
    }




    static if(!is(typeof(CUDNN_MAJOR))) {
        private enum enumMixinStr_CUDNN_MAJOR = `enum CUDNN_MAJOR = 7;`;
        static if(is(typeof({ mixin(enumMixinStr_CUDNN_MAJOR); }))) {
            mixin(enumMixinStr_CUDNN_MAJOR);
        }
    }






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
    static if(!is(typeof(CUDNN_SEQDATA_DIM_COUNT))) {
        private enum enumMixinStr_CUDNN_SEQDATA_DIM_COUNT = `enum CUDNN_SEQDATA_DIM_COUNT = 4;`;
        static if(is(typeof({ mixin(enumMixinStr_CUDNN_SEQDATA_DIM_COUNT); }))) {
            mixin(enumMixinStr_CUDNN_SEQDATA_DIM_COUNT);
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




    static if(!is(typeof(CUDNN_ATTN_QUERYMAP_ALL_TO_ONE))) {
        private enum enumMixinStr_CUDNN_ATTN_QUERYMAP_ALL_TO_ONE = `enum CUDNN_ATTN_QUERYMAP_ALL_TO_ONE = 0;`;
        static if(is(typeof({ mixin(enumMixinStr_CUDNN_ATTN_QUERYMAP_ALL_TO_ONE); }))) {
            mixin(enumMixinStr_CUDNN_ATTN_QUERYMAP_ALL_TO_ONE);
        }
    }




    static if(!is(typeof(CUDNN_ATTN_QUERYMAP_ONE_TO_ONE))) {
        private enum enumMixinStr_CUDNN_ATTN_QUERYMAP_ONE_TO_ONE = `enum CUDNN_ATTN_QUERYMAP_ONE_TO_ONE = ( 1U << 0 );`;
        static if(is(typeof({ mixin(enumMixinStr_CUDNN_ATTN_QUERYMAP_ONE_TO_ONE); }))) {
            mixin(enumMixinStr_CUDNN_ATTN_QUERYMAP_ONE_TO_ONE);
        }
    }




    static if(!is(typeof(CUDNN_ATTN_DISABLE_PROJ_BIASES))) {
        private enum enumMixinStr_CUDNN_ATTN_DISABLE_PROJ_BIASES = `enum CUDNN_ATTN_DISABLE_PROJ_BIASES = 0;`;
        static if(is(typeof({ mixin(enumMixinStr_CUDNN_ATTN_DISABLE_PROJ_BIASES); }))) {
            mixin(enumMixinStr_CUDNN_ATTN_DISABLE_PROJ_BIASES);
        }
    }




    static if(!is(typeof(CUDNN_ATTN_ENABLE_PROJ_BIASES))) {
        private enum enumMixinStr_CUDNN_ATTN_ENABLE_PROJ_BIASES = `enum CUDNN_ATTN_ENABLE_PROJ_BIASES = ( 1U << 1 );`;
        static if(is(typeof({ mixin(enumMixinStr_CUDNN_ATTN_ENABLE_PROJ_BIASES); }))) {
            mixin(enumMixinStr_CUDNN_ATTN_ENABLE_PROJ_BIASES);
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






    static if(!is(typeof(CUDNN_ATTN_WKIND_COUNT))) {
        private enum enumMixinStr_CUDNN_ATTN_WKIND_COUNT = `enum CUDNN_ATTN_WKIND_COUNT = 8;`;
        static if(is(typeof({ mixin(enumMixinStr_CUDNN_ATTN_WKIND_COUNT); }))) {
            mixin(enumMixinStr_CUDNN_ATTN_WKIND_COUNT);
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




    static if(!is(typeof(CUDNN_SEV_ERROR_EN))) {
        private enum enumMixinStr_CUDNN_SEV_ERROR_EN = `enum CUDNN_SEV_ERROR_EN = ( 1U << CUDNN_SEV_ERROR );`;
        static if(is(typeof({ mixin(enumMixinStr_CUDNN_SEV_ERROR_EN); }))) {
            mixin(enumMixinStr_CUDNN_SEV_ERROR_EN);
        }
    }




    static if(!is(typeof(CUDNN_SEV_WARNING_EN))) {
        private enum enumMixinStr_CUDNN_SEV_WARNING_EN = `enum CUDNN_SEV_WARNING_EN = ( 1U << CUDNN_SEV_WARNING );`;
        static if(is(typeof({ mixin(enumMixinStr_CUDNN_SEV_WARNING_EN); }))) {
            mixin(enumMixinStr_CUDNN_SEV_WARNING_EN);
        }
    }




    static if(!is(typeof(CUDNN_SEV_INFO_EN))) {
        private enum enumMixinStr_CUDNN_SEV_INFO_EN = `enum CUDNN_SEV_INFO_EN = ( 1U << CUDNN_SEV_INFO );`;
        static if(is(typeof({ mixin(enumMixinStr_CUDNN_SEV_INFO_EN); }))) {
            mixin(enumMixinStr_CUDNN_SEV_INFO_EN);
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
    static if(!is(typeof(__VECTOR_FUNCTIONS_DECL__))) {
        private enum enumMixinStr___VECTOR_FUNCTIONS_DECL__ = `enum __VECTOR_FUNCTIONS_DECL__ = static __inline__ ;`;
        static if(is(typeof({ mixin(enumMixinStr___VECTOR_FUNCTIONS_DECL__); }))) {
            mixin(enumMixinStr___VECTOR_FUNCTIONS_DECL__);
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
    static if(!is(typeof(__cuda_builtin_vector_align8))) {
        private enum enumMixinStr___cuda_builtin_vector_align8 = `enum __cuda_builtin_vector_align8 = ( tag , members ) __attribute__ ( ( aligned ( 8 ) ) ) tag
 { members
 };`;
        static if(is(typeof({ mixin(enumMixinStr___cuda_builtin_vector_align8); }))) {
            mixin(enumMixinStr___cuda_builtin_vector_align8);
        }
    }



}
