/**
   header translated from https://github.com/pytorch/pytorch/blob/v1.0.0/aten/src/ATen/dlpack.h
*/
module grain.dlpack.header;

@safe @nogc nothrow extern (C):

//// The current version of dlpack
enum DLPACK_VERSION = 10;

/// The device type in DLContext.
alias DLDeviceType = int;
enum : DLDeviceType
{
    kDLCPU = 1,
    kDLGPU = 2,
    // kDLCPUPinned = kDLCPU | kDLGPU
    kDLCPUPinned = 3,
    kDLOpenCL = 4,
    kDLMetal = 8,
    kDLVPI = 9,
    kDLROCM = 10,
}

/// A Device context for Tensor and operator.
struct DLContext
{
    /// The device type used in the device.
    DLDeviceType device_type;
    /// The device index
    int device_id;
}

/// The type code options DLDataType.
alias DLDataTypeCode = uint;
enum : DLDataTypeCode
{
    kDLInt = 0U,
    kDLUInt = 1U,
    kDLFloat = 2U,
}

/** The data type the tensor can hold.
 *
 *  Examples
 *   - float: type_code = 2, bits = 32, lanes=1
 *   - float4(vectorized 4 float): type_code = 2, bits = 32, lanes=4
 *   - int8: type_code = 0, bits = 8, lanes=1
 */
struct DLDataType
{
    /** Type code of base types.
     *  We keep it uint8_t instead of DLDataTypeCode for minimal memory
     *  footprint, but the value should be one of DLDataTypeCode enum values.
     * */
    ubyte code;
    /// Number of bits, common choices are 8, 16, 32.
    ubyte bits;
    //// Number of lanes in the type, used for vector types.
    ushort lanes;
}

/// Plain C Tensor object, does not manage memory.
struct DLTensor
{
    /** The opaque data pointer points to the allocated data.
     *  This will be CUDA device pointer or cl_mem handle in OpenCL.
     *  This pointer is always aligns to 256 bytes as in CUDA.
     */
    void* data;
    /** The device context of the tensor */
    DLContext ctx;
    /** Number of dimensions */
    int ndim;
    /** The data type of the pointer*/
    DLDataType dtype;
    /** The shape of the tensor */
    long* shape;
    /** strides of the tensor,
     *  can be NULL, indicating tensor is compact.
     */
    long* strides;
    /** The offset in bytes to the beginning pointer to data */
    ulong byte_offset;
}

/** C Tensor object, manage memory of DLTensor. This data structure is
 *  intended to faciliate the borrowing of DLTensor by another framework. It is
 *  not meant to transfer the tensor. When the borrowing framework doesn't need
 *  the tensor, it should call the deleter to notify the host that the resource
 *  is no longer needed.
 */
struct DLManagedTensor
{
    /** DLTensor which is being memory managed */
    DLTensor dl_tensor;
    /** the context of the original host framework of DLManagedTensor in
     *  which DLManagedTensor is used in the framework. It can also be NULL.
     */
    void* manager_ctx;
    /** Destructor signature void (*)(void*) - this should be called
     *  to destruct manager_ctx which holds the DLManagedTensor. It can be NULL
     *  if there is no way for the caller to provide a reasonable destructor.
     */
    nothrow @nogc void function(DLManagedTensor*) deleter;
}
