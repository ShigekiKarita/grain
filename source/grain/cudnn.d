/**
   cuDNN high level wrapper for grain.autograd.Variable
 */
module grain.cudnn;

version (grain_cuda):

public import grain.cuda : cudnnHandle, checkCUDNN, CuPtr;
import grain.autograd; //  : Variable, DeviceStorage;
public import derelict.cuda;
public import derelict.cudnn7;

// TODO make shared
__gshared bool deterministic = false;
__gshared bool nanProp = true;

auto isDeterministic() {
    return deterministic ? CUDNN_DETERMINISTIC : CUDNN_NON_DETERMINISTIC;
}

auto isNanProp() {
    return nanProp ? CUDNN_PROPAGATE_NAN : CUDNN_NOT_PROPAGATE_NAN;
}


/// convert floating point types (float, double) into cudnn enum
auto cudnnDataType(T)() {
    // TODO support half
    static if(is(T == float)) return CUDNN_DATA_FLOAT;
    else static if(is(T == double)) return CUDNN_DATA_DOUBLE;
    else static assert(false, "unsupported type");
}

///
struct TensorDesc {
    cudnnTensorDescriptor_t desc;
    CUdeviceptr ptr;
    alias desc this;

    @disable this(this); // no copy
    @disable new(size_t); // no allocation on heap

    ~this() {
        checkCUDNN( cudnnDestroyTensorDescriptor(desc) );
    }
}

/// convert variable to cudnn tensor discriptor object
auto makeCudnnTensor(T, size_t dim)(Variable!(T, dim, DeviceStorage) x) {
    static assert(dim < CUDNN_DIM_MAX);
    static if (dim < 4) {
        enum int ddim = 4;
        int[ddim] shape;
        int[ddim] strides;
        shape[] = 1;
        strides[] = 1;
        foreach (d; 0 .. dim) {
            assert(x.shape[d] < int.max);
            shape[d] = cast(int) x.shape[d];
        }
        // shape[0..dim] = x.shape;
        strides[0..dim] = x.strides;
    } else {
        enum int ddim = cast(int) dim;
        auto shape = x.shape;
        auto strides = x.strides;
    }

    TensorDesc tdesc;
    tdesc.ptr = x.data.ptr;
    checkCUDNN(cudnnCreateTensorDescriptor(&tdesc.desc));
    checkCUDNN(cudnnSetTensorNdDescriptor(tdesc.desc,
                                          cudnnDataType!T,
                                          ddim,
                                          shape.ptr,
                                          strides.ptr));
    return tdesc;
}

/// y = alpha * f(x) + beta * y
void activationForward(cudnnActivationMode_t A, T, size_t dim)(
    Variable!(T, dim, DeviceStorage) x, Variable!(T, dim, DeviceStorage) y,
    T alpha=1.0, T beta=0.0, double coeff=0.0) {
    static assert(dim <= 5, "cuDNN only supports <= 5 dim tensors. and pack dim is not supported yet.");
    // init descriptors
    cudnnActivationDescriptor_t  activDesc;
    checkCUDNN( cudnnCreateActivationDescriptor(&activDesc) );
    scope(exit) checkCUDNN( cudnnCreateActivationDescriptor(&activDesc) );
    checkCUDNN( cudnnSetActivationDescriptor(activDesc,
                                             A, // CUDNN_ACTIVATION_RELU,
                                             CUDNN_PROPAGATE_NAN,
                                             coeff) );
    auto tx = x.makeCudnnTensor;
    auto ty = y.makeCudnnTensor;
    checkCUDNN( cudnnActivationForward(cudnnHandle,
                                       activDesc,
                                       &alpha,
                                       tx,
                                       cast(void*) tx.ptr,
                                       &beta,
                                       ty,
                                       cast(void*) ty.ptr) );
}

///
void activationBackward(cudnnActivationMode_t A, T, size_t dim)(
    Variable!(T, dim, DeviceStorage) gx, Variable!(T, dim, DeviceStorage) gy,
    Variable!(T, dim, DeviceStorage) x, Variable!(T, dim, DeviceStorage) y,
    T alpha=1.0, T beta=0.0, double coeff=0.0) {
    static assert(dim <= 5, "cuDNN only supports <= 5 dim tensors. and pack dim is not supported yet.");
    // init descriptors
    cudnnActivationDescriptor_t  activDesc;
    checkCUDNN( cudnnCreateActivationDescriptor(&activDesc) );
    scope(exit) checkCUDNN( cudnnCreateActivationDescriptor(&activDesc) );
    checkCUDNN( cudnnSetActivationDescriptor(activDesc,
                                             A, // CUDNN_ACTIVATION_RELU,
                                             isNanProp(), // CUDNN_PROPAGATE_NAN,
                                             coeff) );
    auto tgx = gx.makeCudnnTensor;
    auto tgy = gy.makeCudnnTensor;
    auto tx = x.makeCudnnTensor;
    auto ty = y.makeCudnnTensor;
    checkCUDNN( cudnnActivationBackward(cudnnHandle,
                                        activDesc,
                                        &alpha,
                                        ty,
                                        cast(void*) ty.ptr,
                                        tgy,
                                        cast(void*) tgy.ptr,
                                        tx,
                                        cast(void*) tx.ptr,
                                        &beta,
                                        tgx,
                                        cast(void*) tgx.ptr,
                    ) );
}

/// compute the softmax over all C for each H, W, N
void softmaxForward(cudnnSoftmaxAlgorithm_t A, T, size_t dim)(
    Variable!(T, dim, DeviceStorage) x, Variable!(T, dim, DeviceStorage) y, T alpha=1.0, T beta=0.0) {
    static assert(dim <= 4, "cuDNN only supports <= 4 dim tensors. and pack dim is not supported yet.");
    checkCUDNN( cudnnSoftmaxForward(cudnnHandle,
                                    A,
                                    CUDNN_SOFTMAX_MODE_CHANNEL,
                                    &alpha,
                                    x.makeCudnnTensor,
                                    cast(void*) x.data.ptr,
                                    &beta,
                                    y.makeCudnnTensor,
                                    cast(void*) y.data.ptr));
}

///
void softmaxBackward(cudnnSoftmaxAlgorithm_t A, T, size_t dim)(
    Variable!(T, dim, DeviceStorage) gx, Variable!(T, dim, DeviceStorage) gy,
    Variable!(T, dim, DeviceStorage) y, T alpha=1.0, T beta=0.0) {
    static assert(dim <= 4, "cuDNN only supports <= 4 dim tensors. and pack dim is not supported yet.");
    checkCUDNN( cudnnSoftmaxBackward(cudnnHandle,
                                     A,
                                     CUDNN_SOFTMAX_MODE_CHANNEL,
                                     &alpha,
                                     y.makeCudnnTensor,
                                     cast(const void*) y.data.ptr,
                                     gy.makeCudnnTensor,
                                     cast(const void*) gy.data.ptr,
                                     &beta,
                                     gx.makeCudnnTensor,
                                     cast(void*) gx.data.ptr
                    ));
}

/**
   Tensor operation : C = op( alpha1 * A, alpha2 * B ) + beta * C

   - list of ops
    CUDNN_OP_TENSOR_ADD  = 0,
    CUDNN_OP_TENSOR_MUL  = 1,
    CUDNN_OP_TENSOR_MIN  = 2,
    CUDNN_OP_TENSOR_MAX  = 3,
    CUDNN_OP_TENSOR_SQRT = 4,
    CUDNN_OP_TENSOR_NOT  = 5,

   B tensor is ignored for CUDNN_OP_TENSOR_SQRT, CUDNN_OP_TENSOR_NOT.
*/
void tensorOp(cudnnOpTensorOp_t op, T, size_t dim)(
    Variable!(T, dim, DeviceStorage) c, Variable!(T, dim, DeviceStorage) a, Variable!(T, dim, DeviceStorage) b,
    T alpha1 = 1, T alpha2 = 1, T beta = 0
) {
    import grain.functions.common : broadcastable;
    assert(broadcastable(a, b).ok);
    cudnnOpTensorDescriptor_t opDisc;
    checkCUDNN( cudnnCreateOpTensorDescriptor(&opDisc) );
    scope(exit) cudnnDestroyOpTensorDescriptor(opDisc);
    checkCUDNN( cudnnSetOpTensorDescriptor(opDisc, op, cudnnDataType!T, isNanProp()) );
    checkCUDNN( cudnnOpTensor(cudnnHandle, opDisc,
                              &alpha1, a.makeCudnnTensor, cast(const void*) a.data.ptr,
                              &alpha2, b.makeCudnnTensor, cast(const void*) b.data.ptr,
                              &beta, c.makeCudnnTensor, cast(void*) c.data.ptr) );
}

void scale(T, size_t dim)(Variable!(T, dim, DeviceStorage) x, T alpha) {
    checkCUDNN( cudnnScaleTensor(cudnnHandle, x.makeCudnnTensor, cast(void*) x.data.ptr, &alpha) );
}

/**
   Tensor operation : C = reduce op( alpha * A ) + beta * C

   - list of op
    CUDNN_REDUCE_TENSOR_ADD          = 0,
    CUDNN_REDUCE_TENSOR_MUL          = 1,
    CUDNN_REDUCE_TENSOR_MIN          = 2,
    CUDNN_REDUCE_TENSOR_MAX          = 3,
    CUDNN_REDUCE_TENSOR_AMAX         = 4,
    CUDNN_REDUCE_TENSOR_AVG          = 5,
    CUDNN_REDUCE_TENSOR_NORM1        = 6,
    CUDNN_REDUCE_TENSOR_NORM2        = 7,
    CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS = 8,

   The NaN propagation enum applies to only the min and max reduce ops;
   the other reduce ops propagate NaN as usual.
   The indices space is ignored for reduce ops other than min or max.
*/
void reduce(cudnnReduceTensorOp_t op, T, size_t dim)(
    Variable!(T, dim, DeviceStorage) src, Variable!(T, dim, DeviceStorage) dst, T alpha=1, T beta=0)
{
    // create tensor
    auto srcDesc = src.makeCudnnTensor;
    auto dstDesc = dst.makeCudnnTensor;

    // create descriptor
    cudnnReduceTensorDescriptor_t opDesc;
    checkCUDNN( cudnnCreateReduceTensorDescriptor(&opDesc) );
    scope(exit) cudnnDestroyReduceTensorDescriptor(opDesc);
    checkCUDNN( cudnnSetReduceTensorDescriptor(
                    opDesc, op, cudnnDataType!T, isNanProp(),
                    CUDNN_REDUCE_TENSOR_NO_INDICES, // CUDNN_REDUCE_TENSOR_FLATTENED_INDICES for backprop?
                    CUDNN_32BIT_INDICES // only uint is supported in cudnn7
                    ) );

    // create indices (for backprop???)
    size_t indicesBytes;
    checkCUDNN( cudnnGetReductionIndicesSize(cudnnHandle, opDesc, srcDesc, dstDesc, &indicesBytes) );
    auto indices = CuPtr!uint(indicesBytes / uint.sizeof);

    // create workspace
    size_t workspaceBytes;
    checkCUDNN( cudnnGetReductionWorkspaceSize(cudnnHandle, opDesc, srcDesc, dstDesc, &workspaceBytes) );
    auto workspace = CuPtr!byte(workspaceBytes);

    checkCUDNN( cudnnReduceTensor(
                    cudnnHandle, opDesc,
                    cast(void*) indices.ptr, indicesBytes,
                    cast(void*) workspace.ptr, workspaceBytes,
                    cast(const void*) &alpha, srcDesc, cast(const void*) srcDesc.ptr,
                    cast(const void*) &beta, dstDesc, cast(void*) dstDesc.ptr
                    ) );
}
