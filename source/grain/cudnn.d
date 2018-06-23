/**
   cuDNN high level wrapper for grain.autograd.Variable

   TODO: support global workspace instead of frequent allocation
 */
module grain.cudnn;

version (grain_cuda):

public import grain.cuda : cudnnHandle, checkCUDNN, CuPtr, isDeviceMemory;
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

/// convert contiguous cuda storage to 1-D tensor disc
auto makeCudnnTensor(T)(ref T storage) if (isDeviceMemory!T) {
    import grain.cuda : CudaElementType;
    assert(storage.length <= int.max);
    int[1] shape = [cast(int) storage.length];
    int[1] strides = [1];
    int ddim = 1;
    TensorDesc tdesc;
    tdesc.ptr = storage.ptr;
    checkCUDNN(cudnnCreateTensorDescriptor(&tdesc.desc));
    checkCUDNN(cudnnSetTensorNdDescriptor(tdesc.desc,
                                          cudnnDataType!(CudaElementType!T),
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

void fill(T, size_t dim)(Variable!(T, dim, DeviceStorage) x, T value) {
    checkCUDNN( cudnnSetTensor(cudnnHandle, x.makeCudnnTensor, cast(void*) x.data.ptr, cast(const void*) &value) );
}

///
bool isContiguous(T, size_t dim, alias Storage)(Variable!(T, dim, Storage) x) {
    // FIXME reconsider this when I support reshape, reversed and transposed
    bool ret = x.strides[$-1] == 1;
    int s = 1;
    foreach_reverse(i; 0..dim-1) {
        ret &= x.strides[i] == x.strides[i + 1] * x.shape[i+1];
    }
    return ret;
}

///
unittest {
    {
        import std.stdio;
        auto x = [[0.1f, 0.2f], [0.3f, 0.4f]].variable;
        writeln(x.strides);
        assert(x.isContiguous);
        x.strides = [2, 2];
        assert(!x.isContiguous);
    }
    version (grain_cuda) {
        auto x = [[0.1f, 0.2f], [0.3f, 0.4f]].variable.to!DeviceStorage;
        assert(x.isContiguous);
        x.strides = [2, 2];
        assert(!x.isContiguous);
    }
}


void transform(T, size_t dim)(Variable!(T, dim, DeviceStorage) src, ref Variable!(T, dim, DeviceStorage) dst, T alpha=1, T beta=0) {
    assert(src.shape == dst.shape);

    if (src.isContiguous && dst.isContiguous && beta == 1) {
        import grain.cuda : axpy;
        axpy(src.data, dst.data, alpha);
        return;
    }

    checkCUDNN(
        cudnnTransformTensor(
            cudnnHandle,
            cast(const void*) &alpha, src.makeCudnnTensor, cast(const void*) src.data.ptr,
            cast(const void*) &beta, dst.makeCudnnTensor, cast(void*) dst.data.ptr
            ) );
}

auto contiguous(T, size_t dim)(Variable!(T, dim, DeviceStorage) x) {
    auto y = x.uninit;
    y.bprop = x.bprop;
    transform(x, y);
    return y;
}

/// test cudnnTransformTensor with array ptr manipulations
unittest {
    import std.stdio;
    // skipping stride 2
    {
        auto x = [1f, 0f, 2f, 0f, 3f].variable;
        x.strides = [2];
        x.shape = [3];
        auto y = x.to!DeviceStorage.contiguous.to!HostStorage;
        assert(y.data == [1f, 2f, 3f]);
        assert(y.strides == [1]);
        assert(y.shape == [3]);
    }
    // reverse skipping stride -2
    {
        auto x = [1f, 0f, 2f, 0f, 3f].variable;
        x.strides = [-2];
        x.shape = [3];
        auto dx = x.to!DeviceStorage;
        dx.data.ptr += 4 * float.sizeof;
        scope(exit) dx.data.ptr -= 4 * float.sizeof;
        auto y = dx.contiguous.to!HostStorage;
        assert(y.data == [3f, 2f, 1f]);
        assert(y.strides == [1]);
        assert(y.shape == [3]);
    }
    // multi-dim transposed stride [3, 1]
    {
        auto x = [[1f, 0f, 2f],
                  [0f, 3f, 0f]].variable;
        x.strides = [1, 3];
        x.shape = [3, 2];
        auto dx = x.to!DeviceStorage;
        auto y = dx.contiguous.to!HostStorage;
        assert(y.sliced == [[1f, 0f], [0f, 3f], [2f, 0f]]);
        assert(y.strides == [2, 1]);
        assert(y.shape == [3, 2]);
    }
    // multi-dim skipping stride [3, 2]
    {
        auto x = [[1f, 0f, 2f],
                  [0f, 3f, 0f]].variable;
        x.strides = [3, 2];
        x.shape = [2, 2];
        auto dx = x.to!DeviceStorage;
        auto y = dx.contiguous.to!HostStorage;
        assert(y.sliced == [[1f, 2f],  [0f, 0f]]);
        assert(y.strides == [2, 1]);
        assert(y.shape == [2, 2]);
    }
    // multi-dim transposed skipping stride [2, 3]
    {
        auto x = [[1f, 0f, 2f],
                  [0f, 3f, 0f]].variable;
        x.strides = [2, 3];
        x.shape = [2, 2];
        auto dx = x.to!DeviceStorage;
        // dx.data.ptr += (2 * 3 - 1) * float.sizeof;
        // scope(exit) dx.data.ptr -= (2 * 3 - 1) * float.sizeof;
        auto y = dx.contiguous.to!HostStorage;
        assert(y.sliced == [[1f, 0f],  [2f, 0f]]);
        assert(y.strides == [2, 1]);
        assert(y.shape == [2, 2]);
    }
    // multi-dim transposed reverse skipping stride [-2, -3]
    {
        auto x = [[1f, 0f, 2f],
                  [0f, 3f, 0f]].variable;
        x.strides = [-2, -3];
        x.shape = [2, 2];
        auto dx = x.to!DeviceStorage;
        dx.data.ptr += (2 * 3 - 1) * float.sizeof;
        scope(exit) dx.data.ptr -= (2 * 3 - 1) * float.sizeof;
        auto y = dx.contiguous.to!HostStorage;
        assert(y.sliced == [[0f, 2f],  [0f, 1f]]);
        assert(y.strides == [2, 1]);
        assert(y.shape == [2, 2]);
    }

}


void convForward(cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                 cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION,
                 cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW,
                 T, size_t dim)
    (
     Variable!(T, dim, DeviceStorage) input,      // [N, CI, HI, WI]
     Variable!(T, dim, DeviceStorage) filter,     // [CO, CI/G, KH, KW]
     ref Variable!(T, dim, DeviceStorage) output, // [N, CO, HO, WO]
     const int[dim-2]   stride,
     const int[dim-2]   pad,
     const int[dim-2]   dilation,
     int ngroup = 1,
     float alpha = 1,
     float beta = 0
     ) {
    static assert(dim < CUDNN_DIM_MAX);

    // TODO cache these?
    cudnnFilterDescriptor_t cudnnFdesc;
    checkCUDNN( cudnnCreateFilterDescriptor(&cudnnFdesc) );
    scope(exit) cudnnDestroyFilterDescriptor(cudnnFdesc);
    checkCUDNN( cudnnSetFilterNdDescriptor(cudnnFdesc, cudnnDataType!T, format,
                                           cast(int) dim, filter.shape.ptr
                                           ) );

    cudnnConvolutionDescriptor_t cudnnConvDesc;
    checkCUDNN( cudnnCreateConvolutionDescriptor(&cudnnConvDesc) );
    scope(exit) cudnnDestroyConvolutionDescriptor(cudnnConvDesc);
    checkCUDNN( cudnnSetConvolutionGroupCount(cudnnConvDesc, ngroup) );
    checkCUDNN( cudnnSetConvolutionNdDescriptor(cudnnConvDesc, cast(int) dim,
                                                pad.ptr, stride.ptr, dilation.ptr,
                                                mode, cudnnDataType!T
                                                ) );

    auto cudnnIdesc = input.makeCudnnTensor;
    auto cudnnOdesc = output.makeCudnnTensor;
    size_t workSpaceSize;
    checkCUDNN ( cudnnGetConvolutionForwardWorkspaceSize
                 (cudnnHandle, cudnnIdesc, cudnnFdesc, cudnnConvDesc,
                  cudnnOdesc, algo, &workSpaceSize) );
    auto workSpace = CuPtr!byte(workSpaceSize);

    checkCudnnErr ( cudnnConvolutionForward (cudnnHandle,
                                             cast(const void*) &alpha,
                                             cudnnIdesc, cast(const void*) input.data.ptr,
                                             cudnnFdesc, cast(const void*) filter.data.ptr,
                                             cudnnConvDesc,
                                             algo,
                                             cast(void*) workSpace.data.ptr, workSpaceSize,
                                             cast(const void*) &beta,
                                             cudnnOdesc, cast(void*) output.data.ptr) );
}


void convBackward(cudnnConvolutionBwdDataAlgo_t algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
                  cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION,
                  cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW,
                  T, size_t dim)
    (
     ref Variable!(T, dim, DeviceStorage) gradInput,      // [N, CI, HI, WI]
     Variable!(T, dim, DeviceStorage) input,      // [N, CI, HI, WI]
     ref Variable!(T, dim, DeviceStorage) gradFilter,     // [CO, CI/G, KH, KW]
     Variable!(T, dim, DeviceStorage) filter,     // [CO, CI/G, KH, KW]
     Variable!(T, dim, DeviceStorage) gradOutput, // [N, CO, HO, WO]
     const int[dim-2]   stride,
     const int[dim-2]   pad,
     const int[dim-2]   dilation,
     int ngroup = 1,
     float alpha = 1,
     float beta = 0
) {
    static assert(dim < CUDNN_DIM_MAX);

    // TODO cache these?
    cudnnFilterDescriptor_t cudnnFdesc;
    checkCUDNN( cudnnCreateFilterDescriptor(&cudnnFdesc) );
    scope(exit) cudnnDestroyFilterDescriptor(cudnnFdesc);
    checkCUDNN( cudnnSetFilterNdDescriptor(cudnnFdesc, cudnnDataType!T, format,
                                           cast(int) dim, filter.shape.ptr
                                           ) );

    cudnnConvolutionDescriptor_t cudnnConvDesc;
    checkCUDNN( cudnnCreateConvolutionDescriptor(&cudnnConvDesc) );
    scope(exit) cudnnDestroyConvolutionDescriptor(cudnnConvDesc);
    checkCUDNN( cudnnSetConvolutionGroupCount(cudnnConvDesc, ngroup) );
    checkCUDNN( cudnnSetConvolutionNdDescriptor(cudnnConvDesc, cast(int) dim,
                                                pad.ptr, stride.ptr, dilation.ptr,
                                                mode, cudnnDataType!T
                                                ) );

    auto cudnnIdesc = input.makeCudnnTensor;
    auto cudnnOdesc = output.makeCudnnTensor;
    auto cudnnGIdesc = gradInput.makeCudnnTensor;
    auto cudnnGOdesc = gradOutput.makeCudnnTensor;

    size_t dworkSpaceSize;
    checkCudnnErr ( cudnnGetConvolutionBackwardDataWorkspaceSize
                    (cudnnHandle, cudnnFdesc, cudnnGOdesc, cudnnConvDesc,
                     cudnnGIdesc, algo, &dworkSpaceSize) );
    auto dworkSpace = CuPtr!byte(workSpaceSize);
    checkCudnnErr ( cudnnConvolutionBackwardData (cudnnHandle,
                                                  cast(void*)(&alpha),
                                                  cudnnFdesc, cast(const void*) filter.data.ptr,
                                                  cudnnGOdesc, cast(const void*) gradOutput.data.ptr,
                                                  cudnnConvDesc,
                                                  algo,
                                                  cast(const void*) dworkSpace.ptr, dworkSpaceSize,
                                                  cast(const void*)(&beta),
                                                  cudnnGIdesc, cast(void*) gradInput.data.ptr) );

    size_t fworkSpaceSize;
    checkCudnnErr ( cudnnGetConvolutionBackwardFilterWorkspaceSize
                    (cudnnHandle, cudnnIdesc, cudnnGOdesc, cudnnConvDesc,
                     cudnnFdesc, algo, &fworkSpaceSize) );
    auto fworkSpace = CuPtr!byte(workSpaceSize);
    checkCudnnErr ( cudnnConvolutionBackwardFilter (cudnnHandle,
                                                    cast(const void*)(&alpha),
                                                    cudnnIdesc,  cast(const void*) input.data.ptr,
                                                    cudnnGOdesc, cast(const void*) gradOutput.data.ptr,
                                                    cudnnConvDesc,
                                                    balgo,
                                                    cast(void*) fworkSpace.ptr, fworkSpaceSize,
                                                    cast(const void*)(&beta),
                                                    cudnnFdesc, cast(void*) gradFilter.data.ptr) );
}

// TODO setup convolutionBias or find other way (just use add)

