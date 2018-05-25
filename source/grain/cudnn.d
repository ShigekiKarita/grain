module grain.cudnn;

public import grain.cuda : cudnnHandle, checkCUDNN;
import grain.autograd; //  : Variable, DeviceStorage;
public import derelict.cuda;
public import derelict.cudnn7;

auto cudnnDataType(T)() {
    // TODO support half
    static if(is(T == float)) return CUDNN_DATA_FLOAT;
    else static if(is(T == double)) return CUDNN_DATA_DOUBLE;
    else static assert(false, "unsupported type");
}


private struct TensorDesc {
    cudnnTensorDescriptor_t desc;
    CUdeviceptr ptr;
    alias desc this;

    @disable this(this); // no copy

    ~this() {
        checkCUDNN( cudnnDestroyTensorDescriptor(desc) );
    }
}


auto makeCudnnTensor(T, size_t dim)(Variable!(T, dim, DeviceStorage) x) {
    static assert(dim < CUDNN_DIM_MAX);
    static if (dim < 4) {
        enum int ddim = 4;
        int[ddim] shape, strides;
        shape[] = 1;
        strides[] = 1;
        shape[0..dim] = x.shape;
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

void activationForward(cudnnActivationMode_t A, T, size_t dim)(
    Variable!(T, dim, DeviceStorage) vx, Variable!(T, dim, DeviceStorage) vy,
    T alpha=1.0, T beta=0.0, double coeff=0.0) {
    // init descriptors
    cudnnActivationDescriptor_t  activDesc;
    checkCUDNN( cudnnCreateActivationDescriptor(&activDesc) );
    scope(exit) checkCUDNN( cudnnCreateActivationDescriptor(&activDesc) );
    checkCUDNN( cudnnSetActivationDescriptor(activDesc,
                                             A, // CUDNN_ACTIVATION_RELU,
                                             CUDNN_PROPAGATE_NAN,
                                             coeff) );
    auto x = vx.makeCudnnTensor;
    auto y = vy.makeCudnnTensor;
    checkCUDNN( cudnnActivationForward(cudnnHandle,
                                       activDesc,
                                       &alpha,
                                       x,
                                       cast(void*) x.ptr,
                                       &beta,
                                       y,
                                       cast(void*) y.ptr) );
}
