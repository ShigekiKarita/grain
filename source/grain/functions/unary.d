/**
   A module for unary functions

   TODO: support cudnn functions (see PDF manual in .deb for detail https://developer.nvidia.com/cudnn)
   - activation (e.g., clipped-relu, elu), cudnnActivationForward/Backward
   - (non-log) softmax, cudnnSoftmaxForward/Backward
   - sqrt not, cudnnOpTensor
   - transform (e.g., contiguous or permute strides), cudnnTransformTensor
   - reshape (i.e., view), ...???
   - reduce (sum, prod, min, max, amax, avg), cudnnReduceTensor
   - pool (max, average), cudnnPoolingForward/Backward
   - dropout, cudnnDropoutForward/Backward
 */
module grain.functions.unary;

import grain.autograd;
import grain.cuda;
import grain.utility;
import grain.functions.common;

version (grain_cuda) {
    import grain.cudnn;
    // FIXME do not know why this mixin won't work
    // mixin template CudnnActivation(T, size_t dim, cudnnActivationMode_t mode) {

    ///
    enum CUDNN_ACTIVATION_IMPL_MIXIN = q{
        // TODO support inplace
        Variable!(T, dim, DeviceStorage) dx, dy;
        ///
        auto forward(Variable!(T, dim, DeviceStorage) x) {
            // FIXME if train
            this.dx = x.dup;
            auto y = x.uninit;
            activationForward!mode(x, y);
            this.dy = y;
            return y;
        }
        ///
        auto backward(Variable!(T, dim, DeviceStorage) gy) {
            auto gx = gy.uninit;
            activationBackward!mode(gx, gy, this.dx, this.dy);
            return gx;
        }
    };
}

/// sigmoid function
struct Sigmoid(T, size_t dim) {
    import mir.math : exp;
    import std.math : tanh;
    import mir.ndslice : sliced, slice, map;

    Variable!(T, dim, HostStorage) hy;

    ///
    auto forward(Variable!(T, dim, HostStorage) x) {
        enum z = T(0.5);
        auto ys = x.sliced.map!(a => tanh(a * z) * z + z);
        auto y = ys.slice.variable(x.requiresGrad);
        this.hy = y;
        return y;
    }

    ///
    auto backward(Variable!(T, dim, HostStorage) gy) {
        auto gx = this.hy.dup;
        gx.sliced[] *= T(1) - this.hy.sliced;
        gx.sliced[] *= gy.sliced;
        return gx;
    }

    version (grain_cuda) {
        // mixin CudnnActivation!(T, dim, CUDNN_ACTIVATION_TANH);
        enum mode = CUDNN_ACTIVATION_SIGMOID;
        mixin(CUDNN_ACTIVATION_IMPL_MIXIN);
    }

    mixin FunctionCommon;
}

///
unittest {
    // test CPU
    import grain.testing;
    import std.math : tanh;
    import numir;
    auto func = new Sigmoid!(float, 1);
    auto hx = [-1.0f, 1.0f, 0.0f].variable;
    gradCheck(func, hx, [0.1f, 0.1f, 0.1f].variable);

    auto hy = func.forward(hx);
    // assert(hy.data == [tanh(-1.0f), tanh(1.0f), tanh(0.0f)]);
    auto hgy = [1.0f, 2.0f, 3.0f].variable;
    auto hgx = func.backward(hgy);

    // test CUDA
    version(grain_cuda) {
        auto dfunc = new Sigmoid!(float, 1);
        auto dx = hx.to!DeviceStorage;
        auto dy = dfunc.forward(dx);
        assert(approxEqual(dy.to!HostStorage.sliced, hy.sliced));
        auto dgx = dfunc.backward(hgy.to!DeviceStorage);
        assert(approxEqual(dgx.to!HostStorage.sliced, hgx.sliced));
    }
}


/// hyperbolic tangent
struct Tanh(T, size_t dim) {
    import std.math : tanh;
    import mir.ndslice : sliced, slice, map;

    Variable!(T, dim, HostStorage) hy;

    ///
    auto forward(Variable!(T, dim, HostStorage) x) {
        auto ys = x.sliced.map!tanh;
        auto y = ys.slice.variable(x.requiresGrad);
        this.hy = y;
        return y;
    }

    ///
    auto backward(Variable!(T, dim, HostStorage) gy) {
        auto gx = this.hy.dup;
        gx.sliced[] *= gx.sliced; // hy ^^ 2
        gx.sliced[] = T(1.0) - gx.sliced;
        gx.sliced[] *= gy.sliced;
        return gx; // .slice.variable;
    }

    version (grain_cuda) {
        // mixin CudnnActivation!(T, dim, CUDNN_ACTIVATION_TANH);
        enum mode = CUDNN_ACTIVATION_TANH;
        mixin(CUDNN_ACTIVATION_IMPL_MIXIN);
    }

    mixin FunctionCommon;
}

///
unittest {
    // test CPU
    import grain.testing;
    import std.math : tanh;
    import numir;
    auto func = new Tanh!(float, 1);
    auto hx = [-1.0f, 1.0f, 0.0f].variable;
    gradCheck(func, hx, [0.1f, 0.1f, 0.1f].variable);

    auto hy = func.forward(hx);
    assert(hy.data == [tanh(-1.0f), tanh(1.0f), tanh(0.0f)]);
    auto hgy = [1.0f, 2.0f, 3.0f].variable;
    auto hgx = func.backward(hgy);

    // test CUDA
    version(grain_cuda) {
        auto dfunc = new Tanh!(float, 1);
        auto dx = hx.to!DeviceStorage;
        auto dy = dfunc.forward(dx);
        assert(approxEqual(dy.to!HostStorage.sliced, hy.sliced));
        auto dgx = dfunc.backward(hgy.to!DeviceStorage);
        assert(approxEqual(dgx.to!HostStorage.sliced, hgx.sliced));
    }
}

/// TODO implement scale with cudnnScaleTensor

/// rectified linear unit nonlinearity (using cuDNN)
struct ReLU(T, size_t dim) {
    mixin FunctionCommon;
    bool inplace = false;
    bool useCuDNN = true;
    Variable!(T, dim, HostStorage) hx;

    auto forward(Variable!(T, dim, HostStorage) x) {
        import mir.ndslice : each;
        // FIXME if train
        this.hx = x.dup;
        auto y = this.inplace ? x : x.dup;
        y.sliced.each!((ref a) { if (a < 0) a = 0; });
        return y;
    }

    auto backward(Variable!(T, dim, HostStorage) gy) {
        auto gx = gy.dup;
        foreach (i; 0..gx.data.length) {
            if (this.hx.data[i] < 0.0) gx.data[i] = 0.0;
        }
        return gx;
    }

    // TODO use cudnn
    version(grain_cuda) {
        import grain.cudnn;
        Variable!(T, dim, DeviceStorage) dx, dy;

        auto forward(Variable!(T, dim, DeviceStorage) x) {
            // FIXME if train
            this.dx = x.dup;
            auto y = this.inplace ? x : x.dup;

            if (this.useCuDNN) {
                activationForward!CUDNN_ACTIVATION_RELU(x, y);
                this.dy = y;
            } else {
                import grain.kernel : relu;
                auto n = cast(uint) y.data.length; // FIXME use y.nElement
                Global.kernel!relu
                    .call(y.data.ptr, n).launch(n);
            }
            return y;
        }

        auto backward(Variable!(T, dim, DeviceStorage) gy) {
            auto gx = gy.uninit;
            if (this.useCuDNN) {
                activationBackward!CUDNN_ACTIVATION_RELU(gx, gy, dx, dy);
            } else {
                import grain.kernel : reluGrad;
                auto n = cast(uint) gy.data.length;
                Global.kernel!reluGrad
                    .call(gx.data.ptr, gy.data.ptr, this.dx.data.ptr, n).launch(n);
            }
            return gx;
        }
    }
}

/// test relu
unittest {
    import grain.testing : gradCheck;
    foreach (inplace; [true, false]) {
        foreach (useCuDNN; [true, false]) {
            auto func = new ReLU!(float, 1);
            func.inplace = inplace;
            func.useCuDNN = useCuDNN;

            // test CPU
            {
                auto x = [-1.0f, 1.0f, 0.0f].variable;
                // fail because of non-smooth function?
                // gradCheck(func, x, [0.1f, 0.1f, 0.1f].variable);

                auto y = func.forward(x);
                assert(x.data == (inplace ? y.data : [-1.0f, 1.0f, 0.0f]));
                assert(y.data == [0.0f, 1.0f, 0.0f]);

                auto gy = [1.0f, 2.0f, 3.0f].variable;
                auto gx = func.backward(gy);
                assert(gx.data == [0.0f, 2.0f, 3.0f]);
            }

            // test CUDA
            version(grain_cuda) {
                auto x = [-1.0f, 1.0f, 0.0f].variable;
                auto xd = x.to!DeviceStorage;
                auto yd = func.forward(xd);
                x = xd.to!HostStorage;
                auto y = yd.to!HostStorage;
                assert(x.data == (inplace ? y.data : [-1.0f, 1.0f, 0.0f]));
                assert(y.data == [0.0f, 1.0f, 0.0f]);

                x = [-1.0f, 1.0f, 0.0f].variable;
                auto gy = [1.0f, 2.0f, 3.0f].variable;
                auto gxd = func.backward(gy.to!DeviceStorage);
                auto gx = gxd.to!HostStorage;
                assert(gx.data == [0.0, 2.0, 0.0]);
            }
        }
    }
}

// forward two functions parallel
unittest {
    import std.typecons;
    grain.autograd.backprop = true;
    // scope (exit) grain.autograd.backprop = false;
    {
        auto x = [-1.0f, 2.0f, 3.0f].variable(true);
        x.requiresGrad = true;
        Variable!(float, 1) y, h;
        y.requiresGrad = true;
        h.requiresGrad = true;
        // bprop will survive even if deeper scope
        {
            // FIXME cannot use RefCounted instead of new here
            // RefCounted!(ReLU!(float, 1)) func0 = ReLU!(float, 1)();
            auto func0 = new ReLU!(float, 1);
            h = func0.applyForward(x);
            assert(h.bprop.inputs[0].data == x.data);
            auto func1 = new ReLU!(float, 1);
            y = func1.applyForward(h);
            // test the chain to backprop
            assert(y.bprop.inputs[0].data == h.data);
            assert(y.bprop.inputs[0].bprop.inputs[0].data == x.data);
        }
        auto gy = [1.0f, 2.0f, 3.0f].variable;
        auto ugy = UntypedVariable(gy);
        y.backward(&ugy);
        assert(x.grad == [0, 2, 3]);

        auto func2 = new ReLU!(float, 1);
        auto y2 = func2.applyForward(x);
        y2.backward(&ugy);
        assert(x.grad == [0, 4, 6]); // summation
    }
    version (grain_cuda) {
        auto func = new ReLU!(float, 1);
        auto x = [-1.0f, 2.0f, 3.0f].variable(true).to!DeviceStorage;
        auto y = func.applyForward(x);
        auto gy = [1.0f, 2.0f, 3.0f].variable.to!DeviceStorage;
        auto ugy = UntypedVariable(gy);
        y.backward(&ugy);
        assert(x.grad.toHost() == [0, 2, 3]);

        auto func2 = new ReLU!(float, 1);
        auto y2 = func.applyForward(x);
        y2.backward(&ugy);
        assert(x.grad.toHost() == [0, 4, 6]); // summation
    }
}



// TODO add to numir
import mir.ndslice : isSlice;
import numir : Ndim;
pure nothrow @nogc
logsumexp(S)(S x) if (isSlice!S && Ndim!S == 1) {
    import mir.ndslice : map, maxIndex;
    import mir.math : log, sum, exp;
    auto m = x[x.maxIndex];
    auto s = map!exp(x - m).sum!"fast".log;
    return m + s;
}

///
pure nothrow @nogc
unittest {
    import numir;
    import mir.ndslice;
    // import mir.math;
    import std.math;
    static immutable x = [-1.0, 2.0, 3.0];
    static immutable e = log(exp(-1.0) + exp(2.0) + exp(3.0));
    assert(approxEqual(x.sliced.logsumexp, e));
    static immutable xs = [-1.0, 2.0, 3.0,
                           -1.0, 2.0, 3.0,
                           -1.0, 2.0, 3.0];
    static immutable es = [e, e, e];
    assert(approxEqual(xs.sliced(3, 3).alongDim!1.map!logsumexp, es));
}


/++
See_also: https://github.com/chainer/chainer/blob/v1/chainer/functions/activation/log_softmax.py
 +/
struct LogSoftmax(T, size_t dim=2) {
    // TODO support custom dim to compute softmax over (now only dim=1)
    mixin FunctionCommon;

    Variable!(T, dim, HostStorage) hy;

    auto forward(Variable!(T, dim, HostStorage) x) {
        import mir.ndslice;
        import numir;
        // return slice(x.sliced.alongDim!0.map!(e => e - e.logsumexp)).variable;
        auto y = x.dup;
        foreach (i; 0 .. y.shape[0]) {
            y.sliced[i][] -= x.sliced[i].logsumexp;
        }
        // TODO if train
        this.hy = y;
        return y;
    }

    auto backward(Variable!(T, dim, HostStorage) gy) {
        import mir.math;
        import numir;
        import mir.ndslice;
        auto gx = gy.dup;
        auto m = gy.sliced.alongDim!1.map!(sum!"fast");
        foreach (i; 0 .. gx.shape[0]) {
            gx.sliced[i][] -= this.hy.sliced[i].map!exp * m[i];
        }
        return gx;
    }

    version (grain_cuda) {
        import grain.cudnn;
        Variable!(T, dim, DeviceStorage) dy;

        auto forward(Variable!(T, dim, DeviceStorage) x) {
            auto y = x.dup;
            softmaxForward!CUDNN_SOFTMAX_LOG(x, y);
            // TODO if train
            this.dy = y;
            return y;
        }

        auto backward(Variable!(T, dim, DeviceStorage) gy) {
            auto gx = gy.dup;
            softmaxBackward!CUDNN_SOFTMAX_LOG(gx, gy, this.dy);
            return gx;
        }
    }
}

/// test logsoftmax simple case, gradcheck and cpu/cuda equality
unittest {
    import grain.testing;
    import std.typecons;
    import numir;
    import mir.ndslice;
    import mir.math;
    auto e = log(exp(-1.0) + exp(2.0) + exp(3.0));
    auto xs = [[-1.0f, 2.0f, 3.0f], [-1.0f, 2.0f, 3.0f], [-1.0f, 2.0f, 3.0f]].nparray;
    LogSoftmax!float hfunc;
    auto _hx = xs.variable;
    auto _hy = hfunc.forward(_hx);
    assert(approxEqual(_hy.sliced, xs - e));

    auto hx = uniform!float(2, 2).slice.variable;
    auto hy = hfunc.forward(hx);
    auto hgy = uniform!float(2, 2).slice.variable;
    auto hgx = hfunc.backward(hgy);
    gradCheck(hfunc, hx, hgy, 1e-3, 1e-3, 1e-3);

    version (grain_cuda) {
        alias Storage = DeviceStorage;
        auto func = LogSoftmax!float();
        auto dx = hx.to!Storage;
        auto dy = func.forward(dx);
        assert(approxEqual(dy.to!HostStorage.sliced, hy.sliced));
        auto dgy = hgy.to!Storage;
        auto dgx = func.backward(dgy);
        assert(approxEqual(dgx.to!HostStorage.sliced, hgx.sliced));
    }
}


/// wrapper of CUDA kernel unary functions
void unaryFunc(alias K, size_t dim)(Variable!(float, dim, DeviceStorage) x) {
    auto shape = CuPtr!uint(x.shape[0..$]);
    auto strides = CuPtr!int(x.strides[0..$]);
    auto ndim = cast(uint) dim;
    auto len = cast(uint) x.data.length;
    Global.kernel!K
        .call(x.data.ptr, len, ndim, shape.ptr, strides.ptr)
        .launch(len);
}

/// test neg kernel
version (grain_cuda) unittest {
    import numir;
    import grain.kernel;
    auto x = [[1f, 2f, 3f], [4f, 5f, 6f]].variable.to!DeviceStorage;
    unaryFunc!neg(x);
    assert(x.to!HostStorage.sliced == -  [[1f, 2f, 3f], [4f, 5f, 6f]].nparray);
}


/// test reciprocal kernel
version (grain_cuda) unittest {
    import grain.kernel;
    auto x = [[1f, 2f, 3f], [4f, 5f, 6f]].variable.to!DeviceStorage;
    unaryFunc!reciprocal(x);
    assert(x.to!HostStorage.sliced == [[1f,1f/2f,1f/3f], [1f/4f,1f/5f,1f/6f]]);
}

/**
   test math function kernels. these functions are available in mir.math (as LDC intrinsic) or CUDA fast math

   See_also:
   - http://mir.dlang.io/mir_math_common.html
   - https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#intrinsic-functions
*/
version (grain_cuda) unittest {
    // FIXME mir.math.log will exit 1
    import std.math : log, tan; // , log2, log10, exp, exp2, cos, sin, tan;
    import mir.math : log2, log10, exp, exp2, cos, sin;
    import grain.kernel : log, log2, log10, exp, exp2, cos, sin, tan;
    import std.format : format;
    import numir : approxEqual;
    import mir.ndslice : iota, as, slice, map;
    static foreach (name; ["log", "log2", "log10", "exp", "exp2", "cos", "sin", "tan"]) {
        {
            auto x = iota([2, 3], 1).as!float.slice.variable.to!DeviceStorage;
            mixin(format!q{  unaryFunc!(grain.kernel.%s)(x);  }(name));
            mixin(format!q{  alias func = %s;  }(name));
            assert(approxEqual(x.to!HostStorage.sliced, iota([2, 3], 1).as!float.map!func));
        }
    }
}

/// wrapper of CUDA kernel pow function
void unaryPow(size_t dim)(Variable!(float, dim, DeviceStorage) x, float power) {
    import grain.kernel : pow;
    auto shape = CuPtr!uint(x.shape[0..$]);
    auto strides = CuPtr!int(x.strides[0..$]);
    auto ndim = cast(uint) dim;
    auto len = cast(uint) x.data.length;
    Global.kernel!pow
        .call(power, x.data.ptr, len, ndim, shape.ptr, strides.ptr)
        .launch(len);
}

/// test pow kernel
version (grain_cuda) unittest {
    import numir;
    import mir.ndslice;
    import grain.kernel;
    import mir.math : pow;
    auto x = iota([2, 3], 1).as!float.slice.variable.to!DeviceStorage;
    unaryPow(x, 2f);
    assert(approxEqual(x.to!HostStorage.sliced,
                       iota([2, 3], 1).as!float.map!(x => pow(x, 2))));
}


/// y = 1 / x
struct Reciprocal(T, size_t dim) {
    mixin FunctionCommon;

    Variable!(T, dim, HostStorage) hy;

    auto forward(Variable!(T, dim, HostStorage) x) {
        import mir.ndslice : map, slice;
        auto y = x.sliced.map!(a => T(1) / a).slice.variable(x.requiresGrad);
        this.hy = y; // TODO if train
        return y;
    }

    auto backward(Variable!(T, dim, HostStorage) gy) {
        auto gx = this.hy.dup;
        gx.sliced[] *= gx.sliced;
        gx.sliced[] *= -gy.sliced;
        return gx;
    }

    version (grain_cuda) {
        Variable!(T, dim, DeviceStorage) dy;

        auto forward(Variable!(T, dim, DeviceStorage) x) {
            import grain.kernel : reciprocal;
            auto y = x.dup;
            unaryFunc!reciprocal(y);
            this.dy = y; // TODO if train
            return y;
        }

        auto backward(Variable!(T, dim, DeviceStorage) gy) {
            import grain.cudnn : tensorOp, CUDNN_OP_TENSOR_MUL;
            auto gx = this.dy.dup;
            tensorOp!CUDNN_OP_TENSOR_MUL(gx, gx, this.dy, T(-1));
            tensorOp!CUDNN_OP_TENSOR_MUL(gx, gx, gy);
            return gx;
        }
    }
}

/// test reciprocal simple case, gradcheck and cpu/cuda equality
unittest {
    import grain.testing;
    import std.typecons;
    import numir;
    import mir.ndslice;

    // simple case
    auto e = [[-1.0f,1f/2f,1f/3f], [1f, 10f,1f/3f]].nparray;
    auto xs = [[-1.0f, 2.0f, 3.0f], [1.0f, 0.1f, 3.0f]].nparray;
    Reciprocal!(float, 2) hfunc;
    auto _hx = xs.variable;
    auto _hy = hfunc.forward(_hx);
    assert(approxEqual(_hy.sliced, e));

    auto hx = uniform!float(2, 2).slice.variable;
    auto hy = hfunc.forward(hx);
    auto hgy = uniform!float(2, 2).slice.variable;
    auto hgx = hfunc.backward(hgy);
    gradCheck(hfunc, hx, hgy, 1e-3, 5e-2, 5e-2);

    version (grain_cuda) {
        Reciprocal!(float, 2) dfunc;
        auto dy = dfunc.forward(hx.to!DeviceStorage);
        assert(approxEqual(dy.to!HostStorage.sliced, hy.sliced));
        auto dgx = dfunc.backward(hgy.to!DeviceStorage);
        assert(approxEqual(dgx.to!HostStorage.sliced, hgx.sliced));
    }
}

/// y = exp x
struct Exp(T, size_t dim) {
    import mir.ndslice : slice, map;

    mixin FunctionCommon;

    Variable!(T, dim, HostStorage) hy;

    auto forward(Variable!(T, dim, HostStorage) x) {
        import mir.math.common : exp;
        auto y = slice(x.sliced.map!exp).variable(x.requiresGrad);
        this.hy = y; // TODO if train
        return y;
    }

    auto backward(Variable!(T, dim, HostStorage) gy) {
        auto gx = gy.dup;
        gx.sliced[] *= this.hy.sliced;
        return gx;
    }

    version (grain_cuda) {
        Variable!(T, dim, DeviceStorage) dy;

        auto forward(Variable!(T, dim, DeviceStorage) x) {
            import grain.kernel : exp;
            auto y = x.dup;
            unaryFunc!exp(y);
            this.dy = y;
            return y;
        }

        auto backward(Variable!(T, dim, DeviceStorage) gy) {
            return this.dy * gy;
        }
    }
}

///
unittest {
    import grain.testing;
    import std.typecons;
    import numir;
    import mir.ndslice;
    import mir.math : exp;

    Exp!(float, 2) hfunc;
    auto hx = uniform!float(2, 3).slice.variable;
    auto hy = hfunc.forward(hx);
    auto hgy = uniform!float(2, 3).slice.variable;
    auto hgx = hfunc.backward(hgy);
    gradCheck(hfunc, hx, hgy);
    assert(approxEqual(hy.sliced, hx.sliced.map!exp));

    version (grain_cuda) {
        Exp!(float, 2) dfunc;
        auto dy = dfunc.forward(hx.to!DeviceStorage);
        assert(approxEqual(dy.to!HostStorage.sliced, hy.sliced));
        auto dgx = dfunc.backward(hgy.to!DeviceStorage);
        assert(approxEqual(dgx.to!HostStorage.sliced, hgx.sliced));
    }
}

/// y = exp x
struct Log(T, size_t dim) {
    import mir.ndslice : slice, map;

    mixin FunctionCommon;

    Variable!(T, dim, HostStorage) hx;

    auto forward(Variable!(T, dim, HostStorage) x) {
        import mir.math.common : log;
        auto y = slice(x.sliced.map!log).variable(x.requiresGrad);
        this.hx = x; // TODO if train
        return y;
    }

    auto backward(Variable!(T, dim, HostStorage) gy) {
        auto gx = gy.dup;
        gx.sliced[] /= this.hx.sliced;
        return gx;
    }

    version (grain_cuda) {
        Variable!(T, dim, DeviceStorage) dx;

        auto forward(Variable!(T, dim, DeviceStorage) x) {
            import grain.kernel : log;
            auto y = x.dup;
            unaryFunc!log(y);
            this.dx = x;
            return y;
        }

        auto backward(Variable!(T, dim, DeviceStorage) gy) {
            return gy / this.dx;
        }
    }
}

///
unittest {
    import grain.testing;
    import std.typecons;
    import numir;
    import mir.ndslice;
    import mir.math : log;

    Log!(float, 2) hfunc;
    auto hx = uniform!float(2, 3).slice.variable;
    auto hy = hfunc.forward(hx);
    auto hgy = uniform!float(2, 3).slice.variable;
    auto hgx = hfunc.backward(hgy);
    gradCheck(hfunc, hx, hgy, 1e-3, 5e-2, 5e-2);
    assert(approxEqual(hy.sliced, hx.sliced.map!log));

    version (grain_cuda) {
        Log!(float, 2) dfunc;
        auto dy = dfunc.forward(hx.to!DeviceStorage);
        assert(approxEqual(dy.to!HostStorage.sliced, hy.sliced));
        auto dgx = dfunc.backward(hgy.to!DeviceStorage);
        assert(approxEqual(dgx.to!HostStorage.sliced, hgx.sliced));
    }
}


// TODO implement these autograd fast-math functions
// struct Log2
// struct Log10

// struct Exp2
// struct Exp10

/// y = sin x
struct Sin(T, size_t dim) {
    import mir.ndslice : slice, map;

    mixin FunctionCommon;

    Variable!(T, dim, HostStorage) hx;

    auto forward(Variable!(T, dim, HostStorage) x) {
        import mir.math.common : sin;
        auto y = slice(x.sliced.map!sin).variable(x.requiresGrad);
        this.hx = x; // TODO if train
        return y;
    }

    auto backward(Variable!(T, dim, HostStorage) gy) {
        import mir.math.common : cos;
        auto gx = gy.dup;
        gx.sliced[] *= this.hx.sliced.map!cos;
        return gx;
    }

    version (grain_cuda) {
        Variable!(T, dim, DeviceStorage) dx;

        auto forward(Variable!(T, dim, DeviceStorage) x) {
            import grain.kernel : sin;
            auto y = x.dup;
            unaryFunc!sin(y);
            this.dx = x;
            return y;
        }

        auto backward(Variable!(T, dim, DeviceStorage) gy) {
            import grain.cudnn;
            import grain.kernel : cos;
            auto gx = this.dx.dup;
            unaryFunc!cos(gx);
            tensorOp!CUDNN_OP_TENSOR_MUL(gx, gx, gy);
            return gx;
        }
    }
}

///
unittest {
    import grain.testing;
    import std.typecons;
    import numir;
    import mir.ndslice;
    import mir.math : sin;

    Sin!(float, 2) hfunc;
    auto hx = uniform!float(2, 3).slice.variable;
    auto hy = hfunc.forward(hx);
    auto hgy = uniform!float(2, 3).slice.variable;
    auto hgx = hfunc.backward(hgy);
    gradCheck(hfunc, hx, hgy);
    assert(approxEqual(hy.sliced, hx.sliced.map!sin));

    version (grain_cuda) {
        Sin!(float, 2) dfunc;
        auto dy = dfunc.forward(hx.to!DeviceStorage);
        assert(approxEqual(dy.to!HostStorage.sliced, hy.sliced));
        auto dgx = dfunc.backward(hgy.to!DeviceStorage);
        assert(approxEqual(dgx.to!HostStorage.sliced, hgx.sliced));
    }
}


/// y = cos x
struct Cos(T, size_t dim) {
    import mir.ndslice : slice, map;

    mixin FunctionCommon;

    Variable!(T, dim, HostStorage) hx;

    auto forward(Variable!(T, dim, HostStorage) x) {
        import mir.math.common : cos;
        auto y = slice(x.sliced.map!cos).variable(x.requiresGrad);
        this.hx = x; // TODO if train
        return y;
    }

    auto backward(Variable!(T, dim, HostStorage) gy) {
        import mir.math.common : sin;
        auto gx = gy.dup;
        gx.sliced[] *= -this.hx.sliced.map!sin;
        return gx;
    }

    version (grain_cuda) {
        Variable!(T, dim, DeviceStorage) dx;

        auto forward(Variable!(T, dim, DeviceStorage) x) {
            import grain.kernel : cos;
            auto y = x.dup;
            unaryFunc!cos(y);
            this.dx = x;
            return y;
        }

        auto backward(Variable!(T, dim, DeviceStorage) gy) {
            import grain.cudnn;
            import grain.kernel : sin;
            auto gx = this.dx.dup;
            unaryFunc!sin(gx);
            tensorOp!CUDNN_OP_TENSOR_MUL(gx, gx, gy, -1);
            return gx;
        }
    }
}

///
unittest {
    import grain.testing;
    import std.typecons;
    import numir;
    import mir.ndslice;
    import mir.math : cos;

    Cos!(float, 2) hfunc;
    auto hx = uniform!float(2, 3).slice.variable;
    auto hy = hfunc.forward(hx);
    auto hgy = uniform!float(2, 3).slice.variable;
    auto hgx = hfunc.backward(hgy);
    gradCheck(hfunc, hx, hgy, 1e-3, 1e-3, 1e-3);
    assert(approxEqual(hy.sliced, hx.sliced.map!cos));

    version (grain_cuda) {
        Cos!(float, 2) dfunc;
        auto dy = dfunc.forward(hx.to!DeviceStorage);
        assert(approxEqual(dy.to!HostStorage.sliced, hy.sliced));
        auto dgx = dfunc.backward(hgy.to!DeviceStorage);
        assert(approxEqual(dgx.to!HostStorage.sliced, hgx.sliced));
    }
}

/// y = tan x
struct Tan(T, size_t dim) {
    import mir.ndslice : slice, map, as;

    mixin FunctionCommon;

    Variable!(T, dim, HostStorage) hx;

    auto forward(Variable!(T, dim, HostStorage) x) {
        import std.math : tan;
        auto y = slice(x.sliced.map!tan.as!T).variable(x.requiresGrad);
        this.hx = x; // TODO if train
        return y;
    }

    auto backward(Variable!(T, dim, HostStorage) gy) {
        import mir.math.common : cos;
        auto gx = gy.dup;
        auto c = this.hx.sliced.map!cos.map!"a * a";
        gx.sliced[] /= c;
        return gx;
    }

    version (grain_cuda) {
        Variable!(T, dim, DeviceStorage) dx;

        auto forward(Variable!(T, dim, DeviceStorage) x) {
            import grain.kernel : tan;
            auto y = x.dup;
            unaryFunc!tan(y);
            this.dx = x;
            return y;
        }

        auto backward(Variable!(T, dim, DeviceStorage) gy) {
            import grain.cudnn;
            import grain.kernel : cos;
            auto gx = this.dx.dup;
            unaryFunc!cos(gx);
            tensorOp!CUDNN_OP_TENSOR_MUL(gx, gx, gx); // cos^2 x
            return gy / gx;
        }
    }
}

///
unittest {
    import grain.testing;
    import std.typecons;
    import numir;
    import mir.ndslice;
    import std.math : tan;

    Tan!(float, 2) hfunc;
    auto hx = uniform!float(2, 3).slice.variable;
    auto hy = hfunc.forward(hx);
    auto hgy = uniform!float(2, 3).slice.variable;
    auto hgx = hfunc.backward(hgy);
    gradCheck(hfunc, hx, hgy);
    assert(approxEqual(hy.sliced, hx.sliced.map!tan));

    version (grain_cuda) {
        Tan!(float, 2) dfunc;
        auto dy = dfunc.forward(hx.to!DeviceStorage);
        assert(approxEqual(dy.to!HostStorage.sliced, hy.sliced));
        auto dgx = dfunc.backward(hgy.to!DeviceStorage);
        assert(approxEqual(dgx.to!HostStorage.sliced, hgx.sliced));
    }
}


/// y = alpha * x
struct Scale(T, size_t dim) {
    import mir.ndslice : slice;

    mixin FunctionCommon;

    T alpha = 1.0;

    auto forward(Variable!(T, dim, HostStorage) x) {
        return slice(this.alpha * x.sliced).variable(x.requiresGrad);
    }

    auto backward(Variable!(T, dim, HostStorage) gy) {
        return slice(this.alpha * gy.sliced).variable(gy.requiresGrad);
    }

    version (grain_cuda) {
        import grain.cudnn : scale;

        auto forward(Variable!(T, dim, DeviceStorage) x) {
            auto y = x.dup;
            scale(y, this.alpha);
            return y;
        }

        auto backward(Variable!(T, dim, DeviceStorage) gy) {
            auto gx = gy.dup;
            scale(gx, this.alpha);
            return gx;
        }
    }
}


/// test scale in simple case, gradcheck and cpu/cuda equality
unittest {
    import grain.testing;
    import std.typecons;
    import numir;
    import mir.ndslice;

    // simple case: 2.0 * x
    auto e = [[-2.0f, 4.0f, 6.0f], [2.0f, 0.2f, 0.0f]].nparray;
    auto xs = [[-1.0f, 2.0f, 3.0f], [1.0f, 0.1f, 0.0f]].nparray;
    auto hfunc = Scale!(float, 2)(2f);
    auto _hx = xs.variable;
    auto _hy = hfunc.forward(_hx);
    assert(approxEqual(_hy.sliced, e));

    auto hx = uniform!float(2, 2).slice.variable;
    auto hy = hfunc.forward(hx);
    auto hgy = uniform!float(2, 2).slice.variable;
    auto hgx = hfunc.backward(hgy);
    gradCheck(hfunc, hx, hgy); // , 1e-3, 1e-3, 1e-3);

    version (grain_cuda) {
        auto dfunc = Scale!(float, 2)(2f);
        auto dy = dfunc.forward(hx.to!DeviceStorage);
        assert(approxEqual(dy.to!HostStorage.sliced, hy.sliced));
        auto dgx = dfunc.backward(hgy.to!DeviceStorage);
        assert(approxEqual(dgx.to!HostStorage.sliced, hgx.sliced));
    }
}


/// y = -x
struct Neg(T, size_t dim) {
    import mir.ndslice : slice;

    mixin FunctionCommon;

    auto forward(Variable!(T, dim, HostStorage) x) {
        return slice(-x.sliced).variable(x.requiresGrad);
    }

    auto backward(Variable!(T, dim, HostStorage) gy) {
        return slice(-gy.sliced).variable(gy.requiresGrad);
    }

    version (grain_cuda) {
        import grain.kernel : neg;

        auto forward(Variable!(T, dim, DeviceStorage) x) {
            auto y = x.dup;
            unaryFunc!neg(y);
            return y;
        }

        auto backward(Variable!(T, dim, DeviceStorage) gy) {
            auto gx = gy.dup;
            unaryFunc!neg(gx);
            return gx;
        }
    }
}


/// test neg simple case, gradcheck and cpu/cuda equality
unittest {
    import grain.testing;
    import std.typecons;
    import numir;
    import mir.ndslice;

    // simple case: 2.0 * x
    auto xs = [[-1.0f, 2.0f, 3.0f], [1.0f, 0.1f, 0.0f]].nparray;
    auto hfunc = Neg!(float, 2)();
    auto _hx = xs.variable;
    auto _hy = hfunc.forward(_hx);
    assert(approxEqual(_hy.sliced, -xs));

    auto hx = uniform!float(2, 2).slice.variable;
    auto hy = hfunc.forward(hx);
    auto hgy = uniform!float(2, 2).slice.variable;
    auto hgx = hfunc.backward(hgy);
    gradCheck(hfunc, hx, hgy); // , 1e-3, 1e-3, 1e-3);

    version (grain_cuda) {
        auto dfunc = Neg!(float, 2)();
        auto dy = dfunc.forward(hx.to!DeviceStorage);
        assert(approxEqual(dy.to!HostStorage.sliced, hy.sliced));
        auto dgx = dfunc.backward(hgy.to!DeviceStorage);
        assert(approxEqual(dgx.to!HostStorage.sliced, hgx.sliced));
    }
}


/// y = abs x
struct Abs(T, size_t dim) {
    import mir.ndslice : slice, map;

    mixin FunctionCommon;
    Variable!(T, dim, HostStorage) hx;

    auto forward(Variable!(T, dim, HostStorage) x) {
        import mir.math : fabs;
        this.hx = x; // if train
        return slice(x.sliced.map!fabs).variable(x.requiresGrad);
    }

    auto backward(Variable!(T, dim, HostStorage) gy) {
        auto gx = gy.dup;
        gx.sliced[] *= this.hx.sliced.map!(a => a == 0f ? 0f : (a > 0f ? 1f : -1f));
        return gx;
    }

    version (grain_cuda) {
        Variable!(T, dim, DeviceStorage) dx;

        auto forward(Variable!(T, dim, DeviceStorage) x) {
            import grain.kernel : abs;
            auto y = x.dup;
            unaryFunc!abs(y);
            this.dx = x; // if train
            return y;
        }

        auto backward(Variable!(T, dim, DeviceStorage) gy) {
            import grain.kernel : absGrad;
            auto gx = this.dx.dup;
            unaryFunc!absGrad(gx);
            return gy * gx;
        }
    }
}

/// test abs simple case, gradcheck and cpu/cuda equality
unittest {
    import grain.testing;
    import std.typecons;
    import numir;
    import mir.ndslice;

    auto xs = [[-1.0f, 2.0f, -3.0f], [1.0f, 0.0f, 0.0f]].nparray;
    auto ys = [[1.0f, 2.0f, 3.0f], [1.0f, 0.0f, 0.0f]].nparray;
    auto hfunc = Abs!(float, 2)();
    auto hx = xs.variable;
    auto hy = hfunc.forward(hx);
    assert(approxEqual(hy.sliced, ys));

    auto gxs = [[-0.1f, 0.2f, -0.3f], [0.5f, 0.0f, 0.0f]].nparray;
    auto gys = [[0.1f, 0.2f, 0.3f], [0.5f, 0.6f, 0.7f]].nparray;
    auto hgy = gys.variable;
    auto hgx = hfunc.backward(hgy);
    assert(approxEqual(hgx.sliced, gxs));

    version (grain_cuda) {
        auto dfunc = Abs!(float, 2)();
        auto dy = dfunc.forward(hx.to!DeviceStorage);
        assert(approxEqual(dy.to!HostStorage.sliced, hy.sliced));
        auto dgx = dfunc.backward(hgy.to!DeviceStorage);
        assert(approxEqual(dgx.to!HostStorage.sliced, hgx.sliced));
    }
}


/// y = pow x
struct Pow(T, size_t dim) {
    import mir.ndslice : slice, map;

    mixin FunctionCommon;

    T power;
    Variable!(T, dim, HostStorage) hx;

    this(T power) {
        this.power = power;
    }

    auto forward(Variable!(T, dim, HostStorage) x) {
        import mir.math.common : pow;
        auto y = slice(x.sliced.map!(a => pow(a, this.power))).variable(x.requiresGrad);
        this.hx = x; // TODO if train
        return y;
    }

    auto backward(Variable!(T, dim, HostStorage) gy) {
        import mir.math.common : pow;
        auto gx = gy.dup;
        gx.sliced[] *= this.hx.sliced.map!(a => this.power * pow(a, this.power - 1));
        return gx;
    }

    version (grain_cuda) {
        Variable!(T, dim, DeviceStorage) dx;

        auto forward(Variable!(T, dim, DeviceStorage) _x) {
            this.dx = _x;
            auto x = _x.dup;
            import grain.kernel : pow;
            auto shape = CuPtr!uint(x.shape[0..$]);
            auto strides = CuPtr!int(x.strides[0..$]);
            auto ndim = cast(uint) dim;
            auto len = cast(uint) x.data.length;

            Global.kernel!pow
                .call(this.power, x.data.ptr, len, ndim, shape.ptr, strides.ptr)
                .launch(len);
            return x;
        }

        auto backward(Variable!(T, dim, DeviceStorage) gy) {
            auto x = this.dx.dup;
            import grain.kernel : powGrad;
            auto shape = CuPtr!uint(x.shape[0..$]);
            auto strides = CuPtr!int(x.strides[0..$]);
            auto ndim = cast(uint) dim;
            auto len = cast(uint) x.data.length;

            Global.kernel!powGrad
                .call(this.power, x.data.ptr, len, ndim, shape.ptr, strides.ptr)
                .launch(len);
            return gy * x;
        }
    }
}

///
unittest {
    import grain.testing;
    import std.typecons;
    import numir;
    import mir.ndslice;
    import mir.math : pow;

    auto p = 2.0f;
    auto hfunc = Pow!(float, 2)(p);
    auto hx = uniform!float(2, 3).slice.variable;
    auto hy = hfunc.forward(hx);
    auto hgy = uniform!float(2, 3).slice.variable;
    auto hgx = hfunc.backward(hgy);
    gradCheck(hfunc, hx, hgy, 1e-3, 1e-3, 1e-3);
    assert(approxEqual(hy.sliced, hx.sliced.map!(a => pow(a, p))));

    version (grain_cuda) {
        auto dfunc = Pow!(float, 2)(p);
        auto dy = dfunc.forward(hx.to!DeviceStorage);
        assert(approxEqual(dy.to!HostStorage.sliced, hy.sliced));
        auto dgx = dfunc.backward(hgy.to!DeviceStorage);
        assert(approxEqual(dgx.to!HostStorage.sliced, hgx.sliced));
    }
}

/// reference implementaion of pooling function
struct PoolRefImpl(alias fun) {
    
}

/// max pooling function
struct MaxPool(T, size_t poolDims) {
    static assert(poolDims > 0);
    enum dim = poolDims + 2;
    int[poolDims] window, pad, stride;

    version (grain_cuda) {
        Variable!(T, dim, DeviceStorage) dx, dy;

        import grain.cudnn;
        auto forward(Variable!(T, dim, DeviceStorage) x) {
            auto y = grain.cudnn.poolForward(x, this.window, this.pad, this.stride);
            // TODO if train
            this.dx = x;
            this.dy = y;
            return y;
        }

        auto backward(Variable!(T, dim, DeviceStorage) gy) {
            auto gx = this.dx.uninit;
            grain.cudnn.poolBackward(gx, this.dx, gy, this.dy, this.window, this.pad, this.stride);
            return gx;
        }
    }
}


///
version (grain_cuda) unittest {
    import std.stdio;
    import mir.ndslice;
    import numir;
    auto f = MaxPool!(float, 2)([3, 3], [1, 1], [1, 1]);
    auto x = iota(2, 2, 1, 3).as!float.slice.variable;
    // [[[[0, 1, 2]], [[3, 4, 5]]],
    //  [[[6, 7, 8]], [[9, 10, 11]]]]
    auto y = f.forward(x.to!DeviceStorage);
    enum yex = [[[[1, 2, 2]], [[4, 5, 5]]], [[[7, 8, 8]], [[10, 11, 11]]]];
    assert(y.to!HostStorage.sliced == yex);
    auto gx = f.backward(y);
    enum gxex = [[[[0, 1, 4]], [[0, 4, 10]]], [[[0, 7, 16]], [[0, 10, 22]]]];
    assert(gx.to!HostStorage.sliced == gxex);
}
