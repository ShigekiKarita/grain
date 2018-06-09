/**
   A module for unary functions
 */
module grain.functions.unary;

import grain.autograd;
import grain.cuda;
import grain.utility;
import grain.functions.common;

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
                this.dy = y;
                activationForward!CUDNN_ACTIVATION_RELU(x, y);
            } else {
                import grain.kernel : relu;
                auto n = cast(uint) y.data.length; // FIXME use y.nElement
                Global.kernel!relu
                    .call(y.data.ptr, n).launch(n);
            }
            return y;
        }

        auto backward(Variable!(T, dim, DeviceStorage) gy) {
            auto gx = gy.dup; // TODO: create empty
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
