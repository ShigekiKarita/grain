module grain.functions;

import grain.autograd;
import grain.cuda;
import grain.utility : toTuple, fromTuple, castArray;

import std.stdio;

version (grain_cuda) {
    import cudnn = derelict.cudnn7;
}


mixin template TypeChecker(alias forward, alias backward) {
    static assert(allSatisfy!(isVariable, Parameters!forward),
                  "all the forward function args should be variable.");
    static assert(allSatisfy!(isVariable, Parameters!backward),
                  "all the backward function args should be variable.");
    static if (arity!forward == 1 && arity!backward == 1) {
        static assert(is(ReturnType!backward == Parameters!forward[0]));
        static assert(is(ReturnType!forward == Parameters!backward[0]));
    } else static if (arity!backward == 1) {
        static assert(is(ReturnType!backward == Tuple!(Parameters!forward)));
        static assert(is(ReturnType!forward == Parameters!backward[0]));
    } else static if (arity!forward == 1) {
        static assert(is(ReturnType!backward == Parameters!forward[0]));
        static assert(is(ReturnType!forward == Tuple!(Parameters!backward)));
    } else {
        static assert(is(ReturnType!backward == Tuple!(Parameters!forward)));
        static assert(is(ReturnType!forward == Tuple!(Parameters!backward)));
    }
}



enum bool isFunction(T) = {
    import std.meta : allSatisfy;
    import std.typecons : isTuple, tuple, Tuple, RefCounted;
    import std.traits : arity, Parameters, ReturnType;
    static foreach (i, forward; __traits(getOverloads, T, "forward")) {
        static foreach (i, backward; __traits(getOverloads, T, "backward")) {
            static if (!allSatisfy!(isHost, Parameters!forward) &&
                       !allSatisfy!(isHost, Parameters!backward)) {
                mixin TypeChecker!(forward, backward);
            }
            static if (allSatisfy!(isHost, Parameters!forward) &&
                       allSatisfy!(isHost, Parameters!backward)) {
                mixin TypeChecker!(forward, backward);
            }
        }
    }
    return true;
        }();

mixin template FunctionCommon() {
    import std.meta : allSatisfy;
    import std.typecons : isTuple, tuple, Tuple, RefCounted;
    import std.traits : arity, Parameters, ReturnType;

    @disable this(this); // no copyable

    static foreach (i, forward; __traits(getOverloads, typeof(this), "forward")) {
        static foreach (i, backward; __traits(getOverloads, typeof(this), "backward")) {
            static if (!allSatisfy!(isHost, Parameters!forward) &&
                       !allSatisfy!(isHost, Parameters!backward)) {
                alias DeviceRets = Tuple!(Parameters!backward);
                alias DeviceArgs = Tuple!(Parameters!forward);
                mixin TypeChecker!(forward, backward);
            }
            static if (allSatisfy!(isHost, Parameters!forward) &&
                       allSatisfy!(isHost, Parameters!backward)) {
                alias HostRets = Tuple!(Parameters!backward);
                alias HostArgs = Tuple!(Parameters!forward);
                mixin TypeChecker!(forward, backward);
            }
        }
    }
    static assert(isFunction!(typeof(this)));

    auto applyForward(Args...)(Args args) {
        import std.algorithm : each;
        RefCounted!(UntypedVariable[]) uargs;
        uargs.length = args.length;
        foreach (i, a; args) {
            uargs[i] = UntypedVariable(a);
            uargs[i].bprop = a.bprop; // pass the chain to backprop
        }
        auto rets = this.forward(args).toTuple;
        enum isHost = allSatisfy!(isHost, Args);
        foreach (i, r; rets) {
            auto u = UntypedVariable(r);
            if (grain.autograd.backprop) {
                // RefCounted!
                BackProp bp = BackProp(&this.applyBackward!isHost,
                                       uargs);
                bp.gradOutputs.length = rets.length;
                u.bprop = bp;
                u.outPosition = i;
                rets[i].bprop = bp;
            }
        }
        static if (rets.length > 1) {
            return rets;
        } else {
            return rets[0];
        }
    }

    void applyBackward(bool isHost)(UntypedVariable[] ugradOutputs, UntypedVariable[] uinputs) {
        static if (isHost) {
            HostRets vgradOutputs;
        } else {
            DeviceRets vgradOutputs;
        }
        static foreach (i; 0 .. vgradOutputs.length) {
            vgradOutputs[i] = ugradOutputs[i].to!(typeof(vgradOutputs[i]));
        }
        auto vgradInputs = this.backward(vgradOutputs.expand).toTuple;
        assert(vgradInputs.length == uinputs.length, "invalid number of input gradients");
        UntypedVariable[vgradInputs.length] ugradInputs; // TODO use refcounted?
        foreach (i, v; vgradInputs) {
            ugradInputs[i] = UntypedVariable(v);
        }

        foreach (i, vgi; vgradInputs) {
            // TODO reconsider this condition
            if (uinputs[i].requiresGrad) {
                alias Storage = typeof(vgradInputs[i].data);
                alias V = typeof(vgradInputs[i]);
                auto data = uinputs[i].grad.get!Storage;
                static if (vgradInputs[i].isHost) {
                    import mir.ndslice.slice : sliced;
                    auto shape = vgradInputs[i].shape.castArray!size_t;
                    data[] += vgradInputs[i].data[]; // .sliced(shape); FIXME use shape
                } else version (grain_cuda) {
                    import std.traits : isFloatingPoint;
                    // TODO support integral types
                    static if (isFloatingPoint!(ElementType!V)) {
                        axpy(vgradInputs[i].data, data);
                    }
                }
            }
            uinputs[i].bprop.backward(&ugradInputs[i], uinputs[i].outPosition);
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


/// test NG functions
unittest {
    alias F1H = Variable!(float, 1, HostStorage);
    version (grain_cuda) alias F1D = Variable!(float, 1, HostStorage);
    struct A(DelayInstantiation) {
        mixin FunctionCommon;
        // mismatch of args
        F1H forward(F1H x) { return x; };
        F1H backward(F1H x, F1H y) { return x; };
    }
    static assert(!__traits(compiles, A!void));

    version (grain_cuda) {
        struct B(DelayInstantiation) {
            mixin FunctionCommon;
            F1H forward(F1H x) { return x; };
            F1H backward(F1H x) { return x; };
            // mismatch of args in device
            version (grain_cuda) {
                F1D forward(F1D x) { return x; };
                F1D backward(F1D x, F1D y) { return x; };
            }
        }
        static assert(!__traits(compiles, B!void));
    }
}

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

// forward 2-in 1-out function
unittest {
    import std.typecons;
    import numir;
    import mir.ndslice;
    grain.autograd.backprop = true;
    scope (exit) grain.autograd.backprop = false;
    {
        auto func = new MatMul!float;
        auto a = uniform!float(3, 4).slice.variable(true);
        auto b = uniform!float(4, 2).slice.variable(true);
        auto c = func.applyForward(a, b);
        auto gc = uniform!float(3, 2).slice.variable;
        auto ugc = UntypedVariable(gc);
        c.backward(&ugc);

        auto gab = func.backward(gc);
        assert(a.gradSlice == gab[0].sliced);
        assert(b.gradSlice == gab[1].sliced);
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

/++
 Matrix-Matrix multiplication

 See_Also: https://github.com/chainer/chainer/blob/v1/chainer/functions/connection/linear.py#L11
 +/
struct MatMul(T) {
    import mir.ndslice : transposed, universal;
    import std.typecons : tuple;
    import lubeck : mtimes;
    T alpha = 1;
    T beta = 0;
    Variable!(T, 2, HostStorage) ha, hb;

    // TODO uncomment this line
    mixin FunctionCommon;

    auto forward(Variable!(T, 2, HostStorage) a, Variable!(T, 2, HostStorage) b) {
        // TODO if training
        this.ha = a;
        this.hb = b;
        return mtimes(a.sliced, b.sliced).variable(a.requiresGrad || b.requiresGrad);
    }

    auto backward(Variable!(T, 2, HostStorage) gy) {
        auto ga = mtimes(gy.sliced, this.hb.sliced.transposed).variable;
        auto gb = mtimes(this.ha.sliced.transposed, gy.sliced).variable;
        return tuple(ga, gb);
    }

    version(grain_cuda) {
        Variable!(T, 2, DeviceStorage) da, db;

        auto forward(Variable!(T, 2, DeviceStorage) a, Variable!(T, 2, DeviceStorage) b) {
            import grain.cublas;
            static if (is(T == float)) {
                alias gemm = cublasSgemm_v2;
            } else static if (is(T == double)) {
                alias gemm = cublasDgemm_v2;
            } else {
                static assert(false, "unsupported type");
            }

            import std.typecons : RefCounted;
            assert(a.shape[1] == b.shape[0]);
            auto cdata = RefCounted!(CuPtr!T)(a.shape[0] * b.shape[1]);
            auto c = Variable!(T, 2, DeviceStorage)(
                a.requiresGrad || b.requiresGrad, [a.shape[0], b.shape[1]], [b.shape[1], 1], cdata);
            // C = A x B = (BT x AT)T
            // TODO support transposed (CUBLAS_OP_T)
            // see https://github.com/libmir/mir-blas/blob/master/source/mir/blas.d#L299
            // TODO if train
            this.da = a;
            this.db = b;
            checkCublasErrors(gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                   cast(int) b.shape[1],
                                   cast(int) a.shape[0], cast(int) a.shape[1],
                                   &alpha,
                                   cast(const T*) b.data.ptr, cast(int) b.strides[0],
                                   cast(const T*) a.data.ptr, cast(int) a.strides[0],
                                   &beta,
                                   cast(T*) c.data.ptr, cast(int) c.strides[0]));
            return c;
        }

        auto backward(Variable!(T, 2, DeviceStorage) gc) {
            import grain.cublas;
            static if (is(T == float)) {
                alias gemm = cublasSgemm_v2;
            } else static if (is(T == double)) {
                alias gemm = cublasDgemm_v2;
            } else {
                static assert(false, "unsupported type");
            }
            auto ga = this.da.dup;
            auto gb = this.db.dup;
            // auto ga = mtimes(gc.sliced, this.hb.sliced.transposed).variable;
            checkCublasErrors(gemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                   cast(int) ga.shape[1],
                                   cast(int) ga.shape[0], cast(int) gc.shape[1],
                                   &alpha,
                                   cast(const T*) db.data.ptr, cast(int) db.strides[0],
                                   cast(const T*) gc.data.ptr, cast(int) gc.strides[0],
                                   &beta,
                                   cast(T*) ga.data.ptr, cast(int) ga.strides[0]));
            // auto gb = mtimes(this.ha.sliced.transposed, gc.sliced).variable;
            checkCublasErrors(gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                                   cast(int) gb.shape[1],
                                   cast(int) gb.shape[0], cast(int) da.shape[0],
                                   &alpha,
                                   cast(const T*) gc.data.ptr, cast(int) gc.strides[0],
                                   cast(const T*) da.data.ptr, cast(int) da.strides[0],
                                   &beta,
                                   cast(T*) gb.data.ptr, cast(int) gb.strides[0]));
            return tuple(ga, gb);
        }
    }
}

/// test matmul gradcheck and cpu/cuda equality
unittest {
    foreach (i; [2, 3, 4]) {
        foreach (j; [2, 3, 4]) {
            import std.typecons : tuple;
            import numir : uniform;
            import mir.ndslice : slice;
            import grain.testing;

            auto k = 3;
            auto a = uniform!float(i, k).slice.variable;
            auto b = uniform!float(k, j).slice.variable;
            auto gc = uniform!float(i, j).slice.variable;
            MatMul!float func;
            gradCheck(func, tuple(a, b), gc, 1e-3, 1e-3, 1e-3);

            version (grain_cuda) {
                import numir.testing;
                MatMul!float func2;
                auto hc = func.forward(a, b);
                auto dc = func2.forward(a.to!DeviceStorage, b.to!DeviceStorage);
                assert(approxEqual(dc.to!HostStorage.sliced, hc.sliced));
                auto hgab = func.backward(gc);
                auto dgab = func2.backward(gc.to!DeviceStorage);
                // writefln!"%s vs %s"(dgab[0].to!HostStorage.sliced, hgab[0].sliced);
                assert(approxEqual(dgab[0].to!HostStorage.sliced, hgab[0].sliced));
                assert(approxEqual(dgab[1].to!HostStorage.sliced, hgab[1].sliced));
            }
        }
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


struct NegativeLogLikelihood(F, I=long) {
    /++
    Compute negative log-likelihood: -logP(y=t)
    Params:
      logP: log softmax output as prediction. shape: (nBatch, nClass)
      targetId: target integer id of class. shape: (nBatch)
      +/

    mixin FunctionCommon;

    bool sizeAverage = true;
    int ignoreIndex = -100;
    // TODO: bool reduce = true;

    // cache for backward
    Variable!(I, 1, HostStorage) _htargetId;
    F _normalize;
    int _nClass;

    auto forward(Variable!(F, 2, HostStorage) logP, Variable!(I, 1, HostStorage) targetId) {
        import mir.math;
        import mir.ndslice;
        F result = 0.0;
        size_t count = 0;
        foreach (i; 0 .. targetId.sliced.length) {
            auto t = targetId.sliced[i];
            if (t != this.ignoreIndex) {
                result -= logP.sliced[i, t];
                ++count;
            }
        }
        if (this.sizeAverage && count > 0) {
            result /= count;
        }
        // TODO if train
        this._nClass = logP.shape[1];
        this._htargetId = targetId;
        this._normalize = this.sizeAverage && count > 0 ? 1.0 / count : 1.0;
        return result.variable;
    }

    auto backward(Variable!(F, 0, HostStorage) gy) {
        import std.typecons;
        import mir.math;
        import mir.ndslice;
        import numir;

        auto nBatch = this._htargetId.shape[0];
        auto glogP = zeros!F(nBatch, this._nClass);
        auto coeff = gy.data[0] * this._normalize;
        foreach (i; 0 .. nBatch) {
            auto t = this._htargetId.sliced[i];
            if (t != this.ignoreIndex) {
                glogP[i][t] = -coeff;
            }
        }
        return tuple(glogP.variable, typeof(this._htargetId)());
    }

    version (grain_cuda) {
        Variable!(I, 1, DeviceStorage) _dtargetId;
        auto forward(Variable!(F, 2, DeviceStorage) logP, Variable!(I, 1, DeviceStorage) targetId) {
            static assert(is(F == float), "only float is supported now");
            static assert(is(I == int), "only int is supported now");

            import grain.kernel : nll;
            this._nClass = logP.shape[1];
            auto dresult = CuPtr!F([0]); // [result].variable.to!DeviceStorage; <- FIXME
            auto dcount = CuPtr!int([0]); // [count].variable.to!DeviceStorage;

            auto batchSize = targetId.shape[0];
            Global.kernel!nll
                .call(dresult.ptr, dcount.ptr, logP.data.ptr,
                      targetId.data.ptr, this.ignoreIndex, batchSize, this._nClass).launch(batchSize);

            F result = 0.0;
            int count = 0;
            dresult.toHost(&result);
            dcount.toHost(&count);

            if (this.sizeAverage && count > 0) {
                result /= count;
            }
            // TODO if train
            this._nClass = logP.shape[1];
            this._dtargetId = targetId;
            this._normalize = this.sizeAverage && count > 0 ? 1.0 / count : 1.0;
            return result.variable.to!DeviceStorage;
        }

        auto backward(Variable!(F, 0, DeviceStorage) gy) {
            static assert(is(F == float), "only float is supported now");
            static assert(is(I == int), "only int is supported now");

            import grain.kernel;
            import std.typecons : tuple, RefCounted;
            auto nBatch = this._dtargetId.shape[0];
            RefCounted!(CuPtr!F) glogP = CuPtr!F(nBatch * this._nClass);
            glogP.zero_();
            auto coeff = gy.to!HostStorage.data[0] * this._normalize;
            Global.kernel!nllGrad
                .call(glogP.ptr, -coeff, this._dtargetId.data.ptr, this.ignoreIndex, nBatch, this._nClass).launch(nBatch);
            auto v = Variable!(F, 2, DeviceStorage)(false, [nBatch, this._nClass], [this._nClass, 1], glogP);
            return tuple(v, typeof(this._dtargetId)());
        }

    }
}

/// test nll simple case, gradcheck and cpu/cuda equality
unittest {
    /++ equivalent torch v0.4 code
     x = torch.FloatTensor([[0.2, 0.4, 0.4], [0.1,0.5,0.4]])
     x.requires_grad = True
     t = torch.LongTensor([1, 0])
     l = torch.nn.functional.nll_loss(x, t)
     print(l)       # tensor(-0.2500)
     l.backward()
     print(x.grad)  # tensor([[0.0, -0.5, 0.0], [-0.5, 0.0, 0.0]])
     +/
    import std.typecons;
    import grain.testing;
    NegativeLogLikelihood!(float, int) func;
    auto hx = [[0.2f, 0.4f, 0.4f], [0.1f, 0.5f, 0.4f], [0.1f, 0.5f, 0.4f]].variable;
    auto ht = [1, 0, func.ignoreIndex].variable;
    auto hl = func.forward(hx, ht);
    assert(func._normalize == 0.5);
    assert(hl.sliced == [-(0.4f + 0.1f + 0.0f) / 2]);
    auto hgx = func.backward(1.0f.variable);
    assert(hgx[0].sliced == [[0.0, -0.5, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    assert(!hgx[1].defined);
    gradCheck(func, tuple(hx, ht), 1.0f.variable);

    version (grain_cuda) {
        auto dx = hx.to!DeviceStorage;
        auto dt = ht.to!DeviceStorage;
        auto dl = func.forward(dx, dt);
        assert(func._normalize == 0.5);
        assert(dl.to!HostStorage.sliced == [-(0.4f + 0.1f + 0.0f) / 2]);
        auto dgx = func.backward(1.0f.variable.to!DeviceStorage);
        assert(dgx[0].to!HostStorage.sliced == [[0.0, -0.5, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]);
        assert(!dgx[1].defined);
    }
}


auto broadcastable(T, size_t dim, alias Storage)(Variable!(T, dim, Storage) a, Variable!(T, dim, Storage) b) {
    int[dim] resultShape;
    bool ok = false;
    foreach (i; 0 .. dim) {
        ok = a.shape[i] == b.shape[i] || a.shape[i] == 1 || b.shape[i] == 1;
        if (ok) {
            resultShape[i] = max(a.shape[i], b.shape[i]);
        } else break;
    }
    return tuple!("ok", "shape")(ok, resultShape);
}


/// TODO generalize to broadcastable addition
struct AddBias(T) {
    mixin FunctionCommon;

    import mir.ndslice : map, slice;
    import std.typecons : tuple, RefCounted;
    auto forward(Variable!(T, 2, HostStorage) a, Variable!(T, 1, HostStorage) b) {
        assert(a.shape[1] == b.shape[0]);
        auto ret = a.dup;
        foreach (i; 0 .. a.shape[0]) {
            ret.sliced[i][] += b.sliced;
        }
        return ret;
    }

    auto backward(Variable!(T, 2, HostStorage) gy) {
        import numir : alongDim;
        import mir.math : sum;
        auto gb = gy.sliced.alongDim!0.map!sum.slice.variable;
        return tuple(gy, gb);
    }

    version (grain_cuda) {
        import grain.kernel : addBias, addBiasGrad;

        auto forward(Variable!(T, 2, DeviceStorage) a, Variable!(T, 1, DeviceStorage) b) {
            assert(a.shape[1] == b.shape[0]);
            auto y = a.dup;
            auto n = cast(uint) y.data.length;
            auto blen = cast(uint) b.data.length;
            Global.kernel!addBias
                .call(y.data.ptr, b.data.ptr, blen, n).launch(n);
            return y;
        }

        auto backward(Variable!(T, 2, DeviceStorage) gy) {
            RefCounted!(CuPtr!T) gb = CuPtr!T(gy.shape[1]);
            gb.zero_();
            auto n = cast(uint) gy.data.length;
            auto blen = cast(uint) gb.length;
            Global.kernel!addBiasGrad
                .call(gy.data.ptr, gb.ptr, blen, n).launch(n);
            return tuple(gy, Variable!(T, 1, DeviceStorage)(false, [cast(int) blen], [1], gb));
        }
    }
}


unittest {
    import std.typecons;
    import grain.testing;
    import numir;
    import mir.ndslice;

    AddBias!float func;
    auto hx = [[0f, 1f], [2f, 3f], [4f, 5f]].variable; // 3x2
    auto hb = [-1f, 1f].variable; // 2
    auto hy = func.forward(hx, hb);
    assert(hy.sliced == [[-1f, 2f], [1f, 4f], [3f, 6f]]);

    auto hgy = uniform!float(hy.shape.castArray!size_t).slice.variable;
    auto hgxb = func.backward(hgy);
    assert(hgxb[0].sliced == hgy.sliced);
    assert(hgxb[1].sliced == [hgy.sliced[0, 0] + hgy.sliced[1, 0] + hgy.sliced[2, 0],
                              hgy.sliced[0, 1] + hgy.sliced[1, 1] + hgy.sliced[2, 1]]);
    gradCheck(func, tuple(hx, hb), hgy);

    version (grain_cuda) {
        auto dx = hx.to!DeviceStorage;
        auto db = hb.to!DeviceStorage;
        auto dy = func.forward(dx, db);
        assert(dy.to!HostStorage.sliced == [[-1f, 2f], [1f, 4f], [3f, 6f]]);
        auto dgy = hgy.to!DeviceStorage;
        auto dgxb = func.backward(dgy);
        assert(dgxb[0].to!HostStorage.sliced == hgxb[0].sliced);
        assert(dgxb[1].to!HostStorage.sliced == hgxb[1].sliced);
    }
}



/// test variable.backward
unittest {
    import std.typecons;
    import grain.testing;
    import mir.ndslice;

    grain.autograd.backprop = true;

    NegativeLogLikelihood!(float, int) func;
    auto hx = [[0.2f, 0.4f, 0.4f], [0.1f, 0.5f, 0.4f], [0.1f, 0.5f, 0.4f]].variable;
    hx.requiresGrad = true;
    auto ht = [1, 0, func.ignoreIndex].variable;
    auto hl = func.applyForward(hx, ht);
    // hl.bprop.writeln;
    assert(func._normalize == 0.5);
    assert(hl.sliced == [-(0.4f + 0.1f + 0.0f) / 2]);
    auto u = UntypedVariable(1.0f.variable);
    hl.backward(&u);
    // hl.bprop.inputs[0].writeln;
    assert(hx.grad[].sliced(3, 3) == [[0.0, -0.5, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    // assert(!hgx[1].defined);
}
