module grain.functions;

import grain.autograd;
import grain.cuda;

import std.stdio;

version (grain_cuda) {
    import cudnn = derelict.cudnn7;
}


mixin template TypeChecker(alias forward, alias backward) {
    static assert(allSatisfy!(isVariable, Parameters!forward),
                  "all the forward function args should be variable.");
    static assert(allSatisfy!(isVariable, Parameters!backward),
                  "all the backward function args should be variable.");
    static assert(arity!forward == Tuple!(ReturnType!backward).length);
    static assert(arity!backward == Tuple!(ReturnType!forward).length);
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

mixin template FunctionCommon() {
    import std.meta : allSatisfy;
    import std.typecons : isTuple, tuple, Tuple, RefCounted;
    import std.traits : arity, Parameters, ReturnType;

    RefCounted!(UntypedVariable[]) vargs,  vrets;

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

    auto applyForward(Args...)(Args args) {
        import std.algorithm : each;
        this.vargs.length = args.length;
        this.vrets.length = HostRets.length; // TODO: remove this line
        foreach (i, a; args) {
            this.vargs[i] = UntypedVariable(a);
        }
        auto ret = this.forward(args);
        static if (isTuple!(typeof(ret))) {
            auto rets = ret;
        } else {
            auto rets = tuple(ret);
        }

        enum isHost = allSatisfy!(isHost, Args);
        auto ugradOuts = new UntypedVariable[rets.length];
        foreach (i, r; rets) {
            auto u = UntypedVariable(r);
            if (grain.autograd.backprop) {
                RefCounted!BackProp bp = BackProp(&this.applyBackward!isHost,
                                                  this.vargs, ugradOuts);
                u.bprop = bp;
                u.outPosition = i;
                rets[i].bprop = bp;
            }
            this.vrets[i] = u;
        }
        static if (isTuple!(typeof(ret))) {
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
        static if (vgradOutputs.length == 1) {
            auto _vgradInputs = this.backward(vgradOutputs[0]);
        } else {
            auto _vgradInputs = vgradOutputs.apply!(this.backward);
        }
        static if (isTuple!(typeof(_vgradInputs))) {
            auto vgradInputs = _vgradInputs;
        } else {
            auto vgradInputs = tuple(_vgradInputs);
        }
        auto ugradInputs = new UntypedVariable[vgradInputs.length];
        foreach (i, v; vgradInputs) {
            ugradInputs[i] = UntypedVariable(v);
        }
        assert(vgradInputs.length == uinputs.length, "invalid number of input gradients");
        foreach (i, vgi; vgradInputs) {
            if (uinputs[i].requiresGrad) {
                alias Storage = typeof(vgradInputs[i].data);
                alias V = typeof(vgradInputs[i]);
                static if (vgradInputs[i].isHost) {
                    import mir.ndslice.slice : sliced;
                    auto gs = uinputs[i].gradSlice!V;
                    gs[] += vgradInputs[i].data[].sliced(gs.shape);
                } else {
                    auto data = uinputs[i].grad.get!Storage;
                    axpy(vgradInputs[i].data, data);
                }
            }
            uinputs[i].backward(&ugradInputs[i]);
        }
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
                    .call(y.data.ptr, n).launch([1,1,1], [n,1,1]);
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
                    .call(gx.data.ptr, gy.data.ptr, this.dx.data.ptr, n).launch([1,1,1], [n,1,1]);
            }
            return gx;
        }
    }
}

unittest {
    import std.typecons;
    grain.autograd.backprop = true;
    scope (exit) grain.autograd.backprop = false;
    {
        auto func = new ReLU!(float, 1);
        auto x = [-1.0f, 2.0f, 3.0f].variable(true);
        auto y = func.applyForward(x);
        auto gy = [1.0f, 2.0f, 3.0f].variable;
        auto ugy = UntypedVariable(gy);
        y.backward(&ugy);
        assert(x.grad == [0, 2, 3]);

        auto func2 = new ReLU!(float, 1);
        auto y2 = func.applyForward(x);
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


///
unittest {
    import grain.testing : gradCheck;
    foreach (inplace; [false]) {
        auto func = new ReLU!(float, 1);
        func.inplace = inplace;

        // test CPU
        {
            auto x = [-1.0f, 1.0f, 0.0f].variable;
            // gradCheck(func, x, [0.1f, 0.1f, 0.1f].variable);

            auto y = func.forward(x);
            assert(x.data == (inplace ? y.data : [-1.0f, 1.0f, 0.0f]));
            assert(y.data[0] == 0.0);
            assert(y.data[1] == 1.0);
            assert(y.data[2] == 0.0);
            // Why fail?
            // assert(y.data == [0.0f, 1.0f, 0.0f]);

            // x = [-1.0f, 1.0f, 0.0f].variable;
            // writeln(func.hx);
            auto gy = [1.0f, 2.0f, 3.0f].variable;
            auto gx = func.backward(gy);
            assert(gx.data[0] == 0.0);
            assert(gx.data[1] == 2.0);
            assert(gx.data[2] == 3.0);
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
            import std.format;
            assert(gx.data == [0.0, 2.0, 0.0], format!"%s"(gx.data[]));
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
    // mixin FunctionCommon;

    auto forward(Variable!(T, 2, HostStorage) a, Variable!(T, 2, HostStorage) b) {
        // TODO if training
        this.ha = a;
        this.hb = b;
        return mtimes(a.sliced, b.sliced).variable(a.requiresGrad || b.requiresGrad);
    }

    auto backward(Variable!(T, 2, HostStorage) gy) {
        auto ga = mtimes(gy.sliced, this.hb.sliced.transposed).variable;
        auto gb = mtimes(gy.sliced.transposed, this.ha.sliced).variable;
        return tuple(ga, gb);
    }

    version(grain_cuda) {
        auto forward(Variable!(T, 2, DeviceStorage) a, Variable!(T, 2, DeviceStorage) b) {
            import std.typecons : RefCounted;
            import grain.cublas;
            assert(a.shape[1] == b.shape[0]);
            static if (is(T == float)) {
                alias gemm = cublasSgemm_v2;
            } else static if (is(T == double)) {
                alias gemm = cublasDgemm_v2;
            } else {
                static assert(false, "unsupported type");
            }

            auto cdata = RefCounted!(CuPtr!T)(a.shape[0] * b.shape[1]);
            auto c = Variable!(T, 2, DeviceStorage)(
                false, [a.shape[0], b.shape[1]], [b.shape[1], 1], cdata);
            // import numir;
            // auto c = empty!float(a.shape[0], b.shape[1]).variable.to!DeviceStorage;

            // C = A x B = (BT x AT)T
            // TODO support transposed (CUBLAS_OP_T)
            // see https://github.com/libmir/mir-blas/blob/master/source/mir/blas.d#L299
            // FIXME:
            // see https://peterwittek.com/cublas-matrix-c-style.html
            // see https://www.beechwood.eu/cublas-sgemm-dimensions-mishmash-in-row-major-order-environment/
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
    }
}

/// test (3x2) x (2x3)
unittest {
    import std.stdio;
    import mir.ndslice;

    auto a = [[1f, 3f],
              [5f, 7f],
              [9f, 11f]].variable;
    auto b = [[2f, 4f, 6f],
              [8f, 10f, 12f]].variable;
    auto expected = [[1*2+3*8, 1*4+3*10, 1*6+3*12],
                     [5*2+7*8, 5*4+7*10, 5*6+7*12],
                     [9*2+11*8, 9*4+11*10, 9*6+11*12]];

    version(grain_cuda) {{
        import numir;
        import grain.cublas;
        auto ad = a.to!DeviceStorage.data;
        auto bd = b.to!DeviceStorage.data;
        auto z = zeros!float(3, 3).variable;
        auto c = z.to!DeviceStorage;

        float alpha = 1.0, beta = 0.0;
        checkCublasErrors(cublasSgemm_v2(
                              cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                              3, // cast(int) b.shape[1],
                              3, // cast(int) a.shape[0],c.
                              2, // cast(int) a.shape[1],
                              &alpha,
                              cast(float*) bd.ptr, cast(int) 3,
                              cast(float*) ad.ptr, cast(int) 2,
                              &beta,
                              cast(float*) c.data.ptr, cast(int) 3));
        auto cdata = c.to!HostStorage.sliced;
        assert(cdata == expected);
    }}
}




/// test (2x2) x (2x2)
unittest {
        // test (2x2) x (2x2)
    auto a = [[1f, 2f],
              [3f, 4f]].variable;
    auto b = [[-1f, -2f],
              [-3f, -4f]].variable;
    auto expected = [[1*-1+2*-3, 1*-2+2*-4],
                     [3*-1+4*-3, 3*-2+4*-4]];

    // test CPU
    {
        auto c = MatMul!float().forward(a, b);
        assert(c.sliced == expected);
    }
    version(grain_cuda) {
        auto c = MatMul!float().forward(a.to!DeviceStorage,
                                        b.to!DeviceStorage).to!HostStorage;
        writeln(c.sliced, expected);
        assert(c.sliced == expected);
    }
}


/// test (3x5) x (5x4)
unittest {
    import mir.ndslice : sliced;
    import std.format : format;
    auto a =
        [-5.0f,  1,  7, 7, -4,
           -1, -5,  6, 3, -3,
         -5, -2, -3, 6,  0].sliced(3, 5).variable;
    auto b =
        [-5.0f, -3,  3,  1,
            4,  3,  6,  4,
           -4, -2, -2,  2,
           -1,  9,  4,  8,
            9,  8,  3, -2].sliced(5, 4).variable;
    auto expected =
        [[-42,  35,  -7, 77],
         [-69, -21, -42, 21],
         [ 23,  69,   3, 29]];
    // C = 1 * AB + 0 * C
    {
        auto c = MatMul!float().forward(a, b);
        assert(c.sliced == expected);
    }
    version(grain_cuda) {
        auto c = MatMul!float().forward(a.to!DeviceStorage,
                                        b.to!DeviceStorage).to!HostStorage;
        writeln(c.sliced, expected);
        // assert(c.sliced == expected, "c.sliced %s != %s".format(c.sliced, expected));
    }
}


/// test (3x2) x (2x3)
unittest {
    import std.stdio;
    import mir.ndslice;

    auto a = [[1f, 3f],
              [5f, 7f],
              [9f, 11f]].variable;
    auto b = [[2f, 4f, 6f],
              [8f, 10f, 12f]].variable;
    auto expected = [[1*2+3*8, 1*4+3*10, 1*6+3*12],
                     [5*2+7*8, 5*4+7*10, 5*6+7*12],
                     [9*2+11*8, 9*4+11*10, 9*6+11*12]];

    version(grain_cuda) {{
        auto c = MatMul!(float)().forward(a.to!DeviceStorage,
                                          b.to!DeviceStorage).to!HostStorage;
        writeln(c.data);
        writeln(expected);
        // FIXME assert(c.sliced == expected);
    }}


    // test CPU
    {
        auto c = MatMul!(float)().forward(a, b);
        assert(c.sliced == expected);
    }
}

unittest {
    import std.typecons : tuple;
    import grain.testing;
    auto a = [[1.0f,2.0f],[3.0f,4.0f]].variable;
    auto b = [[1.0f,2.0f],[3.0f,4.0f]].variable;
    auto gc = [[1.0f,2.0f],[3.0f,4.0f]].variable;
    MatMul!float func;
    gradCheck(func, tuple(a, b), gc);
}

struct LogSoftmax(F) {
    // TODO use cudnn
    auto forward(Variable!(F, 2, HostStorage) x) {
        
    }
}

struct CrossEntropy(F, I=long) {
    Variable!(F, 0, HostStorage) forward(Variable!(F, 2, HostStorage) x, Variable!(I, 1, HostStorage) t) {
        return;
    }
}

