module grain.functions;

import grain.autograd;
import grain.cuda;

import std.stdio;


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
        // static foreach (i; 0..rets.length) {
        //     ugradOuts[i].isHost = rets[i].isHost;
        // }

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
                static if (vgradInputs[i].isHost) {
                    // TODO += grad
                    uinputs[i].grad.get!Storage[] = vgradInputs[i].data;
                } else {
                    // TODO += grad
                    import derelict.cuda.driverapi;
                    auto data = uinputs[i].grad.get!Storage;
                    copy(vgradInputs[i].data, data);
                    // data.ptr = vgradInputs[i].data.ptr;
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

    version(grain_cuda) {
        Variable!(T, dim, DeviceStorage) dx;

        auto forward(Variable!(T, dim, DeviceStorage) x) {
            import grain.kernel : relu;
            // FIXME if train
            this.dx = x.dup;
            auto y = this.inplace ? x : x.dup;
            auto n = cast(uint) y.data.length;
            Global.kernel!relu
                .launch(y.data.ptr, n, [1,1,1], [n,1,1]);
            return y;
        }

        auto backward(Variable!(T, dim, DeviceStorage) gy) {
            import grain.kernel : reluGrad;
            auto gx = gy.dup; // TODO: create empty
            auto n = cast(uint) gy.data.length;
            Global.kernel!reluGrad
                .launch(gx.data.ptr, gy.data.ptr, this.dx.data.ptr, n, [1,1,1], [n,1,1]);
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
        assert(ugy.isHost);
        y.backward(&ugy);
        assert(x.grad == [0, 2, 3]);
    }
    version (grain_cuda) {
        auto func = new ReLU!(float, 1);
        auto x = [-1.0f, 2.0f, 3.0f].variable(true).to!DeviceStorage;
        auto y = func.applyForward(x);
        auto gy = [1.0f, 2.0f, 3.0f].variable.to!DeviceStorage;
        auto ugy = UntypedVariable(gy);
        assert(!ugy.isHost);
        y.backward(&ugy);
        assert(x.grad.toHost() == [0, 2, 3]);
    }
}


///
unittest {
    foreach (inplace; [true, false]) {
        auto func = new ReLU!(float, 1);
        func.inplace = inplace;

        // test CPU
        {
            auto x = [-1.0f, 1.0f, 0.0f].variable;
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
            assert(gx.data == [0.0, 2.0, 3.0]);
        }
    }
}


struct MatMul(T, size_t dim) {
    T alpha = 1;
    T beta = 0;

    auto forward(Variable!(T, 2, HostStorage) x, Variable!(T, 2, HostStorage) y) {
        import lubeck : mtimes;
        return mtimes(x.sliced, y.sliced).variable(x.requiresGrad || y.requiresGrad);
    }

    version(grain_cuda) {
        auto forward(Variable!(T, 2, DeviceStorage) x, Variable!(T, 2, DeviceStorage) y) {
            import std.typecons : RefCounted;
            import grain.cublas; // : CUBLAS_STATUS_SUCCESS, cublasSgemm_v2;
            assert(x.shape[1] == y.shape[0]);
            auto dshape = [x.shape[0], y.shape[1]];
            auto d = RefCounted!(CuPtr!T)(x.shape[0] * y.shape[1]);
            static if (is(T == float)) {
                alias gemm = cublasSgemm_v2;
            } else {
                // TODO support double
                static assert(false, "unsupported type");
            }
            // TODO support transposed (CUBLAS_OP_T)
            // see https://github.com/libmir/mir-blas/blob/master/source/mir/blas.d#L299
            auto status = gemm(Global.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                               cast(int) dshape[0], cast(int) dshape[1], cast(int) x.shape[1],
                               &alpha,
                               cast(T*) x.data.ptr, cast(int) x.shape[0],
                               cast(T*) y.data.ptr, cast(int) y.shape[0],
                               &beta,
                               cast(T*) d.ptr, cast(int) dshape[0]);
            assert(status == CUBLAS_STATUS_SUCCESS);
            return Variable!(T, 2, DeviceStorage)(
                x.requiresGrad || y.requiresGrad,
                [x.shape[0], y.shape[1]],
                [x.shape[0], 1],
                d);
        }
    }
}

///
unittest {
    import std.stdio;
    import mir.ndslice;
    auto a = [[1f, 3f],
              [5f, 7f],
              [9f, 11f]].variable;
    auto b = [[2f, 4f, 6f],
              [8f, 10f, 12f]].variable;

    // test CPU
    {
        auto c = MatMul!(float, 2)().forward(a, b);
        assert(c.sliced == [[1*2+3*8, 1*4+3*10, 1*6+3*12],
                            [5*2+7*8, 5*4+7*10, 5*6+7*12],
                            [9*2+11*8, 9*4+11*10, 9*6+11*12]]);
        writeln(c.sliced);
    }

    /* FIXME
    version(grain_cuda) {
        auto c = MatMul!(float, 2)().forward(a.to!DeviceStorage,
                                             b.to!DeviceStorage).to!HostStorage;
        writeln(c.sliced);
        assert(c.sliced == [[1*2+3*8, 1*4+3*10, 1*6+3*12],
                            [5*2+7*8, 5*4+7*10, 5*6+7*12],
                            [9*2+11*8, 9*4+11*10, 9*6+11*12]]);
    }
    */
}

struct SoftmaxCrossEntropy {
    
}

