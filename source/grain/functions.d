module grain.functions;

import grain.autograd;
import grain.cuda;

import std.stdio;

/++ NOTE: instead of inheriting Function, make FunctionImpl non-copyable and pass delegate

mixin template FunctionCommon() {

    auto applyForward(F, Args...)(F func, Args args) {
        ...
        auto f = uniq!F(func);
        auto rets = func(args);
        foreach (r; rets) {
            r.children = this.vargs;
            r.bros = this.vrets;
            r.backProc = (UntypedVariable[] grads) { this.gradOutputs = f.backward(grads); };
        }
        return ret;
    }
}

struct Variable(...) {

    void backward() {
         if (this.bros.any!"a.grad.empty") return;
         this.backwardProc(this.bros.map!"a.grad".array);
         this.children.each!"a.backward()";
    }
}

 +/

mixin template FunctionCommon() {
    import std.typecons : Tuple;
    import std.traits : arity, Parameters, ReturnType;

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

    auto applyForward(Args...)(Args args) {
        import std.traits : hasMember;
        // maybe useless
        this.vargs = [];
        this.vrets = [];
        this.gradOutputs.clear;
        foreach (a; args) {
            // TODO find better way
            static if (hasMember!(typeof(a), "isVariable")) {
                this.vargs ~= [UntypedVariable(a)];
            }
        }
        auto rets = this.forward(args);
        static if (hasMember!(typeof(rets), "length")) {
            foreach (r; rets) {
                // TODO find better way
                static if (hasMember!(typeof(r), "isVariable")) {
                    auto u = UntypedVariable(r);
                    u.func = this;
                    this.vrets ~= [u];
                }
            }
        } else {
            auto u = UntypedVariable(rets);
            u.func = this;
            this.vrets = [u];
        }
        return rets;
    }

    // TODO: applyBackward
    auto applyBackward(UntypedVariable[] uargs) {
        UntypedVariable[] ret;
        return ret;
    }

    // TODO arg type check vrets(forward) == vargs(backward) && vargs(forward) == vrets(backward)
}

class ReLU(T, size_t dim) : Function {
    mixin FunctionCommon;
    bool inplace = false;
    Variable!(T, dim, HostStorage) hx;
    Variable!(T, dim, DeviceStorage) dx;

    auto forward(Variable!(T, dim, HostStorage) x) {
        import mir.ndslice : each;
        // FIXME if train
        this.hx = x.dup;
        auto y = this.inplace ? x : x.dup;
        y.sliced.each!((ref a) { if (a < 0) a = 0; });
        writeln("hx: ", this.hx.dup);
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
    auto func = new ReLU!(float, 1);
    auto x = [-1.0f, 2.0f, 3.0f].variable;
    auto y = func.applyForward(x);
    auto gy = [1.0f, 2.0f, 3.0f].variable;
    auto gys = [UntypedVariable(gy)];
    // auto gxs = func.applyBackward(gys);
    func.vargs.writeln;
    func.vrets.writeln;
    x.writeln;
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
        return mtimes(x.sliced, y.sliced).variable(x.autograd || y.autograd);
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
                x.autograd || y.autograd,
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

