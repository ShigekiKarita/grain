module grain.ops;

import grain.variable;
import grain.cuda;

struct Context(Storage) {
    Storage[string] dict;
}


struct Autograd {
    auto forward(Op, Args...)(Args args) {
        auto result = Op().forward(args);
        return;
    }
}


struct ReLU(T, size_t dim) {
    bool inplace = false;

    auto forward(Variable!(T, dim, HostStorage) x) {
        import mir.ndslice : each;
        auto y = this.inplace ? x : x.dup;
        y.sliced.each!((ref a) { if (a < 0) a = 0; });
        return y;
    }

    auto backward(Variable!(T, dim, HostStorage) gy, Variable!(T, dim, HostStorage) x) {
        auto gx = gy.dup;
        foreach (i; 0..gx.data.length) {
            if (x.data[i] < 0.0) gx.data[i] = 0.0;
        }
        return gx;
    }

    version(grain_cuda) {
        auto forward(Variable!(T, dim, DeviceStorage) x) {
            import grain.kernel : relu;
            auto y = this.inplace ? x : x.dup;
            auto n = cast(uint) y.data.length;
            Global.kernel!relu
                .launch(y.data.ptr, n, [1,1,1], [n,1,1]);
            return y;
        }

        auto backward(Variable!(T, dim, DeviceStorage) gy, Variable!(T, dim, DeviceStorage) x) {
            import grain.kernel : reluGrad;
            auto gx = gy.dup; // TODO: create empty
            auto n = cast(uint) gy.data.length;
            Global.kernel!reluGrad
                .launch(gx.data.ptr, gy.data.ptr, x.data.ptr, n, [1,1,1], [n,1,1]);
            return gx;
        }
    }
}

///
unittest {
    foreach (inplace; [true, false]) {
        ReLU!(float, 1) func;
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

            x = [-1.0f, 1.0f, 0.0f].variable;
            auto gy = [1.0f, 2.0f, 3.0f].variable;
            auto gx = func.backward(gy, x);
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
            auto gxd = func.backward(gy.to!DeviceStorage, x.to!DeviceStorage);
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
                x.autograd || y.autograd, d,
                [x.shape[0], y.shape[1]],
                [x.shape[0], 1], null);
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

