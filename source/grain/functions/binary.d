/**
   A module for binary autograd functions.
 */
module grain.functions.binary;

import grain.functions.common; // : FunctionCommon, TypeChecker;
import grain.autograd; //  : variable, Variable, UntypedVariable, HostStorage, DeviceStorage;
import grain.cuda;
import grain.utility;


/++
 Matrix-Matrix multiplication (using cuBLAS)

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

/// matmul with variable.backward
unittest {
    import std.typecons;
    import numir;
    import mir.ndslice;
    grain.autograd.backprop = true;
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

/**
   Add bias vector to matrix used inside grain.chain.Linear
   TODO: generalize to broadcastable addition
*/
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

///
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
