/**
   A module for binary autograd functions.

   TODO:
   - support add tensor with cudnnAddTensor and mir.ndslice
   - support opBinary(add, mul, min, max) cudnnOpTensor
   - convolution
   - batchnorm
*/
module grain.functions.binary;

import grain.functions.common; // : FunctionCommon, TypeChecker;
import grain.autograd; //  : variable, Variable, UntypedVariable, HostStorage, DeviceStorage;
import grain.cuda;
import grain.utility;
import std.traits : isFloatingPoint;
import std.typecons : tuple;
import mir.ndslice : isSlice;
import std.format : format;


/// c = op(alpha1 * a + alpha2 * b) + beta * c;
struct OpBinary(T, size_t dim, string ops) if (isFloatingPoint!T) {
    import mir.ndslice;

    T alpha1 = 1, alpha2 = 1;

    uint[dim] shape1, shape2;
    Variable!(T, dim, HostStorage) ha, hb;

    auto forward(Variable!(T, dim, HostStorage) a, Variable!(T, dim, HostStorage) b) {
        auto info = broadcastable(a, b);
        assert(info.ok);
        auto abx = broadcast(a.sliced, b.sliced);
        auto ax = abx[0];
        auto bx = abx[1];
        auto c = slice(this.alpha1 * ax);

        // TODO if train
        this.shape1 = a.shape;
        this.shape2 = b.shape;
        // ops
        static if (ops == "+") {
            c[] += this.alpha2 * bx;
        } else static if (ops == "*") {
            this.ha = ax.variable;
            this.hb = bx.variable;
            c[] *= this.alpha2 * bx;
        } else {
            static assert("unknown operator: " ~ ops);
        }
        return c.variable(a.requiresGrad || b.requiresGrad);
    }

    auto backward(Variable!(T, dim, HostStorage) gc) {
        import numir;
        import mir.math : sum;
        import mir.ndslice;
        static if (ops == "+") {
            auto ga = this.alpha1 == 1 ? gc.sliced.slice.universal : slice(this.alpha1 * gc.sliced).universal;
            if (ga.shape != this.shape1) {
                ga = reduceShape!(sum!"fast")(ga, this.shape1.castArray!size_t).universal;
            }
            auto gb = this.alpha2 == 1 ? gc.sliced.slice.universal : slice(this.alpha2 * gc.sliced).universal;
            if (gb.shape != this.shape2) {
                gb = reduceShape!(sum!"fast")(gb, this.shape2.castArray!size_t).universal;
            }
            return tuple(ga.variable, gb.variable);
        } else static if (ops == "*") {
            assert(this.ha.defined);
            assert(this.hb.defined);
            auto ga = gc.sliced.slice.universal;
            ga[] *= this.alpha1 * this.alpha2 * this.hb.sliced;
            if (ga.shape != this.shape1) {
                ga = reduceShape!(sum!"fast")(ga, this.shape1.castArray!size_t).universal;
            }
            auto gb = gc.sliced.slice.universal;
            gb[] *= this.alpha1 * this.alpha2 * this.ha.sliced;
            if (gb.shape != this.shape2) {
                gb = reduceShape!(sum!"fast")(gb, this.shape2.castArray!size_t).universal;
            }
            return tuple(ga.variable, gb.variable);
        } else {
            static assert("unknown operator: " ~ ops);
        }
    }

    version (grain_cuda) {
        import grain.cudnn;
        import derelict.cudnn7;
        import std.algorithm : find;

        enum opBinaryDict = [
                             "+": CUDNN_OP_TENSOR_ADD,
                             "*": CUDNN_OP_TENSOR_MUL,
                             "min": CUDNN_OP_TENSOR_MIN,
                             "max": CUDNN_OP_TENSOR_MAX
                             ];

        static if (opBinaryDict.keys.find(ops)) {
            static if (ops == "*") {
                Variable!(T, dim, DeviceStorage) da, db;
            }

            auto forward(Variable!(T, dim, DeviceStorage) a, Variable!(T, dim, DeviceStorage) b) {
                // TODO implement non-cudnn case
                foreach (d; 0 .. dim) {
                    assert(a.shape[d] == b.shape[d] || b.shape[d] == 1,
                           "cuDNN does not support complete broadcasting");
                }
                // TODO if train
                this.shape1 = a.shape;
                this.shape2 = b.shape;
                static if (ops == "*") {
                    this.da = a;
                    this.db = b;
                }

                auto c = a.uninit;
                c.requiresGrad = a.requiresGrad || b.requiresGrad;
                import grain.cudnn;
                tensorOp!(opBinaryDict[ops], T, dim)(c, a, b, this.alpha1, this.alpha2);
                return c;
            }
        } else {
            static assert("unknown operator: " ~ ops);
        }

        static if (ops == "+") {
            auto backward(Variable!(T, dim, DeviceStorage) gc) {
                Variable!(T, dim, DeviceStorage) ga, gb;
                if (this.shape1 == gc.shape) {
                    ga = gc.dup;
                    if (this.alpha1 != 1.0) grain.cudnn.scale(ga, this.alpha1);
                } else {
                    ga = uninitVariable!(T, DeviceStorage)(this.shape1);
                    grain.cudnn.reduce!CUDNN_REDUCE_TENSOR_ADD(gc, ga, this.alpha1);
                }

                if (this.shape2 == gc.shape) {
                    gb = gc.dup;
                    if (this.alpha2 != 1.0) grain.cudnn.scale(gb, this.alpha2);
                } else {
                    gb = uninitVariable!(T, DeviceStorage)(this.shape2);
                    grain.cudnn.reduce!CUDNN_REDUCE_TENSOR_ADD(gc, gb, this.alpha2);
                }
                return tuple(ga, gb);
            }
        } else static if (ops == "*") {
            auto backward(Variable!(T, dim, DeviceStorage) gc) {
                auto gax = uninitVariable!(T, DeviceStorage)(gc.shape);
                auto gbx = uninitVariable!(T, DeviceStorage)(gc.shape);
                auto alpha = this.alpha1 * this.alpha2;
                grain.cudnn.tensorOp!CUDNN_OP_TENSOR_MUL(gax, gc, this.db, alpha);
                grain.cudnn.tensorOp!CUDNN_OP_TENSOR_MUL(gbx, gc, this.da, alpha);

                Variable!(T, dim, DeviceStorage) ga, gb;
                if (this.shape1 == gc.shape) {
                    ga = gax;
                } else {
                    ga = uninitVariable!(T, DeviceStorage)(this.shape1);
                    grain.cudnn.reduce!CUDNN_REDUCE_TENSOR_ADD(gax, ga);
                }

                if (this.shape2 == gc.shape) {
                    gb = gbx;
                } else {
                    gb = uninitVariable!(T, DeviceStorage)(this.shape2);
                    grain.cudnn.reduce!CUDNN_REDUCE_TENSOR_ADD(gbx, gb);
                }
                return tuple(ga, gb);
            }
        }
    }

    mixin FunctionCommon;
}

///
unittest {
    static foreach (op; ["+", "*"]) {
        foreach (j; [1, 2]) {
            import std.typecons : tuple;
            import numir : uniform, approxEqual;
            import mir.ndslice : slice;
            import grain.testing;

            auto a = uniform!float(3, 2).slice.variable;
            auto b = uniform!float(3, j).slice.variable;
            auto gc = uniform!float(3, 2).slice.variable;
            auto func = OpBinary!(float, 2, op)(1, 2);
            gradCheck(func, tuple(a, b), gc);

            auto c = func.forward(a, b);
            auto gab = func.backward(gc);
            version (grain_cuda) {
                auto dfunc = OpBinary!(float, 2, op)(1, 2);
                auto dc = dfunc.forward(a.to!DeviceStorage, b.to!DeviceStorage);
                assert(approxEqual(dc.to!HostStorage.sliced, c.sliced));
                auto dgab = dfunc.backward(gc.to!DeviceStorage);
                assert(approxEqual(dgab[0].to!HostStorage.sliced, gab[0].sliced));
                assert(approxEqual(dgab[1].to!HostStorage.sliced, gab[1].sliced));
            }
        }
    }
}

///
unittest {
    foreach (i; [1, 2]) {
        foreach (j; [1, 2]) {
            import std.typecons : tuple;
            import numir : uniform;
            import mir.ndslice : slice;
            import grain.testing;

            auto a = uniform!float(i, 2).slice.variable;
            auto b = uniform!float(2, j).slice.variable;
            auto gc = uniform!float(2, 2).slice.variable;
            auto func = OpBinary!(float, 2, "*")(1, 2);
            gradCheck(func, tuple(a, b), gc);
        }
    }
}


/// a and b have the same shape
unittest {
    import mir.ndslice;

    auto plus = OpBinary!(float, 2, "+")(1.0f, 2.0f);
    auto a = [[1.0f, 2.0f, 3.0f], [4.0f, 5.0f, 3.0f]].variable;
    auto b = [[-1.0f, 4.0f, 0.0f], [1.0f, 2.0f, 3.0f]].variable;
    auto hc = plus.forward(a, b);
    assert(hc.sliced == [[-1.0f, 10.0f, 3.0f], [6.0f, 9.0f, 9.0f]]);

    version (grain_cuda) {
        auto dplus = OpBinary!(float, 2, "+")(1.0f, 2.0f);
        auto dc = dplus.forward(a.to!DeviceStorage, b.to!DeviceStorage);
        assert(dc.to!HostStorage.sliced == [[-1.0f, 10.0f, 3.0f], [6.0f, 9.0f, 9.0f]]);
    }
}


/// a and b have different shapes
unittest {
    import mir.ndslice;

    auto plus = OpBinary!(float, 2, "+")(1.0f, 2.0f);
    auto a = [[1.0f, 2.0f, 3.0f], [4.0f, 5.0f, 3.0f]].variable;
    auto b = [[-1.0f, 4.0f, 0.0f]].variable;
    auto hc = plus.forward(a, b);
    assert(hc.sliced == [[-1.0f, 10.0f, 3.0f], [2.0f, 13.0f, 3.0f]]);

    version (grain_cuda) {
        auto dc = plus.forward(a.to!DeviceStorage, b.to!DeviceStorage);
        assert(dc.to!HostStorage.sliced == [[-1.0f, 10.0f, 3.0f], [2.0f, 13.0f, 3.0f]]);
    }
}

/// a and b have different shapes
unittest {
    import mir.ndslice;

    auto plus = OpBinary!(float, 2, "*")(1.0f, 2.0f);
    auto a = [[1.0f, 2.0f, 3.0f], [4.0f, 5.0f, 3.0f]].variable;
    auto b = [[-1.0f, 4.0f, 0.0f]].variable;
    auto hc = plus.forward(a, b);
    assert(hc.sliced == [[1*2*-1, 2*2*4, 0], [4*2*-1, 5*2*4, 0]]);

    version (grain_cuda) {
        auto dc = plus.forward(a.to!DeviceStorage, b.to!DeviceStorage);
        assert(dc.to!HostStorage.sliced ==[[1*2*-1, 2*2*4, 0], [4*2*-1, 5*2*4, 0]]);
    }
}


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
   TODO: generalize to broadcastable addition (use cudnnAddTensor)
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

/// Emebedding ID into vector. TODO: support N-dim input. support sparse weight matrix
struct Embedding(T) {
    import std.range : enumerate;
    import numir : view, empty, zeros;

    Variable!(int, 1, HostStorage) hx;
    uint[2] wshape;

    auto forward(Variable!(T, 2, HostStorage) weight, Variable!(int, 1, HostStorage) ids) {
        this.hx = ids; // TODO if train
        this.wshape = weight.shape; // TODO if train
        auto ys = empty!T(ids.shape[0], weight.shape[1]);
        foreach (i, id; ids.sliced.enumerate) {
            ys[i, 0..$] = weight.sliced[id, 0..$];
        }
        return ys.variable(weight.requiresGrad);
    }

    auto backward(Variable!(T, 2, HostStorage) gy) {
        auto gw = zeros!T(this.wshape.castArray!size_t);
        foreach (i, id; this.hx.sliced.enumerate) {
            gw[id, 0..$] += gy.sliced[i];
        }
        return tuple(gw.variable(gy.requiresGrad), typeof(this.hx)());
    }

    version (grain_cuda) {
        Variable!(int, 1, DeviceStorage) dx;

        auto forward(Variable!(T, 2, DeviceStorage) weight, Variable!(int, 1, DeviceStorage) ids) {
            import grain.kernel : embedding;
            this.dx = ids; // TODO if train
            this.wshape = weight.shape; // TODO if train
            auto ys = uninitVariable!(T, DeviceStorage, 2)([ids.shape[0], weight.shape[1]], weight.requiresGrad);
            Global.kernel!embedding
                .call(weight.data.ptr, ids.data.ptr, ys.data.ptr, weight.shape[0], weight.shape[1], ids.shape[0])
                .launch(weight.shape[1] * ids.shape[0]);
            return ys;
        }

        auto backward(Variable!(T, 2, DeviceStorage) gy) {
            import grain.kernel : embeddingGrad;
            auto gw = uninitVariable!(T, DeviceStorage, 2)(this.wshape, gy.requiresGrad);
            Global.kernel!embeddingGrad
                .call(gw.data.ptr, this.dx.data.ptr, gy.data.ptr, this.wshape[0], this.wshape[1], this.dx.shape[0])
                .launch(this.wshape[1] * this.dx.shape[0]);
            return tuple(gw, typeof(this.dx)());
        }
    }

    mixin FunctionCommon;
}

///
unittest {
    import numir;

    Embedding!float embed;
    auto w = [[1.0f, 2.0f], [3.0f, 4.0f]].nparray.variable;
    auto x = [0, 1, 0].variable;
    auto y = embed.forward(w, x);
    assert(y.sliced == [[1,2],[3,4],[1,2]]);

    auto gy = [[1f, 2f], [-1f, -2f], [1f, 0f]].nparray.variable;
    auto gw = embed.backward(gy)[0];
    assert(gw.sliced == [[2f, 2f], [-1f, -2f]]);

    version (grain_cuda) {
        Embedding!float dembed;
        auto dy = dembed.forward(w.to!DeviceStorage, x.to!DeviceStorage);
        assert(dy.to!HostStorage.sliced == y.sliced);
        auto dgw = dembed.backward(gy.to!DeviceStorage)[0];
        assert(dgw.to!HostStorage.sliced == gw.sliced);
    }
}


void generateStrides(const int* dimA, int* strideA, int nbDims, bool isNchw) {
    if (isNchw) {
        strideA[nbDims-1] = 1 ;
        for(int d = nbDims-2 ; d >= 0 ; d--) {
            strideA[d] = strideA[d+1] * dimA[d+1] ;
        }
    } else {
        strideA[1] = 1;
        strideA[nbDims-1] = strideA[1]*dimA[1];
        for(int d = nbDims-2 ; d >= 2 ; d--) {
            strideA[d] = strideA[d+1] * dimA[d+1] ;
        }
        strideA[0] = strideA[2]*dimA[2];
    }
}

/** Convert a linear index
i = d_1 s_1 ... s_n + d_2 s_2 ... s_n + d_n-1 s_n + d_n
into a multidimensional index
(d_1, d_2, ..., d_n)
*/
void lin2dim(size_t length)(int id, scope ref int[length] ids, const ref int[length] dims) {
    int idrem = id ;
    int prod  = 1 ; // accumulates the product of the dimensions
    foreach_reverse(i; 0 .. length) {
        ids[i] = (idrem / prod) % dims[i] ;
        idrem = id - ids[i] * prod ;
        prod *= dims[i] ;
    }
}

void doEpilog(float[] o, int idx, float alphaAcc, float beta) {
    if( beta == 0f ) {
        o[idx] = alphaAcc;
    } else {
        o[idx] = alphaAcc + o[idx]*beta;
    }
}

@nogc pure @safe int dim2lin(size_t length)(const ref int[length] ids, const int[] strides) {
    assert(length == strides.length);
    import mir.ndslice;
    import mir.math;
    return sum(ids.sliced * strides.sliced);
}

/// Reference CPU implementation of Convolution function
static struct ConvolutionRefImpl(T, size_t imDims, bool isConv=false, bool isNchw = true) {
    enum int nbDims = imDims + 2;

    static void forward(const T[] inputData,
                        const T[] filterData,
                        T[] outputData,
                        T alpha,
                        T beta,
                        const int[nbDims] inDims,
                        const int[nbDims] filDims,
                        const int[nbDims] outDims,
                        const int[nbDims] inStride,
                        const int[nbDims] filStride,
                        const int[nbDims] outStride,
                        const int[imDims] stride,
                        const int[imDims] pad,
                        const int[imDims] dilation,
                        )
    in {
        // Sanity checks
        // in     is n x c x h x w
        // out    is n x k x p x q
        // filter is k x c x r x s
        assert(inDims[0] == outDims[0]); // n
        assert(inDims[1] == filDims[1]); // k
        assert(outDims[1] == filDims[0]); // c
    } do {
        import std.algorithm : reduce;

        immutable nPixelsOut = outDims[2..$].reduce!"a * b";
        immutable nPixelsFil = filDims[2..$].reduce!"a * b";

        // Used to store coordinates
        int[imDims] filIds, outIds, inIds, tmpIds;
        // For each image in the output
        foreach (ni; 0 .. outDims[0]) {
            // For each feature layer of the output
            foreach (ki; 0 .. outDims[1]) {
                immutable outputOffset = ni * outStride[0] + ki * outStride[1] ;
                // Loop over all entries of the result
                foreach (outId; 0 .. nPixelsOut) {
                    // Get output pixel ids
                    lin2dim(outId, outIds, outDims[2..$]) ; // Skip n and k dimensions
                    // Now we get the coordinates in input space of
                    // the "top left" corner of the filter: multiply by stride and remove pad
                    inIds[] = outIds[] * stride[] - pad[];
                    // We then accumulate
                    T tmp = 0;
                    foreach (ci; 0 .. inDims[1]) {
                        immutable inputOffset = ni * inStride[0] + ci * inStride[1] ;
                        immutable filterOffset = ki * filStride[0] + ci * filStride[1] ;
                        foreach (filId; 0 .. nPixelsFil) {
                            // Get the position of the pixel
                            lin2dim(filId, filIds, filDims[2..$]) ;
                            // Compute the corresponding output pixel
                            // and check wether we are in the padding area on the fly too
                            // (not that for convolution, we flip the image patch
                            // (equivalent to flipping the filter patch))
                            bool inside = true ;
                            for (int d = 0; d < imDims && inside; d++) {
                                if (isConv) {
                                    tmpIds[d] = inIds[d] + dilation[d] * (filDims[2+d]-1 - filIds[d]);
                                } else {
                                    tmpIds[d] = inIds[d] + dilation[d] * filIds[d];
                                }
                                // If we are in the padding area: stop and skip computations
                                inside &= (tmpIds[d] >= 0 && tmpIds[d] < inDims[2+d]) ;
                            }
                            if (inside) {
                                immutable actualTmpId = inputOffset + dim2lin(tmpIds, inStride[2..$]);
                                immutable actualFilId = filterOffset + dim2lin(filIds, filStride[2..$]);
                                tmp += filterData[actualFilId] * inputData [actualTmpId];
                            }
                        }
                    }
                    // We put the result in the output
                    immutable actualOutId = outputOffset + dim2lin(outIds, outStride[2..$]);
                    doEpilog(outputData, actualOutId, alpha*tmp, beta);
                }
            }
        }
    }

    static void backwardData(const T[] weight,
                             const T[] top_diff,
                             scope T[] output,
                             float alpha,
                             float beta,

                             const int[nbDims] inDims,
                             const int[nbDims] filDims,
                             const int[nbDims] outDims,

                             const int[nbDims] inStride,
                             const int[nbDims] filterStride,
                             const int[nbDims] outStride,

                             const int[imDims] stride,
                             const int[imDims] pad,
                             const int[imDims] dilation)
    in {
        // Sanity checks
        // output is n x c x h x w
        // diff   is n x k x p x q
        // filter is k x c x r x s
        assert(inDims[0] == outDims[0]); // n
        assert(inDims[1] == filDims[0]); // k
        assert(outDims[1] == filDims[1]); // c
    } do {
        import std.algorithm : map, any, all, reduce;
        import std.range : iota;

        // Number of pixels in output
        immutable nPixelsOut = outDims[2..$].reduce!"a * b";
        // Number of pixels in filter
        immutable nPixelsFil = filDims[2..$].reduce!"a * b";

        int[imDims] outIds, filIds, accIds;
        foreach (ni; 0 .. outDims[0]) {
            foreach (ci; 0 .. outDims[1]) {
                foreach (outIdx; 0 .. nPixelsOut) {
                    lin2dim(outIdx, outIds, outDims[2..$]);
                    T val = 0;
                    // For every diff channel (k)
                    foreach (ki; 0 .. inDims[1]) {
                        immutable offsetFilter = ki * filterStride[0] + ci * filterStride[1];
                        immutable offsetDiff = ni * inStride[0] + ki * inStride[1];

                        foreach (filIdx; 0 .. nPixelsFil) {
                            lin2dim(filIdx, filIds, filDims[2..$]);

                            // Fetch the value in filter and diff, product and accumulate
                            // So basically, for the convolution,
                            // we replace r by dim-1-r and s by dim-1-s to "flip" the filter
                            // We can then just reason in term of correlation
                            accIds[] = outIds[] + pad[];
                            if (isConv){
                                accIds[] -= (filDims[2..$] - 1 - filIds[]) * dilation[];
                            } else {
                                accIds[] -= filIds[] * dilation[];
                            }
                            immutable outtaStride = iota(imDims).map!(i => accIds[i] % stride[i]).any;
                            if (outtaStride) {
                                continue;
                            }
                            accIds[] /= stride[];

                            immutable inBounds = iota(imDims).map!(i => 0 <= accIds[i] && accIds[i] < inDims[i+2]).all;
                            if (inBounds) {
                                immutable filterIdx = offsetFilter + dim2lin(filIds, filterStride[2..$]);
                                immutable diffIdx = offsetDiff + dim2lin(accIds, inStride[2..$]);
                                val += top_diff[diffIdx] * weight[filterIdx];
                            }
                        }
                        immutable offsetOut = ni * outStride[0] + ci * outStride[1];
                        doEpilog(output, offsetOut + outIdx, alpha*val, beta);
                    }
                }
            }
        }
    }

    static void backwardWeight(/*const TensorNdTestDesc_t *tensorInputDesc,*/
                               const T[] image,
                               /*const TensorNdTestDesc_t *tensorDiffDesc,*/
                               const T[] diffData,
                               /*const ConvNdTestDesc_t *convDesc,*/
                               /*const TensorNdTestDesc_t *filterOutputDesc,*/
                               float alpha,
                               float beta,
                               scope T[] output,

                               const int[nbDims] inDims,
                               const int[nbDims] filDims,
                               const int[nbDims] diffDims,
                               const int[nbDims] inStride,
                               const int[nbDims] filterStride,
                               const int[nbDims] diffStride,
                               const int[imDims] stride,
                               const int[imDims] pad,
                               const int[imDims] dilation)
    in {
        // Some sanity checks
        // image   is n x c x h x w
        // diff    is n x k x p x q
        // filter  is k x c x r x s
        assert(inDims[0] == diffDims[0]) ;
        assert(inDims[1] == filDims[1]) ;
        assert(diffDims[1]  == filDims[0]) ;

    } do {
        import std.algorithm : all, sum, map, reduce;
        import std.range : iota;

        // Number of pixels in output
        immutable nPixelsDiff = diffDims[2..$].reduce!"a * b";
        // Number of pixels in filter
        immutable nPixelsFil = filDims[2..$].reduce!"a * b";

        // For every filter pixel (k x c x r x s)
        int[imDims] filIds, diffIds, accIds;
        foreach (ki; 0 .. filDims[0]){
            foreach (ci; 0 .. filDims[1]) {
                foreach (filIdx; 0 .. nPixelsFil) {
                    lin2dim(filIdx, filIds, filDims[2..$]);
                    T val = 0;
                    // For every image (n)
                    foreach (ni; 0 .. inDims[0]) { // Sum over the batch
                        immutable offsetIn = ni * inStride[0] + ci * inStride[1] ;
                        immutable offsetDiff = ni * diffStride[0] + ki * diffStride[1] ;
                        // For every pixel in diff
                        foreach (diffIdx; 0 .. nPixelsDiff) {
                            lin2dim(diffIdx, diffIds, diffDims[2..$]);
                            // Fetch the value in image and diff, product and accumulate
                            accIds[] = diffIds[] * stride[] - pad[];

                            // Convolution = Correlation with a flipped filter
                            // So basically, for the convolution, we replace r by dim-1-r and s
                            // by dim-1-s to "flip" the filter
                            // We can then just reason in term of correlation
                            if (isConv){
                                accIds[] += (filDims[2..$] - 1 - filIds[]) * dilation[];
                            } else {
                                // The effect of dilation on the gradient is to start the "zone of influence"
                                // of a given pixel further into the image, so dilation
                                // only produces a shift in x and y
                                accIds[] += filIds[] * dilation[];
                            }
                            // Image value
                            immutable inBounds = iota(imDims).map!(i => 0 <= accIds[i] && accIds[i] < inDims[i+2]).all;
                            if (inBounds) {
                                immutable imId = offsetIn + dim2lin(accIds, inStride[2..$]);
                                // Diff value
                                immutable diffId  = offsetDiff + dim2lin(diffIds, diffStride[2..$]);
                                // Prod and accumulate
                                val += image[imId] * diffData[diffId];
                            }
                        }
                    }
                    immutable offsetFilter = ki * filterStride[0] + ci * filterStride[1];
                    doEpilog(output, offsetFilter + filIdx, alpha*val, beta);
                }
            }
        }
    }
}

/++
 Convolution/Cross-correration function

 TODO add cudnn wrapped functions
 +/
struct Convolution(T, size_t imDims, bool isConv=false, bool isNchw = true) {
    int[imDims] stride;
    int[imDims] pad;
    int[imDims] dilation;
    enum int nbDims = imDims + 2;
    enum int ngroup=1; // TODO support ngroup > 1?
    alias RefImpl = ConvolutionRefImpl!(T, imDims, isConv, isNchw);

    Variable!(T, nbDims, HostStorage) hx, hw;

    /// https://pytorch.org/docs/master/nn.html#convolution-layers
    auto outShape(uint[nbDims] inShape, uint[nbDims] weightShape) {
        uint[nbDims] ret;
        ret[0] = inShape[0]; // batchsize
        ret[1] = weightShape[0]; // output ch size
        assert(inShape[1] == weightShape[1]);
        auto kernel = weightShape[2..$];
        foreach (d; 0 .. imDims) {
            ret[d+2] = cast(uint)
                ((inShape[d+2] + 2 * pad[d] - dilation[d] * (kernel[d] - 1) - 1)
                 / stride[d] + 1);
        }
        return ret;
    }

    void setDefault() {
        foreach (d; 0..imDims) {
            if (this.stride[d] == 0) {
                this.stride[d] = 1;
            }
            if (this.dilation[d] == 0) {
                this.dilation[d] = 1;
            }
        }
    }

    auto forward(Variable!(T, nbDims, HostStorage) x, Variable!(T, nbDims, HostStorage) w) {
        this.setDefault();
        // TODO if train
        this.hx = x;
        this.hw = w;
        auto y = uninitVariable!(T, HostStorage)(outShape(x.shape, w.shape));
        RefImpl.forward(x.data, w.data, y.data,
                        1f, 0f,
                        x.shape.castArray!int, w.shape.castArray!int, y.shape.castArray!int,
                        x.strides, w.strides, y.strides,
                        this.stride, this.pad, this.dilation);
        return y;
    }


    auto backward(Variable!(T, nbDims, HostStorage) gy) {
        // TODO use requires_grad for skipping grad calc
        auto gx = this.hx.uninit;
        gx.data.zero_();
        RefImpl.backwardData(this.hw.data, gy.data, gx.data,
                             1f, 0f,
                             gy.shape.castArray!int, this.hw.shape.castArray!int, gx.shape.castArray!int,
                             gy.strides, this.hw.strides, gx.strides,
                             this.stride, this.pad, this.dilation);

        auto gw = this.hw.uninit;
        gw.data.zero_();
        RefImpl.backwardWeight(this.hx.data, gy.data,
                               1f, 0f,
                               gw.data,
                               this.hx.shape.castArray!int, gw.shape.castArray!int, gy.shape.castArray!int,
                               this.hx.strides, gw.strides, gy.strides,
                               this.stride, this.pad, this.dilation);
        return tuple(gx, gw);
    }

    version (grain_cuda) {
        import derelict.cudnn7;
        import grain.cudnn;
        // TODO implement benchmark mode to search the best algo
        cudnnConvolutionFwdAlgo_t forwardAlgo = // CUDNN_CONVOLUTION_FWD_ALGO_DIRECT;
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
        cudnnConvolutionBwdDataAlgo_t backwardAlgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;;
        // CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;

        Variable!(T, nbDims, DeviceStorage) dx, dw;

        auto forward(Variable!(T, nbDims, DeviceStorage) x, Variable!(T, nbDims, DeviceStorage) w) {
            this.setDefault();
            // TODO if train
            this.dx = x;
            this.dw = w;
            auto y = uninitVariable!(T, DeviceStorage)(outShape(x.shape, w.shape));
            grain.cudnn.convForward!(isConv, isNchw)(x, w, y, this.stride, this.pad, this.dilation,
                                                     this.ngroup, this.forwardAlgo);
            return y;
        }


        auto backward(Variable!(T, nbDims, DeviceStorage) gy) {
            // TODO use requires_grad for skipping grad calc
            auto gx = this.dx.uninit;
            gx.data.zero_();
            auto gw = this.dw.uninit;
            gw.data.zero_();
            // TODO separate data/weight backward
            grain.cudnn.convBackward!(isConv, isNchw)
                (gx, this.dx, gw, this.dw, gy, this.stride, this.pad, this.dilation,
                 this.ngroup, this.backwardAlgo);
            return tuple(gx, gw);
        }

    }
}

/** Conv1d pytorch equality test
   ``` python
   >>> iota = lambda s: torch.arange(torch.prod(torch.tensor(s))).view(s)
   >>> torch.nn.functional.conv1d(iota([2, 3, 4]), iota([5, 3, 3]))
   tensor([[[  258.,   294.],
            [  663.,   780.],
            [ 1068.,  1266.],
            [ 1473.,  1752.],
            [ 1878.,  2238.]],

           [[  690.,   726.],
            [ 2067.,  2184.],
            [ 3444.,  3642.],
            [ 4821.,  5100.],
            [ 6198.,  6558.]]])
   >>> y.shape
   [2, 5, 2]

   >>> x = iota([2, 3, 4])
   >>> x.requires_grad = True
   >>> w = iota([5, 3, 3])
   >>> w.requires_grad = True
   >>> y = torch.nn.functional.conv1d(x, w)
   >>> y.backward(torch.ones_like(y))
   >>> x.grad
   tensor(
       [[[  90.,  185.,  195.,  100.],
         [ 105.,  215.,  225.,  115.],
         [ 120.,  245.,  255.,  130.]],

        [[  90.,  185.,  195.,  100.],
         [ 105.,  215.,  225.,  115.],
         [ 120.,  245.,  255.,  130.]]])
   >>> w.grad
   tensor([[[ 26.,  30.,  34.],
            [ 42.,  46.,  50.],
            [ 58.,  62.,  66.]],

           [[ 26.,  30.,  34.],
            [ 42.,  46.,  50.],
            [ 58.,  62.,  66.]],

           [[ 26.,  30.,  34.],
            [ 42.,  46.,  50.],
            [ 58.,  62.,  66.]],

           [[ 26.,  30.,  34.],
            [ 42.,  46.,  50.],
            [ 58.,  62.,  66.]],

           [[ 26.,  30.,  34.],
            [ 42.,  46.,  50.],
            [ 58.,  62.,  66.]]])

   ```
*/
unittest {
    import std.stdio;
    import mir.ndslice;
    import numir;
    auto x = iota(2, 3, 4).as!float.slice.variable;
    auto w = iota(5, 3, 3).as!float.slice.variable;
    Convolution!(float, 1) conv;
    auto y = conv.forward(x, w);
    auto yx = [[[  258.,   294.],
                [  663.,   780.],
                [ 1068.,  1266.],
                [ 1473.,  1752.],
                [ 1878.,  2238.]],

               [[  690.,   726.],
                [ 2067.,  2184.],
                [ 3444.,  3642.],
                [ 4821.,  5100.],
                [ 6198.,  6558.]]];
    assert(y.sliced == yx);

    // test backward
    auto gy = y.uninit;
    gy.data[] = 1;
    auto gs = conv.backward(gy);
    auto gx = gs[0];
    auto gw = gs[1];

    auto gxx = [[[  90.,  185.,  195.,  100.],
                 [ 105.,  215.,  225.,  115.],
                 [ 120.,  245.,  255.,  130.]],

                [[  90.,  185.,  195.,  100.],
                 [ 105.,  215.,  225.,  115.],
                 [ 120.,  245.,  255.,  130.]]];
    assert(gx.sliced == gxx);

    auto gwx = [[[ 26.,  30.,  34.],
                 [ 42.,  46.,  50.],
                 [ 58.,  62.,  66.]],

                [[ 26.,  30.,  34.],
                 [ 42.,  46.,  50.],
                 [ 58.,  62.,  66.]],

                [[ 26.,  30.,  34.],
                 [ 42.,  46.,  50.],
                 [ 58.,  62.,  66.]],

                [[ 26.,  30.,  34.],
                 [ 42.,  46.,  50.],
                 [ 58.,  62.,  66.]],

                [[ 26.,  30.,  34.],
                 [ 42.,  46.,  50.],
                 [ 58.,  62.,  66.]]];
    assert(gw.sliced == gwx);

    import grain.testing : gradCheck;
    auto hx = uniform!float(x.shape.castArray!size_t).slice.variable;
    auto hw = uniform!float(w.shape.castArray!size_t).slice.variable;
    auto hgy = uniform!float(y.shape.castArray!size_t).slice.variable;
    auto hy = conv.forward(hx, hw);
    auto hgx = conv.backward(hgy);
    gradCheck(conv, tuple(hx, hw), hgy, 1e-3, 1e-3, 1e-2);

    version (grain_cuda) {
        auto dy = conv.forward(hx.to!DeviceStorage, hw.to!DeviceStorage);
        auto dgx = conv.backward(hgy.to!DeviceStorage);
        assert(approxEqual(dy.to!HostStorage.sliced, hy.sliced));
        assert(approxEqual(dgx[0].to!HostStorage.sliced, hgx[0].sliced));
        assert(approxEqual(dgx[1].to!HostStorage.sliced, hgx[1].sliced));
    }
}

/** Conv2d pytorch equality test
   ``` python
   >>> import torch
   >>> iota = lambda s: torch.arange(torch.prod(torch.tensor(s))).view(s)
   >>> x = iota([2, 3, 4, 4])
   >>> px.requires_grad = True
   >>> w = iota([2, 3, 3, 3])
   >>> w.requires_grad = True
   >>> y = torch.nn.functional.conv2d(x, w)
   >>> y
   tensor([[[[ 10197.,  10548.],
             [ 11601.,  11952.]],

            [[ 25506.,  26586.],
             [ 29826.,  30906.]]],


            [[[ 27045.,  27396.],
              [ 28449.,  28800.]],

             [[ 77346.,  78426.],
              [ 81666.,  82746.]]]])

   >>> y = torch.nn.functional.conv1d(iota([2, 3, 4]), w)
   >>> y.backward(torch.ones_like(y))
   >>> x.grad
   tensor(
       [[[[  27.,   56.,   60.,   31.],
          [  60.,  124.,  132.,   68.],
          [  72.,  148.,  156.,   80.],
          [  39.,   80.,   84.,   43.]],

         [[  45.,   92.,   96.,   49.],
          [  96.,  196.,  204.,  104.],
          [ 108.,  220.,  228.,  116.],
          [  57.,  116.,  120.,   61.]],

         [[  63.,  128.,  132.,   67.],
          [ 132.,  268.,  276.,  140.],
          [ 144.,  292.,  300.,  152.],
          [  75.,  152.,  156.,   79.]]],


        [[[  27.,   56.,   60.,   31.],
          [  60.,  124.,  132.,   68.],
          [  72.,  148.,  156.,   80.],
          [  39.,   80.,   84.,   43.]],

         [[  45.,   92.,   96.,   49.],
          [  96.,  196.,  204.,  104.],
          [ 108.,  220.,  228.,  116.],
          [  57.,  116.,  120.,   61.]],

         [[  63.,  128.,  132.,   67.],
          [ 132.,  268.,  276.,  140.],
          [ 144.,  292.,  300.,  152.],
          [  75.,  152.,  156.,   79.]]]])
   >>> w.grad
   tensor(
       [[[[ 212.,  220.,  228.],
          [ 244.,  252.,  260.],
          [ 276.,  284.,  292.]],

         [[ 340.,  348.,  356.],
          [ 372.,  380.,  388.],
          [ 404.,  412.,  420.]],

         [[ 468.,  476.,  484.],
          [ 500.,  508.,  516.],
          [ 532.,  540.,  548.]]],


        [[[ 212.,  220.,  228.],
          [ 244.,  252.,  260.],
          [ 276.,  284.,  292.]],

         [[ 340.,  348.,  356.],
          [ 372.,  380.,  388.],
          [ 404.,  412.,  420.]],

         [[ 468.,  476.,  484.],
          [ 500.,  508.,  516.],
          [ 532.,  540.,  548.]]]])
   ```
*/
unittest {
    import std.stdio;
    import mir.ndslice;
    import numir;
    auto x = iota(2, 3, 4, 4).as!float.slice.variable;
    auto w = iota(2, 3, 3, 3).as!float.slice.variable;
    Convolution!(float, 2) conv;
    auto y = conv.forward(x, w);
    auto yx = [[[[ 10197.,  10548.],
                 [ 11601.,  11952.]],
                [[ 25506.,  26586.],
                 [ 29826.,  30906.]]],
               [[[ 27045.,  27396.],
                 [ 28449.,  28800.]],
                [[ 77346.,  78426.],
                 [ 81666.,  82746.]]]];
    assert(y.sliced == yx);

    // test backward
    auto gy = y.uninit;
    gy.data[] = 1;
    auto gs = conv.backward(gy);
    auto gx = gs[0];
    auto gw = gs[1];

    auto gxx = [[[[  27.,   56.,   60.,   31.],
                  [  60.,  124.,  132.,   68.],
                  [  72.,  148.,  156.,   80.],
                  [  39.,   80.,   84.,   43.]],

                 [[  45.,   92.,   96.,   49.],
                  [  96.,  196.,  204.,  104.],
                  [ 108.,  220.,  228.,  116.],
                  [  57.,  116.,  120.,   61.]],

                 [[  63.,  128.,  132.,   67.],
                  [ 132.,  268.,  276.,  140.],
                  [ 144.,  292.,  300.,  152.],
                  [  75.,  152.,  156.,   79.]]],


                [[[  27.,   56.,   60.,   31.],
                  [  60.,  124.,  132.,   68.],
                  [  72.,  148.,  156.,   80.],
                  [  39.,   80.,   84.,   43.]],

                 [[  45.,   92.,   96.,   49.],
                  [  96.,  196.,  204.,  104.],
                  [ 108.,  220.,  228.,  116.],
                  [  57.,  116.,  120.,   61.]],

                 [[  63.,  128.,  132.,   67.],
                  [ 132.,  268.,  276.,  140.],
                  [ 144.,  292.,  300.,  152.],
                  [  75.,  152.,  156.,   79.]]]];
    assert(gx.sliced == gxx);

    auto gwx = [[[[ 212.,  220.,  228.],
                  [ 244.,  252.,  260.],
                  [ 276.,  284.,  292.]],
                 [[ 340.,  348.,  356.],
                  [ 372.,  380.,  388.],
                  [ 404.,  412.,  420.]],
                 [[ 468.,  476.,  484.],
                  [ 500.,  508.,  516.],
                  [ 532.,  540.,  548.]]],
                [[[ 212.,  220.,  228.],
                  [ 244.,  252.,  260.],
                  [ 276.,  284.,  292.]],
                 [[ 340.,  348.,  356.],
                  [ 372.,  380.,  388.],
                  [ 404.,  412.,  420.]],
                 [[ 468.,  476.,  484.],
                  [ 500.,  508.,  516.],
                  [ 532.,  540.,  548.]]]];
    assert(gw.sliced == gwx);

    import grain.testing : gradCheck;
    auto hx = uniform!float(x.shape.castArray!size_t).slice.variable;
    auto hw = uniform!float(w.shape.castArray!size_t).slice.variable;
    auto hgy = uniform!float(y.shape.castArray!size_t).slice.variable;
    auto hy = conv.forward(hx, hw);
    auto hgx = conv.backward(hgy);
    gradCheck(conv, tuple(hx, hw), hgy, 1e-3, 1e-3, 1e-2);

    version (grain_cuda) {
        auto dy = conv.forward(hx.to!DeviceStorage, hw.to!DeviceStorage);
        auto dgx = conv.backward(hgy.to!DeviceStorage);
        assert(approxEqual(dy.to!HostStorage.sliced, hy.sliced));
        assert(approxEqual(dgx[0].to!HostStorage.sliced, hgx[0].sliced));
        assert(approxEqual(dgx[1].to!HostStorage.sliced, hgx[1].sliced));
    }
}
