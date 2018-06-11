/**
   Common components for autograd function object
 */
module grain.functions.common;

import grain.autograd;
import grain.cuda;
import grain.utility : toTuple, fromTuple, castArray;
import mir.ndslice : isSlice;

import std.stdio;

version (grain_cuda) {
    import cudnn = derelict.cudnn7;
}

/// a simple type check of forward/backward functions compatibility
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

/// a trait to identify autograd functions
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

/// common components (typecheck and backprop wrappers) for autograd functions
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

    /// store grain.autograd.BackProp object in returned variables from forward function
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

    /// type-erased version of backward function used in grain.autograd.BackProp object
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


/// check if broadcastable
auto broadcastable(T, size_t dim, alias Storage)(Variable!(T, dim, Storage) a, Variable!(T, dim, Storage) b) {
    import std.typecons : tuple;
    import std.algorithm : max;
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

/// expand dimension i.e. repeat n time on dim
auto expand(size_t dim, S)(S s, size_t n) if (isSlice!S) {
    import numir : Ndim;
    static assert(dim < Ndim!S, format!"acessing invalid dim %d (should be < %d)"(dim, Ndim!S));
    assert(s.length!dim == 1);

    import mir.ndslice : repeat, swapped, transposed, unpack;
    /// [a, 1, b] -> repeat [n, a, 1, b] -> swapped [1, a, n, b]
    return s.repeat(n).unpack.swapped!(0, dim+1)[0];
}

///
nothrow pure @safe
unittest {
    import mir.ndslice;
    assert(iota(1, 1, 3).expand!1(3) ==
           [[[0,1,2],[0,1,2],[0,1,2]]]);
    assert(iota(1, 1, 3).expand!0(2).expand!1(3) ==
           [[[0,1,2],[0,1,2],[0,1,2]],
            [[0,1,2],[0,1,2],[0,1,2]]]);
    assert(iota(1, 3, 2).expand!0(2) == [[[0,1],[2,3],[4,5]],
                                         [[0,1],[2,3],[4,5]]]);
}

/// exapand dimension if s.length!dim == 1 else do nothing but type in the same expressions of repeat/unpack/swapped/index[0]
auto maybeExpand(size_t dim, S)(S s, size_t n) if (isSlice!S) {
    import mir.ndslice;
    import mir.ndslice : repeat, swapped, transposed, unpack;
    return s.length!dim == 1 ? s.expand!dim(n) :
        /// [a, c, b] -> repeat [1, a, c, b] -> swapped [1, a, c, b]
        s.repeat(1).unpack.swapped!(dim+1, dim+1)[0];
}

///
@nogc nothrow pure @safe
unittest {
    import mir.ndslice;
    assert(iota(1, 3, 2).maybeExpand!0(2) == iota(3, 2).repeat(2));
    assert(iota(3, 2).maybeExpand!0(2) == iota(3, 2));
}

/**
   Returns:
      broadcasted slice.
      For example, when a has its shape [a, 1] and b has [1, b],
      this function returns expanded a and b with a broadcasted shape [a, b].

   See_also:
      https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html
*/
auto broadcast(S1, S2)(S1 a0, S2 b0) if (isSlice!S1 && isSlice!S2) {
    import std.format : format;
    import std.typecons : tuple;
    import numir.core : Ndim;
    static assert(Ndim!S1 == Ndim!S2); // TODO support dim mismatched slices by unsqueezing like numpy
    enum dim = Ndim!S1;
    static foreach (d; 1 .. dim+1) {
        mixin(format!q{auto a%d = a%d.maybeExpand!(d-1)(b0.length!(d-1));}(d, d-1));
        mixin(format!q{auto b%d = b%d.maybeExpand!(d-1)(a0.length!(d-1));}(d, d-1));
    }
    mixin(format!q{auto ax = a%d;}(dim));
    mixin(format!q{auto bx = b%d;}(dim));
    return tuple(ax, bx);
}

///
@nogc nothrow pure @safe
unittest {
    import mir.ndslice;
    auto a = iota(1, 3, 1);
    auto b = iota(1, 1, 2);
    auto x = broadcast(a, b);
    assert(broadcast(a, b)[0] == a.expand!2(2));
    assert(broadcast(a, b)[1] == b.expand!1(3));
}

/// reduce slice into targetShape, TODO @nogc
auto reduceShape(alias fun, S, size_t N)(S s0, size_t[N] targetShape) {
    import numir;
    import mir.ndslice;
    import mir.math : sum;
    import std.format : format;
    import std.exception : assumeWontThrow; // TODO unsqueeze can be assumeWontThrow
    auto rec(size_t n, T)(T t) {
        static if (n == N) return t;
        else {
            return
                rec!(n+1)(
                    targetShape[n] == 1
                    ? assumeWontThrow(t.alongDim!n.map!fun.slice.unsqueeze!n).slice
                    : t.slice);
        }
    }
    return rec!0(s0);
}

nothrow pure @safe unittest {
    import mir.ndslice;
    import mir.math;
    import numir;
    import std.exception : assumeWontThrow; // TODO unsqueeze can be assumeWontThrow
    assert(iota(2, 3).reduceShape!sum([2, 1]) == assumeWontThrow(iota(2, 3).alongDim!1.map!sum.slice.unsqueeze!1));
}
