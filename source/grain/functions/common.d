/**
   Common components for autograd function object
 */
module grain.functions.common;

import grain.autograd;
import grain.cuda;
import grain.utility : toTuple, fromTuple, castArray;

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
