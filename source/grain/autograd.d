module grain.autograd;

import std.traits : isArray;
import std.typecons : RefCounted;
import mir.ndslice : isSlice;

import grain.cuda;

alias HostStorage(T) = T[];

version(grain_cuda) {
    alias DeviceStorage(T) = CuPtr!T;

    enum bool isDevice(T) = is(typeof({T.init.toHost();}));

    enum bool isHost(T) = !isDevice!T;

    auto to(alias S : DeviceStorage, T)(T[] src) {
        return DeviceStorage!T(src);
    }

    auto to(alias S : HostStorage, Src)(Src src) if (isDevice!Src) {
        return src.toHost();
    }
}


/// type-erased variable
struct UntypedVariable {
    import std.variant;
    bool requiresGrad;
    size_t dim;
    size_t[] shape;
    ptrdiff_t[] strides;
    bool isHost;
    TypeInfo elem;
    Variant data;
    UntypedVariable* grad; // , gradData;
    size_t outPosition = 0;
    RefCounted!BackProp bprop;

    this(T, size_t dim, alias Storage)(Variable!(T, dim, Storage) v) {
        this.elem = typeid(T);
        this.requiresGrad = v.requiresGrad;
        this.shape = v.shape.dup;
        this.strides = v.strides.dup;
        this.dim = dim;
        this.isHost = is(Storage!T == HostStorage!T);
        this.data = v.data;
        this.grad = &v.grad;
    }

    auto get(T)() {
        return this.data.get!(RefCounted!T);
    }

    auto to(V : Variable!(T, dim, Storage), T, size_t dim, alias Storage)() {
        auto d = this.data.get!(RefCounted!(Storage!T));
        return Variable!(T, dim, Storage)(this.requiresGrad, this.shape[0..dim], this.strides[0..dim], d);
    }

    void backward(UntypedVariable* gradOutput=null) {
        if (bprop.refCountedStore.isInitialized) {
            bprop.backward(gradOutput, outPosition);
        }
    }

    string toString() {
        import std.format : format;
        return "UntypedVariable(%s, dim=%d, isHost=%s, data=%s, shape=%s)".format(
            elem, dim, isHost, data, shape);
    }
}

/// FIXME maybe singleton?
shared bool backprop = false;

/// Informations for backpropagation
struct BackProp {
    alias Proc = UntypedVariable[] delegate(UntypedVariable[]);
    Proc proc;
    UntypedVariable[] inputs;
    UntypedVariable[] gradOutputs;
    size_t nGrad = 0;

    void backward(UntypedVariable* grad=null, size_t pos=0) {
        import std.exception : enforce;
        import std.range : empty;
        enforce(!this.inputs.empty, "nothing to backprop");
        ++this.nGrad;
        if (grad is null) {
            enforce(this.gradOutputs.length == 1, "this variable is not loss");
        } else {
            this.gradOutputs[pos] = *grad; // FIXME??
        }
        if (grad is null || this.nGrad == this.gradOutputs.length) {
            auto gradInputs = proc(this.gradOutputs);
            assert(gradInputs.length == inputs.length, "invalid number of input gradients");
            foreach (i; 0 .. inputs.length) {
                if (inputs[i].requiresGrad) {
                    // *inputs[i].grad = gradInputs[i];
                    writeln(inputs[i]);
                    writeln(gradInputs[i]);
                }
                inputs[i].backward(&gradInputs[i]);
            }
        }

        // FIXME: reconsider this maybe
        import core.memory : GC;
        destroy(gradOutputs);
        GC.free(&gradOutputs);
        destroy(this);
        GC.free(&this);
    }
}

///
unittest {
    import std.stdio;
    UntypedVariable u;
    {
        auto v = [[0f, 1f], [2f, 3f]].variable;
        u = UntypedVariable(v);
    }
    assert(u.get!(HostStorage!float) == [0, 1, 2, 3]);
}

// TODO add SliceKind
struct Variable(T, size_t dim, alias Storage = HostStorage) {
    bool requiresGrad = false;
    size_t[dim] shape;
    ptrdiff_t[dim] strides;
    RefCounted!(Storage!T) data;
    UntypedVariable grad;
    RefCounted!BackProp bprop;

    this(bool requiresGrad, size_t[dim] shape, ptrdiff_t[dim] strides, RefCounted!(Storage!T) data) {
        this.requiresGrad = requiresGrad;
        this.shape = shape;
        this.strides = strides;
        this.data = data;
        this.grad.isHost = is(Storage!T == HostStorage!T);
    }

    auto dup() {
        RefCounted!(Storage!T) d = data.dup;
        auto y = Variable(this.requiresGrad, this.shape, this.strides, d);
        return y;
    }

    static if (is(Storage!T == HostStorage!T)) {
        auto sliced() {
            import mir.ndslice.slice : Slice, Universal;
            return Slice!(Universal, [dim], T*)(shape, strides, data.ptr);
        }
    }

    // TODO pass gradOutput
    void backward(UntypedVariable* grad=null, size_t pos=0) {
        this.bprop.backward(grad, pos);
    }

    string toString() {
        import std.format : format;
        return "Variable!(%s, dim=%d, %s)(data=%s, shape=%s)"
            .format(T.stringof, dim, Storage.stringof,
                    data, shape);
    }
}

enum bool isVariable(T) = is(T : Variable!(Elem, dim, Storage), Elem, size_t dim, alias Storage);

auto variable(Sl)(Sl sl, bool requiresGrad = false) if (isSlice!Sl) {
    import mir.ndslice : universal, DeepElementType;
    import mir.math.sum : sum;
    import numir : Ndim;
    auto s = sl.universal;
    alias S = typeof(s);
    alias E = DeepElementType!S;
    auto size = s._lengths.sum;
    RefCounted!(E[]) data = s._iterator[0..size];
    return Variable!(E, Ndim!S, HostStorage)(
        requiresGrad, s._lengths, s._strides, data);
}

auto variable(A)(A a, bool requiresGrad=false) if (isArray!A) {
    import numir.core : nparray;
    return a.nparray.variable(requiresGrad);
}

Variable!(T, dim, Dst) to(alias Dst, T, size_t dim, alias Src)(Variable!(T, dim, Src) src) {
    RefCounted!(Dst!T) d = src.data.to!Dst;
    // FIXME: consider grad
    return typeof(return)(src.requiresGrad, src.shape, src.strides, d);
}


///
unittest {
    import std.stdio;
    // Variable!(float, 1) x;
    auto x = [-1f, -2f, -3f].variable;
    auto y = x.dup;
    x.data[0] = 1.0;
    static assert(isVariable!(typeof(x)));
    static assert(!isVariable!void);
    assert(y.data[0] == -1);
}


