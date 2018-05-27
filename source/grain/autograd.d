module grain.autograd;

import std.traits : isArray;
import std.typecons : RefCounted, RefCountedAutoInitialize;
import mir.ndslice : isSlice;
import std.range : ElementType;

import grain.cuda;
import grain.utility : castArray;

alias HostStorage(T) = T[];

auto zero_(T)(T[] s) {
    import std.algorithm.mutation : fill;
    fill(s, 0);
    return s;
}

auto zeros(T)(size_t n) if (isArray!T) {
    auto s = new ElementType!T[n];
    return s.zero_();
}

unittest {
    float[] h = [1f, 2f, 3f];
    h.zero_();
    assert(h == [0f, 0f, 0f]);
    assert(zeros!(HostStorage!float)(3) == [0f, 0f, 0f]);
}

version(grain_cuda) {
    alias DeviceStorage(T) = CuPtr!T;

    enum bool isDevice(T) = is(typeof({T.init.toHost();}));


    auto to(alias S : DeviceStorage, T)(T[] src) {
        import std.array : empty;
        return src.empty ? DeviceStorage!T() : DeviceStorage!T(src);
    }

    auto to(alias S : HostStorage, Src)(Src src) if (isDevice!Src) {
        return src.toHost();
    }

    unittest {
        auto h = [[0.1f, 0.2f, 0.3f], [0.4f, 0.5f, 0.6f]].variable;
        auto d = h.to!DeviceStorage;
        assert(h.data == d.to!HostStorage.data);
    }
}


/// type-erased variable
struct UntypedVariable {
    import std.variant;
    bool requiresGrad;
    size_t dim;
    // size_t[]
    int[] shape;
    // ptrdiff_t[]
    int[] strides;
    TypeInfo elem;
    Variant data, grad;
    size_t outPosition = 0;
    RefCounted!BackProp bprop;

    this(T, size_t dim, alias Storage)(Variable!(T, dim, Storage) v) {
        this.elem = typeid(T);
        this.requiresGrad = v.requiresGrad;
        this.shape = v.shape.dup;
        this.strides = v.strides.dup;
        this.dim = dim;
        this.data = v.data;
        this.grad = v.grad;
    }

    auto get(T)() {
        return this.data.get!(RefCounted!T);
    }

    auto to(V : Variable!(T, dim, Storage), T, size_t dim, alias Storage)() {
        auto d = this.data.get!(RefCounted!(Storage!T));
        return Variable!(T, dim, Storage)(
            this.requiresGrad, this.shape[0..dim], this.strides[0..dim], d);
    }

    void backward(UntypedVariable* gradOutput=null) {
        if (bprop.refCountedStore.isInitialized) {
            bprop.backward(gradOutput, outPosition);
        }
    }

    string toString() {
        import std.format : format;
        return "UntypedVariable(%s, dim=%d, data=%s, shape=%s, strides=%s)".format(
            elem, dim, data, shape, strides);
    }

    auto gradSlice(V)() if (isVariable!V && isHost!V) {
        import mir.ndslice.slice : sliced;
        return grad.get!(typeof(V.init.data)).ptr.sliced(this.shape[0 .. Ndim!V].castArray!size_t);
    }
}

auto gradSlice(V)(V v) if (isVariable!V && isHost!V) {
    import mir.ndslice.slice : sliced;
    return v.grad.ptr.sliced(v.shape.castArray!size_t);
}



/// FIXME maybe singleton?
shared bool backprop = false;

/// Informations for backpropagation
struct BackProp {
    alias Proc = void delegate(UntypedVariable[], UntypedVariable[]);
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
            proc(this.gradOutputs, this.inputs);
        }

        // FIXME: reconsider this maybe
        // import core.memory : GC;
        // destroy(gradOutputs);
        // GC.free(&gradOutputs);
        // destroy(this);
        // GC.free(&this);
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
    // size_t[dim]
    int[dim] shape;
    // ptrdiff_t[dim]
    int[dim] strides;
    RefCounted!(Storage!T) data;
    RefCounted!(Storage!T) grad;
    RefCounted!BackProp bprop;
    enum isHost = is(Storage!T == HostStorage!T);

    this(bool requiresGrad, int[dim] shape, int[dim] strides, RefCounted!(Storage!T) data) {
        this.requiresGrad = requiresGrad;
        this.shape = shape;
        this.strides = strides;
        this.data = data;
        // this.grad.isHost = is(Storage!T == HostStorage!T);
        if (this.requiresGrad) {
            static if (is(Storage!T == HostStorage!T)) {
                this.grad = zeros!(Storage!T)(this.data.length);
            } else version (grain_cuda) {
                // TODO why is grain.cuda. required?
                this.grad = grain.cuda.zeros!(CuPtr!T)(this.data.length);
            }
        }
    }

    @property
    bool defined() { return cast(size_t) data.ptr != 0; }

    auto dup() {
        static if (is(Storage!T == HostStorage!T)) {
            RefCounted!(Storage!T) d = new T[data.length];
            d[] = data[];
        } else {
            RefCounted!(Storage!T) d = data.dup;
        }
        auto y = Variable(this.requiresGrad, this.shape, this.strides, d);
        return y;
    }

    static if (is(Storage!T == HostStorage!T)) {
        auto sliced() {
            import mir.ndslice; // .slice : Slice, Universal;
            static if (dim == 0) {
                return [this.data[0]].sliced.universal;
            } else {
                return Slice!(Universal, [dim], T*)(
                    this.shape.castArray!size_t,
                    this.strides.castArray!ptrdiff_t, data.ptr);
            }
        }
    }

    // TODO pass gradOutput
    void backward(UntypedVariable* grad=null, size_t pos=0) {
        this.bprop.backward(grad, pos);
    }

    string toString() {
        import std.format : format;
        return "Variable!(%s, dim=%d, %s)(data=%s, shape=%s, strides=%s)"
            .format(T.stringof, dim, Storage.stringof,
                    data, shape, strides);
    }
}

/// test Variable.defined
unittest {
    Variable!(float, 1, HostStorage) h;
    assert(!h.defined);
    assert(0.variable.defined);
    assert(0.1f.variable.defined);
    assert([0].variable.defined);
    assert([0.1f].variable.defined);

    version (grain_cuda) {
        Variable!(float, 1, DeviceStorage) d;
        assert(!d.defined);
        assert(!h.to!DeviceStorage.defined);
        assert(0.variable.to!DeviceStorage.defined);
        assert(0.1f.variable.to!DeviceStorage.defined);
        assert([0].variable.to!DeviceStorage.defined);
        assert([0.1f].variable.to!DeviceStorage.defined);
    }
}

enum bool isVariable(T) = is(T : Variable!(Elem, dim, Storage), Elem, size_t dim, alias Storage);
enum bool isHost(V : Variable!(Elem, dim, Storage), Elem, size_t dim, alias Storage) = is(Storage!Elem == HostStorage!Elem);
enum size_t Ndim(V : Variable!(Elem, dim, Storage), Elem, size_t dim, alias Storage) = dim;
alias ElementType(V : Variable!(Elem, dim, Storage), Elem, size_t dim, alias Storage) = Elem;


auto variable(Sl)(Sl sl, bool requiresGrad = false) if (isSlice!Sl) {
    import mir.ndslice : universal, DeepElementType;
    import std.algorithm : reduce;

    import numir : Ndim;
    auto s = sl.universal;
    alias S = typeof(s);
    alias E = DeepElementType!S;
    auto size = s._lengths.reduce!"a * b";
    RefCounted!(E[]) data = s._iterator[0..size];
    int[Ndim!S] shape, strides;
    static foreach (i; 0 .. Ndim!S) {
        assert(s._lengths[i] < int.max);
        assert(s._strides[i] < int.max);
        shape[i] = cast(int) s.length!i;
        strides[i] = cast(int) s._strides[i];
    }
    return Variable!(E, Ndim!S, HostStorage)(
        requiresGrad, shape, strides, data);
}

import std.traits : isNumeric;
auto variable(alias Storage=HostStorage, bool requiresGrad=false, T)(T x) if (isNumeric!T) {
    RefCounted!(T[]) data = [x];
    return Variable!(T, 0, Storage)(requiresGrad, [], [], data);
}

auto variable(A)(A a, bool requiresGrad=false) if (isArray!A) {
    import numir.core : nparray;
    return a.nparray.variable(requiresGrad);
}

///
unittest {
    auto h = 0.5f.variable;
    auto d = h.to!DeviceStorage;
    assert(d.to!HostStorage.data == h.data);
}

Variable!(T, dim, Dst) to(alias Dst, T, size_t dim, alias Src)(Variable!(T, dim, Src) src) {
    RefCounted!(Dst!T) d = src.data.to!Dst;
    // FIXME: consider grad
    return typeof(return)(src.requiresGrad, src.shape, src.strides, d);
}


///
unittest {
    import std.stdio;
    {
        // Variable!(float, 1) x;
        auto x = [-1f, -2f, -3f].variable;
        auto y = x.dup;
        x.data[0] = 1.0;
        static assert(isVariable!(typeof(x)));
        static assert(!isVariable!void);
        static assert(isHost!(typeof(x)));
        assert(y.data[0] == -1);
    }
    version (grain_cuda) {
        {
            auto x = [[1f, 3f],
                      [5f, 7f],
                      [9f, 11f]].variable;

            assert(x.data.length == 6);
            static assert(!isHost!(typeof(x.to!DeviceStorage)));
            auto xx = x.dup;
            assert(x.to!DeviceStorage.to!HostStorage.sliced == x.sliced);
        }
    }
}


