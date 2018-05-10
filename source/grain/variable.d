module grain.variable;

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
    Variant data;
    size_t dim;
    size_t[] shape;
    ptrdiff_t[] strides;
    bool isHost;
    TypeInfo elem;
    UntypedVariable* grad;
    this(T, size_t dim, alias Storage)(Variable!(T, dim, Storage) v) {
        this.data = v.data;
        this.shape = v.shape;
        this.strides = v.strides;
        this.dim = dim;
        this.isHost = is(Storage!T == HostStorage!T);
        // this.grad = new UntypedVariable(*v.grad);
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
    assert(u.data.get!(RefCounted!(float[])) == [0, 1, 2, 3]);
}

// TODO add SliceKind
struct Variable(T, size_t dim, alias Storage = HostStorage) {
    bool autograd = false;
    RefCounted!(Storage!T) data;
    size_t[dim] shape;
    ptrdiff_t[dim] strides;
    Variable* grad = null;

    auto dup() {
        RefCounted!(Storage!T) d = data.dup;
        auto y = Variable(this.autograd, d, this.shape, this.strides, this.grad);
        return y;
    }

    static if (is(Storage!T == HostStorage!T)) {
        auto sliced() {
            import mir.ndslice.slice : Slice, Universal;
            return Slice!(Universal, [dim], T*)(shape, strides, data.ptr);
        }
    }
}

auto variable(Sl)(Sl sl, bool autograd = false) if (isSlice!Sl) {
    import mir.ndslice : universal, DeepElementType;
    import mir.math.sum : sum;
    import numir : Ndim;
    auto s = sl.universal;
    alias S = typeof(s);
    alias E = DeepElementType!S;
    auto size = s._lengths.sum;
    RefCounted!(E[]) data = s._iterator[0..size];
    return Variable!(E, Ndim!S, HostStorage)(
        autograd, data, s._lengths, s._strides, null);
}

auto variable(A)(A a) if (isArray!A) {
    import numir.core : nparray;
    return a.nparray.variable;
}

Variable!(T, dim, Dst) to(alias Dst, T, size_t dim, alias Src)(Variable!(T, dim, Src) src) {
    RefCounted!(Dst!T) d = src.data.to!Dst;
    // FIXME: consider grad
    return typeof(return)(src.autograd, d, src.shape, src.strides, null);
}


///
unittest {
    import std.stdio;
    // Variable!(float, 1) x;
    auto x = [-1f, -2f, -3f].variable;
    auto y = x.dup;
    x.data[0] = 1.0;
    assert(y.data[0] == -1);
}


