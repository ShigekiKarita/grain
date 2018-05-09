module grain.variable;

import std.typecons : RefCounted;
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

// TODO add SliceKind
struct Variable(T, size_t dim, alias Storage = HostStorage) {
    bool autograd = false;
    RefCounted!(Storage!T) data;
    size_t[dim] shapes;
    ptrdiff_t[dim] strides;
    Variable* grad = null;

    auto dup() {
        RefCounted!(Storage!T) d = data.dup;
        auto y = Variable(autograd, d, shapes, strides, grad);
        return y;
    }

    auto slice() {
        return Slice!(Universal, [dim], T*)(shapes, strides, data.ptr);
    }
}

import mir.ndslice;

auto variable(SliceKind kind, size_t[] packs, Iterator)(
    Slice!(kind, packs, Iterator) sl, bool autograd = false) {
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



Variable!(T, dim, Dst) to(alias Dst, T, size_t dim, alias Src)(Variable!(T, dim, Src) src) {
    RefCounted!(Dst!T) d = src.data.to!Dst;
    // FIXME: consider grad
    return typeof(return)(src.autograd, d, src.shapes, src.strides, null);
}


///
unittest {
    import std.stdio;
    Variable!(float, 1) x;
    x.data = [-1, -2, -3];
    auto y = x.dup;
    x.data[0] = 1.0;
    assert(y.data[0] == -1);
}


