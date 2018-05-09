module grain.autograd;

import std.typecons : RefCounted;
import grain.cuda;


class DeviceNotFoundException : Exception {
    this(string file = __FILE__, size_t line = __LINE__) {
        super("device not found! (please report bug)", file, line);
    }
}

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
}

Variable!(T, dim, Dst) to(alias Dst, T, size_t dim, alias Src)(Variable!(T, dim, Src) src) {
    RefCounted!(Dst!T) d = src.data.to!Dst;
    // FIXME: consider grad
    return typeof(return)(src.autograd, d, src.shapes, src.strides, null);
}


struct ReLU(T, size_t dim) {
    bool inplace = false;

    auto forward(Variable!(T, dim, HostStorage) x) {
        import std.algorithm : each;
        auto y = this.inplace ? x : x.dup;
        y.data.each!((ref a) { if (a < 0) a = 0; });
        return y;
    }

    version(grain_cuda)
    auto forward(Variable!(T, dim, DeviceStorage) x) {
        import grain.kernel : relu;
        auto y = this.inplace ? x : x.dup;
        auto n = cast(uint) y.data.length;
        GlobalModule!"kernel".get!relu
            .launch(y.data.ptr, n, [1,1,1], [n,1,1]);
        return y;
    }
}

unittest {
    import std.stdio;
    Variable!(float, 1) x;
    x.data = [-1, -2, -3];
    auto y = x.dup;
    x.data[0] = 1.0;
    assert(y.data[0] == -1);
}

unittest {
    import std.stdio;

    foreach (inplace; [true, false]) {
        Variable!(float, 1) x;
        ReLU!(float, 1) func;
        func.inplace = inplace;

        // test CPU
        {
            x.data = [-1.0f, 1.0f, 0.0f];
            auto y = func.forward(x);
            assert(x.data == (inplace ? y.data : [-1.0f, 1.0f, 0.0f]));
            assert(y.data == [0.0f, 1.0f, 0.0f]);
        }

        // test CUDA
        version(grain_cuda) {
            x.data = [-1.0f, 1.0f, 0.0f];
            auto xd = x.to!DeviceStorage;
            auto yd = func.forward(xd);
            auto x2 = xd.to!HostStorage;
            auto y = yd.to!HostStorage;
            assert(x2.data == (inplace ? y.data : [-1.0f, 1.0f, 0.0f]));
            assert(y.data == [0.0f, 1.0f, 0.0f]);
        }
    }
}

