module grain.autograd;

import std.algorithm;
import std.array;

import grain.cuda;

import std.typecons : RefCounted, refCounted;

struct Storage(T) {
    bool isHost = true;
    T[] host;
    version(grain_cuda) {
    CuPtr!T device;

    ref toDevice() {
        if (isHost) {
            isHost = false;
            device = CuPtr!T(host);
        }
        return this;
    }

    ref toHost() {
        if (!isHost) {
            isHost = true;
            device.toHost(host);
        }
        return this;
    }
    }

    auto dup() {
        version(with_cuda) {
            return isHost
                ? Storage!T(true, host.dup, CuPtr!T())
                : Storage!T(false, [], device.dup);
        } else {
            return Storage!T(true, host.dup);
        }
    }
}

struct Variable(T, size_t dim) {
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

struct ReLU(T, size_t dim) {
    bool inplace = false;

    auto forward(Variable!(T, dim) x) {
        auto y = this.inplace ? x : x.dup;
        with (y.data) {
            if (isHost) {
                host.each!((ref a) { if (a < 0) a = 0; });
            } else {
                version (with_cuda) {
                    import grain.kernel : relu;
                    auto n = cast(uint) device.length;
                    GlobalModule!"kernel".get!relu
                        .launch(device.ptr, n, [1,1,1], [n,1,1]);
                }
            }
        }
        return y;
    }
}

unittest {
    import std.stdio;
    Variable!(float, 1) x;
    x.data.host = [-1, -2, -3];
    auto y = x.dup;
    x.data.host[0] = 1.0;
    assert(y.data.host[0] == -1);
}

version(with_cuda)
unittest {
    import std.stdio;
    import grain.kernel : relu;

    foreach (inplace; [true, false]) {
        Variable!(float, 1) x;
        ReLU!(float, 1) func;
        func.inplace = inplace;

        // test CPU
        {
            x.data.host = [-1.0f, 1.0f, 0.0f];
            auto y = func.forward(x);
            assert(x.data.host == (inplace ? y.data.host : [-1.0f, 1.0f, 0.0f]));
            assert(y.data.host == [0.0f, 1.0f, 0.0f]);
        }

        // test CUDA
        {
            x.data.host = [-1.0f, 1.0f, 0.0f];
            x.data.toDevice();
            assert(!x.data.isHost);
            auto y = func.forward(x);
            x.data.toHost();
            y.data.toHost();
            assert(x.data.host == (inplace ? y.data.host : [-1.0f, 1.0f, 0.0f]));
            assert(y.data.host == [0.0f, 1.0f, 0.0f]);
        }
    }
}
