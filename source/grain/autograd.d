module grain.autograd;

import std.algorithm;
import std.array;

import grain.cuda;

struct Storage(T) {
    bool isHost = true;
    T[] host;
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
            device.toCPU(host);
        }
        return this;
    }
}

class Variable(T, size_t dim) {
    bool autograd = false;
    Storage!T data;
    size_t[dim] shapes;
    ptrdiff_t[dim] strides;
    Variable grad = null;
}

struct ReLU(T, size_t dim) {
    auto forward(Variable!(T, dim) x) {
        if (x.data.isHost) {
            x.data.host.each!((ref a) { if (a < 0) a = 0; });
        } else {
            import grain.kernel : relu;
            auto n = cast(uint) x.data.device.length;
            GlobalModule!"kernel".get!relu
                .launch(x.data.device.ptr, n, [1,1,1], [n,1,1]);
        }
        // return x;
    }
}


unittest {
    import std.stdio;
    import grain.kernel : relu;

    auto x = new Variable!(float, 1);
    ReLU!(float, 1) func;

    // test CPU
    x.data.host = [-1.0f, 1.0f, 0.0f];
    func.forward(x);
    assert(x.data.host == [0.0f, 1.0f, 0.0f]);
    writeln(x.data.host);

    // test CUDA
    x.data.host = [-1.0f, 1.0f, 0.0f];
    x.data.toDevice();
    assert(!x.data.isHost);
    func.forward(x);
    x.data.toHost();
    assert(x.data.host == [0.0f, 1.0f, 0.0f]);
    writeln(x.data.host);
}
