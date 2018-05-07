module grain.autograd;

import std.algorithm;
import std.array;

import grain.cuda;
import grain.kernel : relu;


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

struct Variable(T, size_t dim) {
    bool autograd = false;
    Storage!T data;
    size_t[dim] shapes;
    ptrdiff_t[dim] strides;
    Variable* grad = null;
}

struct ReLU(T, size_t dim) {
    Variable!(T, dim) forward(Variable!(T, dim) x) {
        if (x.data.isHost) {
            x.data.host.each!((ref a) { if (a < 0) a = 0; });
        } else {
            auto k = GlobalModule.get().kernel!relu;
            auto n = cast(uint) x.data.device.length;
            k.launch(x.data.device.ptr, n, [1,1,1], [n,1,1]);
        }
        return x;
    }
}


unittest {
    import std.stdio;
    Variable!(float, 1) x;
    x.data.host = [-1.0f, 1.0f, 0.0f];
    ReLU!(float, 1) func;
    x = func.forward(x);
    assert(x.data.host == [0.0f, 1.0f, 0.0f]);
    writeln(x.data.host);

    x.data.host = [-1.0f, 1.0f, 0.0f];
    x.data.toDevice();
    assert(!x.data.isHost);
    // x = func.forward(x);
    // x.data.toHost();
    // assert(x.data.host == [0.0f, 1.0f, 0.0f]);
    // writeln(x.data.host);
}
