module grain.ops;

import grain.variable;
import grain.cuda;


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

    auto backward(Variable!(T, dim, HostStorage) gy, Variable!(T, dim, HostStorage) x) {
        auto gx = gy.dup;
        foreach (i; 0..gx.data.length) {
            if (x.data[i] == 0.0) gx.data[i] = 0.0;
        }
        return gx;
    }
}

struct MatMul(T, size_t dim) {
    
}

struct SoftmaxCrossEntropy {
    
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

