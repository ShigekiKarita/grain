module grain.ops;

import grain.variable;
import grain.cuda;


struct ReLU(T, size_t dim) {
    bool inplace = false;

    auto forward(Variable!(T, dim, HostStorage) x) {
        import mir.ndslice : each;
        auto y = this.inplace ? x : x.dup;
        y.sliced.each!((ref a) { if (a < 0) a = 0; });
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

    // TODO backward kernel
}

unittest {
    import std.stdio;
    import mir.ndslice;
    import A = std.algorithm;
    import numir;

    foreach (inplace; [true, false]) {
        ReLU!(float, 1) func;
        func.inplace = inplace;

        // test CPU
        {
            auto x = [-1.0f, 1.0f, 0.0f].variable;
            auto y = func.forward(x);
            assert(x.data == (inplace ? y.data : [-1.0f, 1.0f, 0.0f]));
            assert(y.data[0] == 0.0);
            assert(y.data[1] == 1.0);
            assert(y.data[2] == 0.0);
            // Why fail?
            // assert(y.data == [0.0f, 1.0f, 0.0f]);
        }

        // test CUDA
        version(grain_cuda) {
            auto x = [-1.0f, 1.0f, 0.0f].variable;
            auto xd = x.to!DeviceStorage;
            auto yd = func.forward(xd);
            auto x2 = xd.to!HostStorage;
            auto y = yd.to!HostStorage;
            assert(x2.data == (inplace ? y.data : [-1.0f, 1.0f, 0.0f]));
            assert(y.data == [0.0f, 1.0f, 0.0f]);
        }
    }
}


struct MatMul(T, size_t dim) {
    auto forward(Variable!(T, 2, HostStorage) x, Variable!(T, 2, HostStorage) y) {
        import lubeck : mtimes;
        return mtimes(x.sliced, y.sliced).variable(x.autograd || y.autograd);
    }
}

///
unittest {
    import std.stdio;
    import mir.ndslice;
    auto a = [[1, 3],
              [5, 7],
              [9, 11]].variable;
    auto b = [[2, 4, 6],
              [8, 10, 12]].variable;
    auto c = MatMul!(int, 2)().forward(a, b);
    assert(c.sliced == [[1*2+3*8, 1*4+3*10, 1*6+3*12],
                        [5*2+7*8, 5*4+7*10, 5*6+7*12],
                        [9*2+11*8, 9*4+11*10, 9*6+11*12]]);
}

struct SoftmaxCrossEntropy {
    
}

