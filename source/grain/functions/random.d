module grain.functions.random;

import grain.autograd;
import grain.utility : castArray;
import grain.functions.common;

/// TODO create cuRAND wrappers
struct Dropout(T, size_t dim) {
    import numir.random : generate;
    import mir.ndslice : as, slice;
    import mir.random.variable : BernoulliVariable;

    double ratio = 0.5;
    Variable!(T, dim, HostStorage) hostMask;

    this(double ratio) {
        assert(0.0 <= ratio && ratio <= 1.0);
        this.ratio = ratio;
        // version (grain_cuda) {
        //     this.impl = CudnnDropout(ratio);
        // }
    }

    auto forward(Variable!(T, dim, HostStorage) x) {
        if (this.ratio == 0.0) return x;

        const shape = x.shape.castArray!size_t;
        this.hostMask = BernoulliVariable!T(1.0 - this.ratio)
            .generate(shape).as!T.slice.variable;
        return this.hostMask * x;
    }

    auto backward(Variable!(T, dim, HostStorage) gy) {
        assert(gy.shape == this.hostMask.shape);
        return this.hostMask * gy;
    }

    // version (grain_cuda) {
    //     import grain.cudnn : CudnnDropout;
    //     CudnnDropout impl;

    //     auto forward(Variable!(T, dim, DeviceStorage) x) {
    //         return this.impl.forward(x);
    //     }

    //     auto backward(Variable!(T, dim, DeviceStorage) gy) {
    //         return this.impl.backward(gy);
    //     }
    // }

    mixin FunctionCommon;
}


unittest {
    Dropout!(float, 2) func;
    auto x = [[1f, 2f, 3f], [4f, 5f, 6f]].variable;
    auto y = func.forward(x);
    auto gx = func.backward(x);
    foreach (i; 0 .. x.shape[0]) {
        foreach (j; 0 .. x.shape[1]) {
            auto yij = y.sliced[i, j];
            assert(yij == 0 || yij == x.sliced[i, j]);
            assert(yij == gx.sliced[i, j]);
        }
    }

    // import std.stdio;
    // writeln(x);
    // auto cx = x.to!DeviceStorage;
    // {
    //     auto cy = func.forward(cx);
    //     writeln(cy.to!HostStorage);
    //     auto cgx = func.backward(cx);
    //     writeln(cgx.to!HostStorage);
    // }
    // func.impl.setRatio(0.1);
    // func.impl.setRatio(0.5);
    // {
    //     auto cy = func.forward(cx);
    //     writeln(cy.to!HostStorage);
    //     auto cgx = func.backward(cx);
    //     writeln(cgx.to!HostStorage);
    // }

}
