module grain.functions.random;

import grain.autograd;
import grain.utility : castArray;
import grain.functions.common;

/// TODO create cuRAND wrappers
struct Dropout(T, size_t dim) {
    import numir.random : generate;
    import mir.ndslice : as, slice;
    import mir.random.variable : BernoulliVariable;

    float ratio = 0.5;
    Variable!(T, dim, HostStorage) hostMask;

    this(double ratio) {
        assert(0.0 <= ratio && ratio <= 1.0);
        this.ratio = ratio;
    }

    auto forward(Variable!(T, dim, HostStorage) x) {
        if (this.ratio == 0.0) return x;

        import mir.ndslice; //  : universal;
        const shape = x.shape.castArray!size_t;
        const float survived = 1.0 - this.ratio;
        const float scale = 1.0f / (1.0f - survived);
        auto mask = BernoulliVariable!T(survived).generate(shape).as!T.slice.universal;
        mask[] *= scale;
        this.hostMask = mask.variable;
        return this.hostMask * x;
    }

    auto backward(Variable!(T, dim, HostStorage) gy) {
        assert(gy.shape == this.hostMask.shape);
        return this.hostMask * gy;
    }

    version (grain_cuda) {
        import grain.cudnn : CudnnDropout;
        CudnnDropout impl;

        auto forward(Variable!(T, dim, DeviceStorage) x) {
            return this.impl.forward(x, this.ratio);
        }

        auto backward(Variable!(T, dim, DeviceStorage) gy) {
            return this.impl.backward(gy);
        }
    }

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
            assert(yij == 0 || yij == 2.0 * x.sliced[i, j]);
            assert(yij == gx.sliced[i, j]);
        }
    }

    version (grain_cuda) {
        auto cx = x.to!DeviceStorage;
        auto cy = func.forward(cx).to!HostStorage;
        auto cgx = func.backward(cx).to!HostStorage;

        foreach (i; 0 .. x.shape[0]) {
            foreach (j; 0 .. x.shape[1]) {
                auto yij = cy.sliced[i, j];
                assert(yij == 0 || yij == 2.0 * x.sliced[i, j]);
                assert(yij == cgx.sliced[i, j]);
            }
        }
    }
}
