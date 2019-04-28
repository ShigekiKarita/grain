/**
   A module for reductions functions
*/
module grain.functions.reduction;

import grain.autograd;
import grain.cuda;
import grain.functions.common;
import grain.utility : toTuple, fromTuple, castArray;

/// sum to scalar. mode is similar to mir.math.sum
struct Sum(string mode = "fast", T, size_t dim) {
    import std.traits : isFloatingPoint;
    static assert(isFloatingPoint!T, "currently only float point is supported.");

    uint[dim] shape;

    auto forward(Variable!(T, dim, HostStorage) x) {
        import mir.math : sum;

        // TODO if train
        this.shape = x.shape;
        auto result = x.sliced.sum!mode;
        return result.variable;
    }

    auto backward(Variable!(T, 0, HostStorage) y) {
        auto gx = uninitVariable!T(this.shape, y.requiresGrad);
        import std.algorithm : fill;
        fill(gx.data, y.data[0]);
        return gx;
    }

    version (grain_cuda) {
        auto forward(Variable!(T, dim, DeviceStorage) x) {
            import std.algorithm : reduce;
            import grain.cuda : sum, sumNaive;

            this.shape = x.shape;
            // auto y = CuPtr!float([0]);
            // Global.kernel!sum.call(x.data.ptr, y.ptr, cast(int) x.data.length)
            //     .launch(cast(uint[3]) [1U,1,1], cast(uint[3]) [1U,1,1], 0U);
            // checkCudaErrors(cuCtxSynchronize());
            return x.data.sumNaive.variable.to!DeviceStorage;
        }

        auto backward(Variable!(T, 0, DeviceStorage) y) {
            auto gx = uninitVariable!(T, DeviceStorage)(this.shape, y.requiresGrad);
            gx.data.fill_(y.data.toHost[0]);
            return gx;
        }
    }

    mixin FunctionCommon;
}

///
unittest {
    import mir.ndslice;
    import mir.math;
    auto x = [1f, 2f, 3f, 4f].sliced(2, 2).variable;
    Sum!("fast", float, 2) func;
    auto y = func.forward(x);
    assert(y == 10f.variable);
    assert(func.backward(1.2f.variable) == [1.2f, 1.2f, 1.2f, 1.2f].sliced(2, 2).variable);

    version (grain_cuda) {
        auto cx = x.to!DeviceStorage;
        auto cy = func.forward(cx).to!HostStorage;
        assert(cy == 10f.variable);
        auto cgx = func.backward(1.2f.variable.to!DeviceStorage).to!HostStorage;
        assert(cgx.sliced == [1.2f, 1.2f, 1.2f, 1.2f].sliced(2, 2));
    }
}
