module grain.functions.reduce;

import grain.autograd;
import grain.cuda;
import grain.functions.common;
import grain.utility : toTuple, fromTuple, castArray;

struct Sum(string mode = "fast", T, size_t dim) {
    import std.traits : isFloatingPoint;
    static assert(isFloatingPoint!T, "currently only float point is supported.");
    mixin FunctionCommon;

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
}


unittest {
    import mir.ndslice;
    import mir.math;
    auto x = [1f, 2f, 3f, 4f].sliced(2, 2).variable;
    Sum!("fast", float, 2) func;
    auto y = func.forward(x);
    assert(y == 10f.variable);
    assert(func.backward(1.2f.variable) == [1.2f, 1.2f, 1.2f, 1.2f].sliced(2, 2).variable);
}
