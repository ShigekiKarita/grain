module grain.testing;

import std.typecons  : isTuple, tuple;

auto toTuple(T)(T t) {
    static if (isTuple!T) {
        return t;
    } else {
        return tuple(t);
    }
}

auto numericGrad(F, In, Out)(F func, In inputs, Out gradOutputs, float eps) {
    import numir; // : zeros_like, view;
    import grain.autograd : variable;
    import mir.ndslice;
    import mir.math : sum;
    In gradInputs;
    foreach (n, x; inputs) {
        auto xFlat = x.sliced.view(-1);
        auto gxFlat = zeros_like(xFlat);
        foreach (i; 0 .. xFlat.length) {
            auto origin = xFlat[i];
            xFlat[i] = origin + eps;
            auto a = func.forward(inputs.toTuple.expand).toTuple;
            xFlat[i] = origin - eps;
            auto b = func.forward(inputs.toTuple.expand).toTuple;
            xFlat[i] = origin;
            foreach (m, gy; gradOutputs.toTuple) {
                auto sa = a[m].sliced; // copy?
                auto sb = b[m].sliced;
                sa[] -= sb;
                sa[] *= gy.sliced;
                gxFlat[i] += sum!"fast"(sa) / (2.0 * eps);
            }
        }
        auto gx = gxFlat.universal.view(cast(ptrdiff_t[2]) x.shape);
        gradInputs[n] = gx.variable;
    }
    return gradInputs;
}

/// gradient check function to compare numeric grad and autograd
auto gradCheck(F, In, Out)(F func, In inputs, Out gradOutputs,
                           float eps=1e-3, float rtol=1e-3, float atol=1e-5) {
    import numir.testing : approxEqual;
    auto ys = func.forward(inputs.toTuple.expand);
    auto agrad = func.backward(gradOutputs).toTuple;
    // FIXME transfer device variable to host before computing numericGrad
    auto ngrad = numericGrad(func, inputs, gradOutputs, eps).toTuple;
    static foreach (i; 0 .. ngrad.length) {
        assert(approxEqual(agrad[i], ngrad[i].sliced, rtol, atol));
    }
}


// TODO CPU-CUDA comparison function
