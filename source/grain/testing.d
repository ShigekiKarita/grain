/**
   A module for testing
 */
module grain.testing;

import std.traits : isIntegral;
import std.typecons  : isTuple, tuple;

import grain.autograd : variable, ElementType;
import grain.utility : castArray, toTuple;
import grain.functions.common : isFunction;


/// computes numeric grad that emulates analytical differential in eps range
auto numericGrad(F, In, Out)(ref F func, In inputs, Out gradOutputs, float eps) {
    import numir; // : zeros_like, view;
    import mir.ndslice;
    import mir.math : sum;
    In gradInputs;
    foreach (n, x; inputs.toTuple) {
        static if (isIntegral!(ElementType!(typeof(x)))) {
            continue;
        } else {
            auto xFlat = x.sliced.view(-1);
            auto gxFlat = zeros_like(xFlat);
            foreach (i; 0 .. xFlat.length) {
                auto origin = xFlat[i];
                xFlat[i] = origin + eps;
                static if (isFunction!F) {
                    auto a = func.forward(inputs.toTuple.expand).toTuple;
                } else {
                    auto a = func(inputs.toTuple.expand).toTuple;
                }
                xFlat[i] = origin - eps;

                static if (isFunction!F) {
                    auto b = func.forward(inputs.toTuple.expand).toTuple;
                } else {
                   auto b = func(inputs.toTuple.expand).toTuple;
                }

                xFlat[i] = origin;
                foreach (m, gy; gradOutputs.toTuple) {
                    auto sa = a[m].sliced; // copy?
                    auto sb = b[m].sliced;
                    sa[] -= sb;
                    sa[] *= gy.sliced;
                    gxFlat[i] += sum!"fast"(sa) / (2.0 * eps);
                }
            }
            auto gx = gxFlat.universal.view(x.shape.castArray!ptrdiff_t);
            gradInputs[n] = gx.variable;
        }
    }
    return gradInputs;
}


/// computes numeric grad that emulates analytical differential in eps range
auto numericGradChain(alias func, In, Out)(In inputs, Out gradOutputs, float eps) {
    import numir; // : zeros_like, view;
    import mir.ndslice;
    import mir.math : sum;
    In gradInputs;
    foreach (n, x; inputs.toTuple) {
        static if (isIntegral!(ElementType!(typeof(x)))) {
            continue;
        } else {
            auto xFlat = x.sliced.view(-1);
            auto gxFlat = zeros_like(xFlat);
            foreach (i; 0 .. xFlat.length) {
                auto origin = xFlat[i];
                xFlat[i] = origin + eps;
                auto a = func(inputs.toTuple.expand).toTuple;
                xFlat[i] = origin - eps;
                auto b = func(inputs.toTuple.expand).toTuple;

                xFlat[i] = origin;
                foreach (m, gy; gradOutputs.toTuple) {
                    auto sa = a[m].sliced; // copy?
                    auto sb = b[m].sliced;
                    sa[] -= sb;
                    sa[] *= gy.sliced;
                    gxFlat[i] += sum!"fast"(sa) / (2.0 * eps);
                }
            }
            auto gx = gxFlat.universal.view(x.shape.castArray!ptrdiff_t);
            gradInputs[n] = gx.variable;
        }
    }
    return gradInputs;
}

/// gradient check function to compare numeric grad and autograd
auto gradCheck(F, In, Out, string file = __FILE__, size_t line = __LINE__)(
    ref F func, In inputs, Out gradOutputs,
    float eps=1e-3, float rtol=1e-3, float atol=1e-5) {
    import std.format : format;
    import numir.testing : approxEqual;
    static if (isFunction!F) {
        auto ys = func.forward(inputs.toTuple.expand).toTuple;
        auto agrad = func.backward(gradOutputs.toTuple.expand).toTuple;
    } else {
        auto xs = inputs.toTuple;
        auto ys = func(xs.expand).toTuple;
        auto gys = gradOutputs.toTuple;
        foreach (o, y; ys) {
            import grain.autograd;
            auto u = UntypedVariable(gys[o]);
            y.backward(&u);
        }
        typeof(inputs.toTuple) agrad;
        foreach (i, a; agrad) {
            agrad[i] = xs[i].gradSlice.variable; // TODO support CUDA
        }
    }
    // FIXME transfer device variable to host before computing numericGrad
    auto ngrad = numericGrad(func, inputs.toTuple, gradOutputs.toTuple, eps).toTuple;
    static foreach (i; 0 .. inputs.toTuple.length) {
        static if (!isIntegral!(ElementType!(typeof(inputs.toTuple[i])))) {
            assert(approxEqual(agrad[i].sliced, ngrad[i].sliced, rtol, atol),
                   format!"%d th input grad %s (backprop) != %s (numeric) from %s %d"(i, agrad[i].sliced, ngrad[i].sliced, file , line));
        }
    }
}


/// gradient check function to compare numeric grad and autograd
auto gradCheckChain(alias func, In, Out, string file = __FILE__, size_t line = __LINE__)
    (In inputs, Out gradOutputs, float eps=1e-3, float rtol=1e-3, float atol=1e-5) {
    import std.format : format;
    import numir.testing : approxEqual;
    auto xs = inputs.toTuple;
    auto ys = func(xs.expand).toTuple;
    auto gys = gradOutputs.toTuple;
    foreach (o, ref y; ys) {
        import grain.autograd;
        auto u = UntypedVariable(gys[o]);
        y.backward(&u);
    }
    typeof(inputs.toTuple) agrad;
    foreach (i, ref a; agrad) {
        agrad[i] = xs[i].gradSliced.variable; // TODO support CUDA
    }

    // FIXME transfer device variable to host before computing numericGrad
    auto ngrad = numericGradChain!func(inputs.toTuple, gradOutputs.toTuple, eps).toTuple;
    static foreach (i; 0 .. inputs.toTuple.length) {
        static if (!isIntegral!(ElementType!(typeof(inputs.toTuple[i])))) {
            assert(approxEqual(agrad[i].sliced, ngrad[i].sliced, rtol, atol),
                   format!"%d th input grad %s != %s from %s %d"(i, agrad[i].sliced, ngrad[i].sliced, file , line));
        }
    }
}


// TODO CPU-CUDA comparison function
