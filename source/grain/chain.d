/++
Chain means autograd operators in grain that is equivalent to
- pytorch: torch.nn.Module
- chainer: chainer.Chain or chainer.Link

Users cannot apply grain.functions to Variable without new or applyForward.
Instead of that, you can apply grain.chains to Variable with opCall.
 +/
module grain.chain;

import numir : normal;

import grain.autograd : Variable, variable, to;


struct Linear(T, alias Storage) {
    import mir.ndslice : slice;
    import std.traits : isFloatingPoint;
    import grain.functions : MatMul, AddBias;
    static assert(isFloatingPoint!T);
    Variable!(T, 2, Storage) weight;
    Variable!(T, 1, Storage) bias;

    this(int ninput, int noutput) {
        this.weight = normal!T(ninput, noutput).slice.variable.to!Storage;
        this.bias = normal!T(noutput).slice.variable.to!Storage;
    }

    auto opCall(Variable!(T, 2, Storage) x) {
        auto matmul = new MatMul!T;
        auto wx = matmul.applyForward(x, this.weight);
        auto addbias = new AddBias!T;
        return addbias.applyForward(wx, this.bias);
    }
}


auto relu(T, size_t dim, alias Storage)(Variable!(T, dim, Storage) x) {
    import grain.functions : ReLU;
    auto func = new ReLU!(T, dim);
    return func.applyForward(x);
}


auto crossEntropy(alias Storage)(Variable!(float, 2, Storage) x, Variable!(int, 1, Storage) t, int ignoreIndex=-100) {
    import grain.functions : LogSoftmax, NegativeLogLikelihood;
    auto lsmax = new LogSoftmax!(float, 2);
    auto y = lsmax.applyForward(x);
    auto nll = new NegativeLogLikelihood!(float, int);
    nll.ignoreIndex = ignoreIndex;
    return nll.applyForward(y, t);
}


/// test variable.backward
unittest {
    import std.stdio;
    import std.typecons;
    import mir.ndslice;
    import grain.autograd;
    import numir;

    grain.autograd.backprop = true;

    auto hx = [[0.2f, 0.4f, 0.4f], [0.1f, 0.5f, 0.4f], [0.1f, 0.5f, 0.4f]].variable;
    hx.requiresGrad = true;
    auto ht = [1, 0, -100].variable;
    auto hl = crossEntropy(hx, ht);
    auto u = UntypedVariable(1.0f.variable);
    hl.backward(&u);
    writeln(hx.gradSlice);
    // TODO hard code floats from pytorch
    // assert(hx.grad[].sliced(3, 3) == [[0.0, -0.5, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]);

    version (grain_cuda) {
        auto dx = hx.to!DeviceStorage;
        auto dt = ht.to!DeviceStorage;
        auto dl = crossEntropy(dx, dt);
        assert(approxEqual(hl.sliced, dl.to!HostStorage.sliced));
        auto du = UntypedVariable(1.0f.variable.to!DeviceStorage);
        // FIXME
        // dl.backward(&du);
    }
}
