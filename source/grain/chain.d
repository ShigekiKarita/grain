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


auto crossEntropy(alias Storage)(Variable!(float, 2, Storage) x, Variable!(int, 1, Storage) t) {
    import grain.functions : LogSoftmax, NegativeLogLikelihood;
    auto lsmax = new LogSoftmax!(float, 2);
    auto y = lsmax.applyForward(x);
    auto nll = new NegativeLogLikelihood!(float, int);
    return nll.applyForward(y, t);
}
