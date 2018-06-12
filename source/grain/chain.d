/++
Chain means autograd operators in grain that is equivalent to
- pytorch: torch.nn.Module
- chainer: chainer.Chain or chainer.Link

Users cannot apply grain.functions to Variable without new or applyForward.
Instead of that, you can apply grain.chains to Variable with opCall.

TODO test chains as functions
 +/
module grain.chain;

import numir : normal;

import grain.autograd; // : Variable, variable, to;
version (grain_cuda) {
    import grain.cuda : zero_;
}


// enum isChain(T) = {
//     import std.traits;
//     import std.meta;
//     alias R = ReturnType!(T.init);
//     if (isVariable!R) return true;
//     if (isTuple!() AllSatisfy!(isVariable, ReturnType!(T.init));
// }();

/// linear operator
struct Linear(T, alias Storage) {
    import mir.ndslice : slice;
    import std.traits : isFloatingPoint;
    import grain.functions : MatMul, AddBias;
    static assert(isFloatingPoint!T);
    Variable!(T, 2, Storage) weight;
    Variable!(T, 1, Storage) bias;

    this(int ninput, int noutput) {
        import numir;
        import mir.random.variable;
        auto stdv = 1.0 / (cast(T) noutput ^^ 0.5);
        this.weight = UniformVariable!T(-stdv, stdv).generate(ninput, noutput).slice.variable(true).to!Storage;
        this.bias = UniformVariable!T(-stdv, stdv).generate(noutput).slice.variable(true).to!Storage;
    }

    auto opCall(Variable!(T, 2, Storage) x) {
        auto matmul = new MatMul!T;
        auto wx = matmul.applyForward(x, this.weight);
        auto addbias = new AddBias!T;
        return addbias.applyForward(wx, this.bias);
    }
}

//////// Unary functions

/// rectified linear unit nonlinearity
auto relu(T, size_t dim, alias Storage)(Variable!(T, dim, Storage) x) {
    import grain.functions : ReLU;
    auto func = new ReLU!(T, dim);
    return func.applyForward(x);
}

/// sigmoid nonlinearity
auto sigmoid(T, size_t dim, alias Storage)(Variable!(T, dim, Storage) x) {
    import grain.functions : Sigmoid;
    auto func = new Sigmoid!(T, dim);
    return func.applyForward(x);
}

/// tanh nonlinearity
auto tanh(T, size_t dim, alias Storage)(Variable!(T, dim, Storage) x) {
    import grain.functions : Tanh;
    auto func = new Tanh!(T, dim);
    return func.applyForward(x);
}

/// 1 / x
auto reciprocal(T, size_t dim, alias Storage)(Variable!(T, dim, Storage) x) {
    import grain.functions : Reciprocal;
    auto func = new Reciprocal!(T, dim);
    return func.applyForward(x);
}

/// -x
auto neg(T, size_t dim, alias Storage)(Variable!(T, dim, Storage) x) {
    import grain.functions : Neg;
    auto func = new Neg!(T, dim);
    return func.applyForward(x);
}

/// exp x
auto exp(T, size_t dim, alias Storage)(Variable!(T, dim, Storage) x) {
    import grain.functions.unary : Exp;
    auto func = new Exp!(T, dim);
    return func.applyForward(x);
}

/// log x
auto log(T, size_t dim, alias Storage)(Variable!(T, dim, Storage) x) {
    import grain.functions.unary : Log;
    auto func = new Log!(T, dim);
    return func.applyForward(x);
}

/// sin x
auto sin(T, size_t dim, alias Storage)(Variable!(T, dim, Storage) x) {
    import grain.functions.unary : Sin;
    auto func = new Sin!(T, dim);
    return func.applyForward(x);
}

/// cos x
auto cos(T, size_t dim, alias Storage)(Variable!(T, dim, Storage) x) {
    import grain.functions.unary : Cos;
    auto func = new Cos!(T, dim);
    return func.applyForward(x);
}


/// test fast math functions
unittest {
    import grain.testing;
    import numir;
    import mir.ndslice;
    import std.meta;
    foreach (f; AliasSeq!(sigmoid, tanh, reciprocal, neg, exp, log, sin, cos)) {
        auto hx = uniform!float(2, 3).slice.variable(true);
        auto hgy = uniform!float(2, 3).slice.variable;
        gradCheckChain!f(hx, hgy, 1e-3, 1e-2, 1e-2);
    }
}


/////// Loss

/// cross entropy loss (logsoftmax -> negative loglikelihood function)
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
    /* pytorch equivalent
       >>> x = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], requires_grad=True)
       >>> t = torch.tensor([1, 0, -100], dtype=torch.long)
       >>> l = torch.nn.functional.cross_entropy(x, t)
       >>> l
       tensor(0.6944)
       >>> l.backward()
       >>> x.grad
       tensor([[ 0.2375, -0.2375],
               [-0.2625,  0.2625],
               [ 0.0000,  0.0000]])
     */
    import std.stdio;
    import std.typecons;
    import mir.ndslice;
    import grain.autograd;
    import numir;

    grain.autograd.backprop = true;

    auto hx = [[0.1f, 0.2f], [0.3f, 0.4f], [0.5f, 0.6f]].variable(true);
    auto ht = [1, 0, -100].variable;
    auto hl = crossEntropy(hx, ht);
    hl.backward();
    assert(approxEqual(hx.gradSliced,
                       [[ 0.2375, -0.2375],
                        [-0.2625,  0.2625],
                        [ 0.0000,  0.0000]].nparray));

    version (grain_cuda) {
        auto dx = hx.to!DeviceStorage;
        dx.grad.zero_();
        auto dt = ht.to!DeviceStorage;
        auto dl = crossEntropy(dx, dt);
        assert(approxEqual(hl.sliced, dl.to!HostStorage.sliced));
        dl.backward();
        assert(approxEqual(dx.to!HostStorage.gradSliced, hx.gradSliced));
    }
}

/// Binary functions

/**
  op(alpha1 * a, alpha2 * b)

  this function is tested under grain.autograd.Variable.opBinary
*/
auto opBinaryFunc(string op, T, size_t dim, alias Storage)(
    Variable!(T, dim, Storage) a, Variable!(T, dim, Storage) b, T alpha1=1, T alpha2=1)
{
    import grain.functions : OpBinary;
    auto func = new OpBinary!(T, dim, op)(alpha1, alpha2);
    return func.applyForward(a, b);
}
