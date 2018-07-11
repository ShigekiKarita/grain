/++
Chain means autograd operators in grain that is equivalent to
- pytorch: torch.nn.Module
- chainer: chainer.Chain or chainer.Link

Users cannot apply grain.functions to Variable without new or applyForward.
Instead of that, you can apply grain.chains to Variable with opCall.

TODO test chains as functions
 +/
module grain.chain;

import std.traits : isFloatingPoint;
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

/// tan x
auto tan(T, size_t dim, alias Storage)(Variable!(T, dim, Storage) x) {
    import grain.functions.unary : Tan;

    auto func = new Tan!(T, dim);
    return func.applyForward(x);
}

/// abs
auto  abs(T, size_t dim, alias Storage)(Variable!(T, dim, Storage) x) {
    import grain.functions.unary : Abs;

    auto func = new Abs!(T, dim);
    return func.applyForward(x);
}

/// pow
auto pow(T, size_t dim, alias Storage)(Variable!(T, dim, Storage) x, T power) {
    import grain.functions.unary : Pow;

    auto func = new Pow!(T, dim)(power);
    return func.applyForward(x);
}

/// log exp(x_i) / sum_i (exp(x_i))
auto logSoftmax(T, size_t dim, alias Storage)(Variable!(T, dim, Storage) x) {
    import grain.functions.unary : LogSoftmax;

    auto func = new LogSoftmax!(T, dim);
    return func.applyForward(x);
}

/// test fast math functions
unittest {
    import grain.testing;
    import numir;
    import mir.ndslice;
    import std.meta;

    foreach (f; AliasSeq!(sigmoid, tanh, reciprocal, neg, exp, log, sin, cos,
            tan, x => pow(x, 2.0f), logSoftmax)) {
        auto hx = uniform!float(2, 3).slice.variable(true);
        auto hgy = uniform!float(2, 3).slice.variable;
        gradCheckChain!f(hx, hgy, 1e-3, 5e-2, 5e-2);
    }
}

/////// Loss

/// negative loglikelihood - log p(x). note that p(x) should be normalized
auto negativeLogLikelihood(alias Storage)(Variable!(float, 2, Storage) x,
        Variable!(int, 1, Storage) t, int ignoreIndex = -100) {
    import grain.functions : NegativeLogLikelihood;

    auto nll = new NegativeLogLikelihood!(float, int);
    nll.ignoreIndex = ignoreIndex;
    return nll.applyForward(x, t);
}

/// cross entropy loss (logsoftmax -> negative loglikelihood function)
auto crossEntropy(alias Storage)(Variable!(float, 2, Storage) x,
        Variable!(int, 1, Storage) t, int ignoreIndex = -100) {
    import grain.functions : NegativeLogLikelihood;

    auto y = logSoftmax(x);
    return negativeLogLikelihood(y, t, ignoreIndex);
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
    assert(approxEqual(hx.gradSliced, [[0.2375, -0.2375], [-0.2625, 0.2625],
            [0.0000, 0.0000]].nparray));

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
auto opBinaryFunc(string op, T, size_t dim, alias Storage)(Variable!(T, dim,
        Storage) a, Variable!(T, dim, Storage) b, T alpha1 = 1, T alpha2 = 1) {
    import grain.functions : OpBinary;

    auto func = new OpBinary!(T, dim, op)(alpha1, alpha2);
    return func.applyForward(a, b);
}

/// matrix x matrix multiplication
auto matMul(T, alias Storage)(Variable!(T, 2, Storage) a, Variable!(T, 2, Storage) b) {
    import grain.functions : MatMul;

    auto func = new MatMul!T;
    return func.applyForward(a, b);
}

/// matrix + vector row-wise addition. TODO replace this with broadcasted addition
auto addVec(T, alias Storage)(Variable!(T, 2, Storage) a, Variable!(T, 1, Storage) b) {
    import grain.functions : AddBias;

    auto func = new AddBias!T;
    return func.applyForward(a, b);
}

/// tensor convolution (do cross correlation default)
auto convolution(bool isConv = false, bool isNchw = true, T, size_t dim, size_t imDim,
        alias Storage)(Variable!(T, dim, Storage) x, Variable!(T, dim,
        Storage) w, int[imDim] stride, int[imDim] pad, int[imDim] dilation) {
    static assert(dim == imDim + 2);
    import grain.functions : Convolution;

    static assert(dim > 2); // TODO support 1d, 2d inputs
    auto func = new Convolution!(T, dim - 2, isConv, isNchw);
    func.stride = stride;
    func.pad = pad;
    func.dilation = dilation;
    return func.applyForward(x, w);
}

////// Parametric chains

/// convolution operator
struct Convolution(T, size_t dim, alias Storage) {
    import grain.utility : castArray;
    import mir.ndslice : slice;

    Variable!(T, dim + 2, Storage) weight;
    Variable!(T, dim + 2, Storage) bias; // TODO implement unsqueeze
    int nInput, nOutput;
    bool useBias = true;
    int[dim] kernel, stride, pad, dilation;

    auto outShape(uint[dim + 2] inShape) {
        import grain.functions;

        auto func = grain.functions.Convolution!(T, dim)(this.stride, this.pad, this
                .dilation);
        return func.outShape(inShape, this.weight.shape);
    }

    ///
    this(int nInput, int nOutput, int[dim] kernel, bool useBias = true) {
        this.nInput = nInput;
        this.nOutput = nOutput;
        this.kernel = kernel;
        this.stride[] = 1;
        this.pad[] = 0;
        this.dilation[] = 1;
        this.useBias = useBias;
        this.resetParameters();
    }

    ///
    this(int nInput, int nOutput, int[dim] kernel, int[dim] stride, int[dim] pad,
            int[dim] dilation, bool useBias = true) {
        this.nInput = nInput;
        this.nOutput = nOutput;
        this.kernel = kernel;
        this.stride = stride;
        this.pad = pad;
        this.dilation = dilation;
        this.useBias = useBias;
        this.resetParameters();
    }

    /// pytorch style init (LeCun uniform init)
    void resetParameters() {
        import std.algorithm : reduce;
        import numir : generate;
        import mir.random.variable : UniformVariable;

        // TODO create initializer
        const receptiveSize = this.kernel.reduce!"a * b";
        const fanIn = this.nInput * receptiveSize;
        const fanOut = this.nOutput * receptiveSize;
        auto stdv = 1.0 / (cast(T) fanIn ^^ 0.5);
        int[dim + 2] wshape;
        wshape[0] = this.nOutput;
        wshape[1] = this.nInput;
        wshape[2 .. $] = this.kernel;
        this.weight = UniformVariable!T(-stdv, stdv)
            .generate(wshape.castArray!size_t).slice.variable(true).to!Storage;
        if (this.useBias) {
            int[dim + 2] bshape;
            bshape[] = 1;
            bshape[1] = this.nOutput;
            this.bias = UniformVariable!T(-stdv, stdv).generate(
                    bshape.castArray!size_t).slice.variable(true).to!Storage;
        }
    }

    ///
    auto opCall(Variable!(T, dim + 2, Storage) x) {
        auto wx = convolution(x, this.weight, this.stride, this.pad, this
                .dilation);
        if (this.useBias) {
            return wx + this.bias;
        }
        else {
            return wx;
        }
    }
}

///
unittest {
    import grain.testing;
    import grain.utility;
    import numir;
    import mir.ndslice;

    auto conv = Convolution!(float, 2, HostStorage)(3, 4, [3, 3]);
    auto x = uniform!float(2, 3, 4, 4).slice.variable(true);
    auto y = conv(x);
    auto gy = uniform!float(y.shape.castArray!size_t).slice.variable;
    gradCheckChain!conv(x, gy, 1e-3, 5e-2, 5e-2);
}

/// linear operator
struct Linear(T, alias Storage) if (isFloatingPoint!T) {
    import mir.ndslice : slice;

    Variable!(T, 2, Storage) weight;
    Variable!(T, 1, Storage) bias;
    int nInput, nOutput;
    bool useBias = true;

    this(int nInput, int nOutput, bool useBias = true) {
        this.nInput = nInput;
        this.nOutput = nOutput;
        this.useBias = useBias;
        this.resetParameters();
    }

    // pytorch style init (LeCun uniform init)
    void resetParameters() {
        import numir : generate;
        import mir.random.variable : UniformVariable;

        auto stdv = 1.0 / (cast(T) this.nOutput ^^ 0.5);
        this.weight = UniformVariable!T(-stdv, stdv).generate(this.nInput,
                this.nOutput).slice.variable(true).to!Storage;
        if (this.useBias) {
            this.bias = UniformVariable!T(-stdv, stdv).generate(this.nOutput)
                .slice.variable(true).to!Storage;
        }
    }

    auto opCall(Variable!(T, 2, Storage) x) {
        auto wx = matMul(x, this.weight);
        if (this.useBias) {
            return addVec(wx, this.bias);
        }
        else {
            return wx;
        }
    }
}

///
unittest {
    import grain.testing;
    import grain.utility;
    import numir;
    import mir.ndslice;

    auto f = Linear!(float, HostStorage)(2, 3);
    auto x = uniform!float(2, 2).slice.variable(true);
    auto y = f(x);
    auto gy = uniform!float(y.shape.castArray!size_t).slice.variable;
    gradCheckChain!f(x, gy, 1e-3, 5e-2, 5e-2);
}

/// Emebedding ID into vector
struct Embedding(T, alias Storage) if (isFloatingPoint!T) {
    import mir.ndslice : slice;

    Variable!(T, 2, Storage) weight;
    uint nVocab, nEmbed;

    this(uint nVocab, uint nEmbed) {
        this.nVocab = nVocab;
        this.nEmbed = nEmbed;
        this.resetParameters();
    }

    void resetParameters() {
        import numir : normal;

        this.weight = normal!T(this.nVocab, this.nEmbed).slice.variable(true).to!Storage;
    }

    auto opCall(Variable!(int, 1, Storage) ids) {
        import grain.functions;

        auto func = new grain.functions.Embedding!T;
        return func.applyForward(this.weight, ids);
    }
}

//// Topology functions

/// reorganizing shape while it hold total elements a.k.a. reshape.
/// At most one dimension of the new shape can be -1.
/// In this case, the value is inferred from the size of the tensor and the remaining dimensions.
auto view(T, size_t sdim, size_t tdim, alias Storage)(Variable!(T, sdim,
        Storage) x, ptrdiff_t[tdim] shape...) {
    import grain.functions;

    auto func = new grain.functions.View!(T, sdim, tdim, Storage)(shape);
    return func.applyForward(x);
}

///
unittest {
    import numir;
    import mir.ndslice;
    import grain.testing;

    auto hx = uniform!float(6, 4).slice.variable(true);
    auto hgy = uniform!float(2, 3, 4).slice.variable;
    auto ugy = UntypedVariable(hgy);
    auto hy = hx.view([2, 3, -1]);
    assert(hy.sliced == numir.view(hx.sliced, [2, 3, -1]));
    hy.backward(&ugy);
    assert(hx.gradSliced == numir.view(hgy.sliced, [6, 4]));
    // gradCheckChain!(x => x.view([2, 3, -1]))(hx, hgy, 1e-3, 5e-2, 5e-2);
}


/// squeeze/remove redundant size-1 dimension (axis) d
auto squeeze(size_t d, T, size_t dim, alias Storage)(Variable!(T, dim, Storage) x) {
    import grain.utility : castArray;
    static assert(dim >= 1);
    assert(x.shape[d] == 1);
    ptrdiff_t[dim-1] s;
    s[0..d] = x.shape[0..d].castArray!ptrdiff_t;
    s[d..$] = x.shape[d+1..$].castArray!ptrdiff_t;
    return x.view(s);
}

///
unittest {
    import mir.ndslice;
    auto x = iota(3, 4, 1, 5).as!double.slice.variable;
    assert(x.squeeze!2.shape == [3, 4, 5]);
}


/// unsqueeze/add redundant size-1 dimension (axis) d
auto unsqueeze(size_t d, T, size_t dim, alias Storage)(Variable!(T, dim, Storage) x) {
    import grain.utility : castArray;
    static assert(dim >= d);
    ptrdiff_t[dim+1] s;
    s[0..d] = x.shape[0..d].castArray!ptrdiff_t;
    s[d] = 1;
    s[d+1..$] = x.shape[d..$].castArray!ptrdiff_t;
    return x.view(s);
}

///
unittest {
    import mir.ndslice;
    auto x = iota(3, 4, 5).as!double.slice.variable;
    assert(x.unsqueeze!2.shape == [3, 4, 1, 5]);
}
