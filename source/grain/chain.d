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
