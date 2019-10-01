module grain.autograd;

import std.stdio;
import mir.ndslice;


struct AnyVar {
    void[] data;
    uint[] lengths;
    uint[] strides;
}

struct Var(T, size_t dim) {
    AnyVar payload;
    alias payload this;
}

abstract class AnyOp {
    abstract AnyVar[] forwardAny(AnyVar[]);
}

abstract class Op(Impl) : AnyOp {
    override AnyVar[] forwardAny(AnyVar[] xs) {
        return xs;
    }
}

class ReLU : Op!ReLU {
}

unittest
{
    writeln("Edit source/app.d to start your project.");
}
