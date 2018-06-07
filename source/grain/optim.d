module grain.optim;


void zeroGrad(C)(ref C chain) {
    import grain.autograd : isVariable, zero_;
    import std.traits : hasMember;
    foreach (field; chain.tupleof) {
        alias F = typeof(field);
        static if (isVariable!F) {
            field.grad.zero_();
        } else static if (hasMember!(F, "tupleof")) {// static if (isChain!F) {
            field.zeroGrad();
        }
    }
}

unittest {
    import std.stdio;
    import numir;
    import grain.autograd : Variable, HostStorage;
    import grain.chain : Linear, relu;

    struct MLP(T, alias Storage) {
        alias L = Linear!(T, Storage);
        L fc1, fc2, fc3;

        this(int nhidden) {
            this.fc1 = L(2, nhidden);
            this.fc2 = L(nhidden, nhidden);
            this.fc3 = L(nhidden, 10);
        }

        auto opCall(Variable!(T, 2, Storage) x) {
            auto h1 = relu(this.fc1(x));
            auto h2 = relu(this.fc2(h1));
            auto h3 = this.fc3(h2);
            return h3;
        }
    }

    auto mlp = MLP!(float, HostStorage)(3);
    mlp.fc1.weight.grad[0] = 1.0;
    mlp.zeroGrad();
    assert(mlp.fc1.weight.grad[0] == 0.0);
}
