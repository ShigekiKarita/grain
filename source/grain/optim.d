module grain.optim;

import grain.autograd : isVariable, zero_, isHost;
import std.traits : hasMember;
import std.stdio;



void zeroGrad(C)(ref C chain) {
    foreach (ref field; chain.tupleof) {
        alias F = typeof(field);
        static if (isVariable!F) {
            field.grad.zero_();
        } else static if (hasMember!(F, "tupleof")) {// static if (isChain!F) {
            field.zeroGrad();
        }
    }
}

struct SGD {
    float lr = 1.0;
    // float momentum = 0.0;
    // float weightDecay = 0.0;

    void update(C)(ref C chain) {
        import grain.autograd;
        foreach (ref field; chain.tupleof) {
            alias F = typeof(field);
            static if (isVariable!F) {
                static if (isHost!F) {
                    field.sliced[] += this.lr * field.gradSliced[];
                } else {
                    import grain.cuda : axpy;
                    axpy(field.grad, field.data, this.lr);
                }
            } else static if (hasMember!(F, "tupleof")) {// static if (isChain!F) {
                this.update(field);
            }
        }
    }
}

unittest {
    import std.stdio;
    import numir;
    import grain.autograd; // : Variable, HostStorage;
    import grain.chain : Linear, relu;

    struct MLP(T, alias Storage) {
        alias L = Linear!(T, Storage);
        L fc1, fc2, fc3;

        this(int nhidden) {
            this.fc1 = L(2, nhidden);
            // this.fc2 = L(nhidden, nhidden);
            // this.fc3 = L(nhidden, 10);
        }

        auto opCall(Variable!(T, 2, Storage) x) {
            auto h1 = relu(this.fc1(x));
            // auto h2 = relu(this.fc2(h1));
            // auto h3 = this.fc3(h2);
            return h1;
        }
    }

    {
        auto mlp = MLP!(float, HostStorage)(3);
        mlp.fc1.weight.grad[0] = 1.0;
        mlp.zeroGrad();
        assert(mlp.fc1.weight.grad[0] == 0.0);

        auto sgd = SGD(0.5);
        mlp.fc1.weight.data.zero_();
        mlp.fc1.weight.grad = [[1.0f, 0.0f, 0.0f], [0.0f, 0.0f, 0.0f]].variable.data;
        sgd.update(mlp);
        assert(mlp.fc1.weight.sliced == [[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    }
    version (grain_cuda) {
        auto mlp = MLP!(float, DeviceStorage)(3);
        mlp.fc1.weight.grad = [[1.0f, 0.0f, 0.0f], [0.0f, 0.0f, 0.0f]].variable.to!DeviceStorage.data;
        mlp.zeroGrad();
        assert(mlp.fc1.weight.to!HostStorage.gradSliced == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);

        auto sgd = SGD(0.5);
        mlp.fc1.weight.data.zero_();
        mlp.fc1.weight.grad = [[1.0f, 0.0f, 0.0f], [0.0f, 0.0f, 0.0f]].variable.to!DeviceStorage.data;
        sgd.update(mlp);
        assert(mlp.fc1.weight.to!HostStorage.sliced == [[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    }
}
