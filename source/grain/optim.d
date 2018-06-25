/**
   A module for gradient descent optimizer
 */
module grain.optim;

import std.stdio;
import grain.autograd : isVariable, zero_, isHost, UntypedVariable, Variable, HostStorage, iterVariables;
import std.traits : hasMember;
import std.stdio;
version (grain_cuda) {
    import grain.cuda : zero_;
    import grain.cudnn : transform;
}


/// fill gradient arrays with zero
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

enum bool isOptimizer(T) = is(typeof({
            import grain.autograd;
            Variable!(float, 2) v;
            T.init.step("", v);
        }));

alias StateDict = UntypedVariable[string];

void update(O)(ref O optimizer) { // if (isOptimizer!O) {
    iterVariables!( (k, v) {optimizer.step(k, v);} )(optimizer.target, "");
}

void transform(T, size_t dim)(Variable!(T, dim, HostStorage) src, ref Variable!(T, dim, HostStorage) dst, T alpha=1, T beta=0) {
    if (beta == 0) {
        dst.sliced[] = alpha * src.sliced;
        return;
    }
    if (beta != 1) dst.sliced[] = beta * dst.sliced;
    dst.sliced[] += alpha * src.sliced;
}


/// stochastic gradient descent optimizer
struct SGD(Chain) {
    Chain* target;
    float lr = 1.0;
    // float momentum = 0.0;
    // float weightDecay = 0.0;
    this(ref Chain target, float lr=1.0) {
        this.target = &target;
        this.lr = lr;
    }

    void step(V)(string name, ref V field) if (isVariable!V) {
        // transform(field.gradVariable, field, -this.lr, 1.0);

        // FIXME : this code is much faster than above (250fps -> 300fps in example/mnist.d)
        static if (isHost!V) {
            field.sliced[] -= this.lr * field.gradSliced[];
        } else {
            import grain.cuda : axpy;
            axpy(field.grad, field.data, -this.lr);
        }
    }
}

version (unittest) {
    struct MLP(T, alias Storage) {
        import grain.autograd : Variable;
        import grain.chain : Linear, relu;

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
            return h1;
        }
    }
}


///
unittest {
    import std.stdio;
    import numir;
    import grain.autograd; // : Variable, HostStorage;

    {
        auto mlp = MLP!(float, HostStorage)(3);
        mlp.fc1.weight.grad[0] = 1.0;
        mlp.zeroGrad();
        assert(mlp.fc1.weight.grad[0] == 0.0);

        auto sgd = SGD!(typeof(mlp))(mlp, 0.5);
        mlp.fc1.weight.data.zero_();
        mlp.fc1.weight.grad = [[1.0f, 0.0f, 0.0f], [0.0f, 0.0f, 0.0f]].variable.data;
        sgd.update();
        assert(mlp.fc1.weight.sliced == [[-0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    }
    version (grain_cuda) {
        auto mlp = MLP!(float, DeviceStorage)(3);
        mlp.fc1.weight.grad = [[1.0f, 0.0f, 0.0f], [0.0f, 0.0f, 0.0f]].variable.to!DeviceStorage.data;
        mlp.zeroGrad();
        assert(mlp.fc1.weight.to!HostStorage.gradSliced == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);

        auto sgd = SGD!(typeof(mlp))(mlp, 0.5);
        mlp.fc1.weight.data.zero_();
        mlp.fc1.weight.grad = [[1.0f, 0.0f, 0.0f], [0.0f, 0.0f, 0.0f]].variable.to!DeviceStorage.data;
        sgd.update();
        assert(mlp.fc1.weight.to!HostStorage.sliced == [[-0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    }
}


/// http://jmlr.org/papers/v12/duchi11a.html
struct AdaGrad(Chain) {
    import grain.autograd;

    Chain* target;
    float lr = 1.0;
    float eps = 1e-8;
    StateDict memory;

    this(ref Chain target, float lr=1e-3, float eps=1e-8) {
        this.target = &target;
        this.lr = lr;
        this.eps = eps;
        iterVariables!((k, v) { this.initStates(k, v); })(this.target);
    }

    void initStates(V)(string name, ref V field) if (isVariable!V) {
        if (name !in this.memory) {
            auto m = field.uninit();
            m.data.zero_();
            this.memory[name] = UntypedVariable(m);
        }
    }

    void step(V)(string name, ref V field) if (isVariable!V) {
        import grain.chain : pow;

        auto m = memory[name].to!V;
        auto g = field.gradVariable;
        auto mn = m + g * g;
        auto diff = g / pow(mn + this.eps, 0.5); // TODO implement sqrt
        memory[name] = UntypedVariable(mn);
        transform(diff, field, -this.lr, 1.0);
    }
}

///
unittest {
    import grain.autograd;
    import numir;
    {
        float lr = 0.1;
        float eps = 1e-8;
        auto model = MLP!(float, HostStorage)(3);
        auto optim = AdaGrad!(typeof(model))(model, lr, eps);
        static assert(isOptimizer!(typeof(optim)));
        model.fc1.weight.data.zero_();
        model.fc1.weight.grad = [[0.2f, 0.0f, 0.0f], [0.0f, 0.0f, 0.0f]].variable.data;
        optim.update();
        auto w = model.fc1.weight;
        assert(approxEqual(w.sliced, [[-lr * 0.2 / (0.2 * 0.2 + eps) ^^ 0.5, 0.0, 0.0], [0.0, 0.0, 0.0]].nparray));
        auto m = optim.memory[".fc1.weight"].to!(typeof(w));
        assert(approxEqual(m.sliced, [[0.2 * 0.2, 0.0, 0.0], [0.0, 0.0, 0.0]].nparray));
    }
    version (grain_cuda) {
        auto model = MLP!(float, DeviceStorage)(3);
        auto optim = AdaGrad!(typeof(model))(model, 0.1);
        optim.update();
    }
}



/// https://arxiv.org/pdf/1412.6980v8.pdf
struct Adam(Chain) {
    import grain.autograd;

    Chain* target;
    float lr = 1.0;
    float beta1 = 0.9;
    float beta2 = 0.999;
    float eps = 1e-8;

    StateDict moment1, moment2;

    this(ref Chain target, float lr, float eps=1e-8) {
        this.target = &target;
        this.lr = lr;
        this.eps = eps;
        iterVariables!((k, v) { this.initStates(k, v); })(this.target);
    }

    void initStates(V)(string name, ref V field) if (isVariable!V) {
        if (name !in this.moment1) {
            auto m = field.uninit();
            m.data.zero_();
            this.moment1[name] = UntypedVariable(m);
        }
        if (name !in this.moment2) {
            auto m = field.uninit();
            m.data.zero_();
            this.moment2[name] = UntypedVariable(m);
        }
    }

    void step(V)(string name, ref V field) if (isVariable!V) {
        import grain.chain : pow;

        auto g = field.gradVariable;
        auto m1 = this.moment1[name].to!V;
        auto m2 = this.moment1[name].to!V;
        auto nextM1 = (1.0 - this.beta1) * (g - m1) + m1;
        auto nextM2 = (1.0 - this.beta2) * (g * g - m2) + m2;
        auto diff = nextM1 / pow(nextM2 + this.eps, 0.5); // TODO implement sqrt
        this.moment1[name] = UntypedVariable(nextM1);
        this.moment2[name] = UntypedVariable(nextM2);
        transform(diff, field, -this.lr, 1.0);
    }
}

///
unittest {
    import grain.autograd;
    import numir;
    {
        auto model = MLP!(float, HostStorage)(3);
        auto optim = Adam!(typeof(model))(model, 1e-3);
        static assert(isOptimizer!(typeof(optim)));
        model.fc1.weight.data.zero_();
        model.fc1.weight.grad = [[0.2f, 0.0f, 0.0f], [0.0f, 0.0f, 0.0f]].variable.data;
        optim.update();
        auto w = model.fc1.weight;
        auto m1 = (1.0 - optim.beta1) * (0.2 - 0.0) + 0.0;
        auto m2 = (1.0 - optim.beta2) * (0.2 * 0.2 - 0.0) + 0.0;
        assert(approxEqual(w.sliced, [[-optim.lr * m1 / (m2 + optim.eps) ^^ 0.5, 0.0, 0.0], [0.0, 0.0, 0.0]].nparray));
        auto m1_ = optim.moment1[".fc1.weight"].to!(typeof(w));
        assert(approxEqual(m1_.sliced, [[m1, 0.0, 0.0], [0.0, 0.0, 0.0]].nparray));
        auto m2_ = optim.moment2[".fc1.weight"].to!(typeof(w));
        assert(approxEqual(m2_.sliced, [[m2, 0.0, 0.0], [0.0, 0.0, 0.0]].nparray));
    }
    version (grain_cuda) {
        auto model = MLP!(float, DeviceStorage)(3);
        auto optim = Adam!(typeof(model))(model, 0.1);
        optim.update();
    }
}


/// http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf
struct AdaDelta(Chain) {
    import grain.autograd;

    Chain* target;
    float lr = 1.0;
    float rho = 0.95;
    float eps = 1e-6;

    StateDict den, num;

    this(ref Chain target, float lr=1.0, float rho=0.95, float eps=1e-8) {
        this.target = &target;
        this.lr = lr;
        this.rho = rho;
        this.eps = eps;
        iterVariables!((k, v) { this.initStates(k, v); })(this.target);
    }

    void initStates(V)(string name, ref V field) if (isVariable!V) {
        if (name !in this.den) {
            auto m = field.uninit();
            m.data.zero_();
            this.den[name] = UntypedVariable(m);
        }
        if (name !in this.num) {
            auto m = field.uninit();
            m.data.zero_();
            this.num[name] = UntypedVariable(m);
        }
    }

    void step(V)(string name, ref V field) if (isVariable!V) {
        import grain.chain : pow;

        auto g = field.gradVariable;
        auto d = this.den[name].to!V;
        auto n = this.num[name].to!V;
        auto nextDen= (1.0 - this.rho) * g * g + this.rho * d;
        auto diff = pow((n + this.eps) / (nextDen + this.eps), 0.5); // TODO implement sqrt
        auto nextNum = (1.0 - this.rho) * diff * diff + this.rho * n;
        this.den[name] = UntypedVariable(nextDen);
        this.num[name] = UntypedVariable(nextNum);
        transform(diff, field, -this.lr, 1.0);
    }
}

///
unittest {
    import grain.autograd;
    import numir;
    {
        auto model = MLP!(float, HostStorage)(3);
        auto optim = AdaDelta!(typeof(model))(model);
        // static assert(isOptimizer!(typeof(optim)));
        model.fc1.weight.data.zero_();
        model.fc1.weight.grad = [[0.2f, 0.0f, 0.0f], [0.0f, 0.0f, 0.0f]].variable.data;
        optim.update();
        auto w = model.fc1.weight;
        auto d = (1.0 - optim.rho) * 0.2 * 0.2;
        auto diff = cast(float) ((0.0 + optim.eps) / (d + optim.eps)) ^^ 0.5;
        auto n = (1.0 - optim.rho) * diff * diff;
        assert(approxEqual(w.sliced, [[-optim.lr * diff, -optim.lr, -optim.lr], [-optim.lr, -optim.lr, -optim.lr]].nparray));
        auto d_ = optim.den[".fc1.weight"].to!(typeof(w));
        auto n_ = optim.num[".fc1.weight"].to!(typeof(w));
        assert(approxEqual(d_.sliced, [[d, 0.0, 0.0], [0.0, 0.0, 0.0]].nparray));
        assert(approxEqual(n_.sliced[0, 0..1], [n].nparray));
    }
    version (grain_cuda) {
        auto model = MLP!(float, DeviceStorage)(3);
        auto optim = AdaDelta!(typeof(model))(model);
        optim.update();
    }
}
