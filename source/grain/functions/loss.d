/**
   A module for loss functions that always output scalar values to be minimized.
   Loss function is the end of forwardprop and also is the start point of backprop.
 */
module grain.functions.loss;

import grain.autograd;
import grain.cuda;
import grain.functions.common;
import grain.utility : toTuple, fromTuple, castArray;

struct NegativeLogLikelihood(F, I = long) {
    /++
    Compute negative log-likelihood: -logP(y=t)
    Params:
      logP: log softmax output as prediction. shape: (nBatch, nClass)
      targetId: target integer id of class. shape: (nBatch)
      +/

    mixin FunctionCommon;

    bool sizeAverage = true;
    int ignoreIndex = -100;
    // TODO: bool reduce = true;

    // cache for backward
    Variable!(I, 1, HostStorage) _htargetId;
    F _normalize;
    int _nClass;

    auto forward(Variable!(F, 2, HostStorage) logP, Variable!(I, 1, HostStorage) targetId) {
        import mir.math;
        import mir.ndslice;

        F result = 0.0;
        size_t count = 0;
        foreach (i; 0 .. targetId.sliced.length) {
            auto t = targetId.sliced[i];
            if (t != this.ignoreIndex) {
                result -= logP.sliced[i, t];
                ++count;
            }
        }
        if (this.sizeAverage && count > 0) {
            result /= count;
        }
        // TODO if train
        this._nClass = logP.shape[1];
        this._htargetId = targetId;
        this._normalize = this.sizeAverage && count > 0 ? 1.0 / count : 1.0;
        return result.variable;
    }

    auto backward(Variable!(F, 0, HostStorage) gy) {
        import std.typecons;
        import mir.math;
        import mir.ndslice;
        import numir;

        auto nBatch = this._htargetId.shape[0];
        auto glogP = zeros!F(nBatch, this._nClass);
        auto coeff = gy.data[0] * this._normalize;
        foreach (i; 0 .. nBatch) {
            auto t = this._htargetId.sliced[i];
            if (t != this.ignoreIndex) {
                glogP[i][t] = -coeff;
            }
        }
        return tuple(glogP.variable, typeof(this._htargetId)());
    }

    version (grain_cuda) {
        Variable!(I, 1, DeviceStorage) _dtargetId;
        auto forward(Variable!(F, 2, DeviceStorage) logP, Variable!(I, 1, DeviceStorage) targetId) {
            static assert(is(F == float), "only float is supported now");
            static assert(is(I == int), "only int is supported now");

            import grain.kernel : nll;

            this._nClass = logP.shape[1];
            auto dresult = CuArray!F([0]); // [result].variable.to!DeviceStorage; <- FIXME
            auto dcount = CuArray!int([0]); // [count].variable.to!DeviceStorage;

            auto batchSize = targetId.shape[0];
            Global.kernel!nll.call(dresult.ptr, dcount.ptr, logP.data.ptr,
                    targetId.data.ptr, this.ignoreIndex, batchSize, logP.strides[
                    0]).launch(batchSize);

            F result = 0.0;
            int count = 0;
            dresult.toHost(&result);
            dcount.toHost(&count);

            if (this.sizeAverage && count > 0) {
                result /= count;
            }
            // TODO if train
            this._nClass = logP.shape[1];
            this._dtargetId = targetId;
            this._normalize = this.sizeAverage && count > 0 ? 1.0 / count : 1.0;
            return result.variable.to!DeviceStorage;
        }

        auto backward(Variable!(F, 0, DeviceStorage) gy) {
            static assert(is(F == float), "only float is supported now");
            static assert(is(I == int), "only int is supported now");

            import grain.kernel;
            import std.typecons : tuple;

            auto nBatch = this._dtargetId.shape[0];
            auto glogP = CuArray!F(nBatch * this._nClass);
            glogP.zero_();
            auto coeff = gy.to!HostStorage.data[0] * this._normalize;
            Global.kernel!nllGrad.call(glogP.ptr, -coeff,
                    this._dtargetId.data.ptr,
                    this.ignoreIndex, nBatch, this._nClass).launch(nBatch);
            auto v = Variable!(F, 2, DeviceStorage)(false, [nBatch,
                    this._nClass], [this._nClass, 1], glogP);
            return tuple(v, typeof(this._dtargetId)());
        }

    }
}

/// test nll simple case, gradcheck and cpu/cuda equality
unittest {
    /++ equivalent torch v0.4 code
     >>> x = torch.FloatTensor([[0.2, 0.4, 0.4], [0.1,0.5,0.4]])
     >>> x.requires_grad = True
     >>> t = torch.LongTensor([1, 0])
     >>> l = torch.nn.functional.nll_loss(x, t)
     >>> print(l)
     tensor(-0.2500)

     >>> l.backward()
     >>> print(x.grad)
     tensor([[0.0, -0.5, 0.0], [-0.5, 0.0, 0.0]])
     +/
    import std.typecons;
    import grain.testing;

    NegativeLogLikelihood!(float, int) func;
    auto hx = [[0.2f, 0.4f, 0.4f], [0.1f, 0.5f, 0.4f], [0.1f, 0.5f, 0.4f]]
        .variable;
    auto ht = [1, 0, func.ignoreIndex].variable;
    auto hl = func.forward(hx, ht);
    assert(func._normalize == 0.5);
    assert(hl.sliced == [-(0.4f + 0.1f + 0.0f) / 2]);
    auto hgx = func.backward(1.0f.variable);
    assert(hgx[0].sliced == [[0.0, -0.5, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    assert(!hgx[1].defined);
    gradCheck(func, tuple(hx, ht), 1.0f.variable);

    version (grain_cuda) {
        auto dx = hx.to!DeviceStorage;
        auto dt = ht.to!DeviceStorage;
        auto dl = func.forward(dx, dt);
        assert(func._normalize == 0.5);
        assert(dl.to!HostStorage.sliced == [-(0.4f + 0.1f + 0.0f) / 2]);
        auto dgx = func.backward(1.0f.variable.to!DeviceStorage);
        assert(dgx[0].to!HostStorage.sliced ==
               [[0.0, -0.5, 0.0],
                [-0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0]]);
        assert(!dgx[1].defined);
    }
}

/// test variable.backward
unittest {
    import std.typecons;
    import grain.testing;
    import mir.ndslice;
    static import grain.config;

    grain.config.backprop = true;

    NegativeLogLikelihood!(float, int) func;
    auto hx = [[0.2f, 0.4f, 0.4f], [0.1f, 0.5f, 0.4f], [0.1f, 0.5f, 0.4f]]
        .variable;
    hx.requiresGrad = true;
    auto ht = [1, 0, func.ignoreIndex].variable;
    auto hl = func.applyForward(hx, ht);

    assert(func._normalize == 0.5);
    assert(hl.sliced == [-(0.4f + 0.1f + 0.0f) / 2]);
    auto u = UntypedVariable(1.0f.variable);
    hl.backward(&u);

    assert(hx.grad[].sliced(3, 3) == [[0.0, -0.5, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.0,
            0.0]]);
    // TODO assert(!ht.grad.defined);
}

struct HuberLoss(T) {
    auto forward() {

    }
}

/+
/**
   PyTorch equality check
 */
unittest {
    import std.typecons;
    import grain.testing;
    import mir.ndslice;
    static import grain.config;

    grain.config.backprop = true;

    HuberLoss!float func;
    auto hx = [[0.2f, 0.4f, 0.4f], [0.1f, 0.5f, 0.4f], [0.1f, 0.5f, 0.4f]]
        .variable;
    hx.requiresGrad = true;
    auto ht = [1, 0, func.ignoreIndex].variable;
    auto hl = func.applyForward(hx, ht);

    assert(func._normalize == 0.5);
    assert(hl.sliced == [-(0.4f + 0.1f + 0.0f) / 2]);
    auto u = UntypedVariable(1.0f.variable);
    hl.backward(&u);

    assert(hx.grad[].sliced(3, 3) == [[0.0, -0.5, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.0,
            0.0]]);
}
+/
