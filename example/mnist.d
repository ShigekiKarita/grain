import std.stdio;
import std.typecons : tuple;

import mir.ndslice; // : sliced, map, slice;
import numir;
import snck : snck;

import grain.autograd;
import grain.chain; // : Linear, relu;
import grain.optim; //  : zeroGrad;
import grain.serializer : save, load;
import grain.dataset.mnist : prepareDataset, makeBatch;


struct Model(T, alias Storage) {
    alias L = Linear!(T, Storage);
    L fc1, fc2, fc3;

    this(int nin, int nhidden, int nout) {
        this.fc1 = L(nin * nin, nhidden);
        this.fc2 = L(nhidden, nhidden);
        this.fc3 = L(nhidden, nout);
    }

    auto opCall(Variable!(T, 2, Storage) x) {
        auto h1 = relu(this.fc1(x));
        auto h2 = relu(this.fc2(h1));
        auto h3 = this.fc3(h2);
        return h3;
    }
}

// FIXME build with "optimize" option and DeviceStorage backend causes loss=nan (non-release or CPU is OK)
version (grain_cuda) {
    alias S = DeviceStorage;
} else {
    alias S = HostStorage;
}

void main() {
    import std.file : exists;
    import grain.metric : accuracy;
    static import grain.config;

    RNG.setSeed(0);
    grain.config.backprop = true;
    auto datasets = prepareDataset();
    auto batchSize = 64;
    auto inSize = 28;
    auto trainBatch = datasets.train.makeBatch(batchSize);
    auto testBatch = datasets.test.makeBatch(batchSize);
    auto model = Model!(float, S)(inSize, 512, 10);
    auto optimizer = SGD!(typeof(model))(model, 1e-2);
    if ("mnist.h5".exists) {
        model.load("mnist.h5");
    }

    foreach (epoch; 0 .. 10) {
        // TODO implement model.train();
        with (trainBatch) {
            double lossSum = 0;
            double accSum = 0;
            foreach (i; niter.permutation.snck) {
                auto xs = inputs[i].variable(true).to!S;
                auto ts = targets[i].variable.to!S;
                auto ys = model(xs);
                auto loss = crossEntropy(ys, ts);
                auto acc = accuracy(ys, ts);
                lossSum += loss.to!HostStorage.data[0];
                accSum += acc;
                model.zeroGrad();
                loss.backward();
                optimizer.update();
            }
            writefln!"train loss: %f, acc: %f"(lossSum / niter, accSum / niter);
        }
        // TODO implement model.eval(); and grain.autograd.noBackprop
        with (testBatch) {
            double lossSum = 0;
            double accSum = 0;
            foreach (i; 0 .. testBatch.niter) {
                auto xs = inputs[i].variable.to!S;
                auto ts = targets[i].variable.to!S;
                auto ys = model(xs);
                auto loss = crossEntropy(ys, ts);
                auto acc = accuracy(ys, ts);
                lossSum += loss.to!HostStorage.data[0];
                accSum += acc;
            }
            writefln!"test loss: %f, acc: %f"(lossSum / niter, accSum / niter);
        }
        model.save("mnist.h5");
    }
}
