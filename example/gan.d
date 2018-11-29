import grain;

version (grain_cuda) {
    alias S = DeviceStorage;
} else {
    alias S = HostStorage;
}

alias L = Linear!(float, S);

struct Generator {
    L fc1, fc2, fc3, fc4;
    bool isTrain = true;

    this(uint input, uint output) {
        this.fc1 = L(input, 256);
        this.fc2 = L(256, 512);
        this.fc3 = L(512, 1024);
        this.fc4 = L(1024, output);
    }

    auto opCall(Variable!(float, 2, S) zs) {
        // FIXME dropout causes fixed output in cuda
        // auto h1 = relu(dropout(this.fc1(zs), this.isTrain));
        // auto h2 = relu(dropout(this.fc2(h1), this.isTrain));
        // auto h3 = relu(dropout(this.fc3(h2), this.isTrain));
        auto h1 = relu(this.fc1(zs));
        auto h2 = relu(this.fc2(h1));
        auto h3 = relu(this.fc3(h2));
        return tanh(this.fc4(h3));
    }
}

struct Discriminator {
    L fc1, fc2, fc3, fc4;
    bool isTrain = true;

    this(uint input) {
        this.fc1 = L(input, 1024);
        this.fc2 = L(1024, 512);
        this.fc3 = L(512, 256);
        this.fc4 = L(256, 1);
    }

    auto opCall(Variable!(float, 2, S) xs) {
        // auto h1 = relu(dropout(this.fc1(xs), this.isTrain));
        // auto h2 = relu(dropout(this.fc2(h1), this.isTrain));
        // auto h3 = relu(dropout(this.fc3(h2), this.isTrain));
        auto h1 = relu(this.fc1(xs));
        auto h2 = relu(this.fc2(h1));
        auto h3 = relu(this.fc3(h2));
        return sigmoid(this.fc4(h3));
    }
}

void main() {
    import std.range : enumerate;
    import std.stdio;
    import mir.ndslice : slice;
    import numir : RNG, permutation, view, normal;
    import grain.dataset.mnist : prepareDataset, makeBatch;
    import snck : snck;
    static import grain.config;

    RNG.setSeed(0);
    grain.config.backprop = true;

    // config
    auto batchSize = 64;
    auto visibleSize = 28 * 28;
    auto latentSize = 100;
    auto lr = 1e-6;
    auto epochs = 20;

    auto datasets = prepareDataset();
    auto trainBatch = datasets.train.makeBatch(batchSize);

    // FIXME make!Adam causes liker error, why?
    auto gModel = Generator(latentSize, visibleSize);
    auto gOptim = make!SGD(gModel, lr);

    auto dModel = Discriminator(visibleSize);
    auto dOptim = make!SGD(dModel, lr);

    foreach (e; 0 .. epochs) {
        writefln!"epoch %d"(e);
        foreach (n, i; trainBatch.niter.permutation.enumerate) {
            auto x = slice(trainBatch.inputs[i] / 255f);
            auto nBatch = cast(uint) x.shape[0];

            dModel.zeroGrad();
            auto xReal = x.view(nBatch, -1).variable(true).to!S;
            // FIXME current implementation of sum is naive but thrust ver. seems to have bug?
            auto dRealLoss = sum(-1f * log(dModel(xReal)));
            dRealLoss.backward();

            auto zd = normal!float(nBatch, latentSize).slice.variable(true).to!S;
            auto dFake = gModel(zd);
            auto dFakeLoss = sum(log(dModel(dFake)));
            dFakeLoss.backward();
            dOptim.update();

            auto zg = normal!float(nBatch, latentSize).slice.variable(true).to!S;
            auto gFake = gModel(zg);
            auto gLoss = sum(log(dModel(gFake)));
            gModel.zeroGrad();
            gLoss.backward();
            gOptim.update();

            auto dLoss = dRealLoss.to!HostStorage.data[0] + dFakeLoss.to!HostStorage.data[0];
            writefln!"[%d/%d]: dloss(real) %f, dloss(fake) %f, dloss %f, gloss %f"(
                n, trainBatch.niter,
                dRealLoss.to!HostStorage.data[0],
                dFakeLoss.to!HostStorage.data[0],
                dLoss,
                gLoss.to!HostStorage.data[0]);
        }
        gModel.save("generator.h5");
        dModel.save("discriminator.h5");
    }
}
