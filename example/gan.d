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
        auto h1 = relu(dropout(this.fc1(zs), this.isTrain));
        auto h2 = relu(dropout(this.fc2(h1), this.isTrain));
        auto h3 = relu(dropout(this.fc3(h2), this.isTrain));
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
        auto h1 = relu(dropout(this.fc1(xs), this.isTrain));
        auto h2 = relu(dropout(this.fc2(h1), this.isTrain));
        auto h3 = relu(dropout(this.fc3(h2), this.isTrain));
        return sigmoid(this.fc4(h3));
    }
}

void main() {
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
    auto lr = 1e-4;
    auto epochs = 20;

    auto datasets = prepareDataset();
    auto trainBatch = datasets.train.makeBatch(batchSize);

    auto gModel = Generator(latentSize, visibleSize);
    auto gOptim = make!Adam(gModel, lr);

    auto dModel = Discriminator(visibleSize);
    auto dOptim = make!Adam(dModel, lr);

    foreach (e; 0 .. epochs) {
        writefln!"epoch %d"(e);
        foreach (i; trainBatch.niter.permutation) {
            auto x = slice(trainBatch.inputs[i] / 255f);
            auto nBatch = cast(uint) x.shape[0];

            dModel.zeroGrad();
            auto xReal = x.view(nBatch, -1).variable(true).to!S;
            auto dRealLoss = sum(0f - log(dModel(xReal)));
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
            writefln!"dloss(real) %f, dloss(fake) %f, dloss %f, gloss %f"(
                dRealLoss.to!HostStorage.data[0],
                dFakeLoss.to!HostStorage.data[0],
                dLoss,
                gLoss.to!HostStorage.data[0]);
        }
        gModel.save("generator.h5");
        dModel.save("discriminator.h5");
    }
}
