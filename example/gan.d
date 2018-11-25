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
    import numir : RNG, permutation, view;
    import grain.dataset.mnist : prepareDataset, makeBatch;
    import snck : snck;
    static import grain.config;

    RNG.setSeed(0);
    grain.config.backprop = true;

    // config
    auto batchSize = 64;
    auto visibleSize = 28 * 28;
    auto latentSize = 100;
    auto lr = 5e-4;
    auto epochs = 1;

    auto datasets = prepareDataset();
    auto trainBatch = datasets.train.makeBatch(batchSize);

    auto modelG = Generator(latentSize, visibleSize);
    auto modelD = Discriminator(visibleSize);
    auto optimG = Adam!Generator(modelG, lr);
    auto optimD = Adam!Discriminator(modelD, lr);

    foreach (e; 0 .. epochs) {
        writefln!"epoch %d"(e);
        foreach (i; trainBatch.niter.permutation.snck) {
            // writefln!"iter %d / %d"(i, trainBatch.niter);
            auto x = trainBatch.inputs[i];
            auto nBatch = cast(uint) x.shape[0];
            auto xReal = x.view(nBatch, -1).variable(true).to!S;
            auto yReal = uninitVariable!(float, S)([nBatch], true);
            yReal.data.fill_(1);
            auto pReal = modelD(xReal);
            auto dRealLoss = pReal.log.sum;
        }
    }
}
