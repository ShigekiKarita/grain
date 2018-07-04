import std.stdio;
import std.typecons : tuple;

import mir.ndslice; // : sliced, map, slice;
import numir;
import snck : snck;

import grain.autograd;
import grain.chain; // : Linear, relu;
import grain.optim; //  : zeroGrad;
import grain.serializer : save, load;

enum files = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte"
    ];

struct Dataset {
    Slice!(Contiguous, [3], float*) inputs;
    Slice!(Contiguous, [1], int*) targets;
}

auto prepareDataset() {
    import std.file : exists, read, mkdir;
    import std.algorithm : canFind;
    import std.zlib : UnCompress;
    import std.net.curl : download;

    // Dataset train, test;
    auto train = new Dataset;
    auto test = new Dataset;
    if (!exists("data")) {
        mkdir("data");
    }
    foreach (f; files) {
        auto gz = "data/" ~ f ~ ".gz";
        writeln("loading " ~ gz);
        if (!exists(gz)) {
            auto url = "http://yann.lecun.com/exdb/mnist/" ~ f ~ ".gz";
            download(url, gz);
        }
        auto unc = new UnCompress;
        auto decomp = cast(ubyte[]) unc.uncompress(gz.read);
        auto dataset = f.canFind("train") ? train : test;
        if (f.canFind("images")) {
            // skip header
            decomp = decomp[16..$];
            auto ndata = decomp.length / (28 * 28);
            auto imgs = decomp.sliced(ndata, 28, 28);
            // normalize 0 .. 255 to 0.0 .. 1.0
            dataset.inputs = imgs.map!(i => 1.0f * i / 255).slice;
        } else { // labels
            decomp = decomp[8..$];
            dataset.targets = decomp.sliced.as!int.slice;
        }
    }
    return tuple!("train", "test")(train, test);
}

auto makeBatch(Dataset* d, size_t batchSize) {
    auto niter = d.inputs.shape[0] / batchSize; // omit last
    auto inSize = d.inputs[0].view(-1).length!0;
    return tuple!("niter", "inputs", "targets")(
        niter,
        d.inputs.view(-1, inSize)[0..$ - ($ % batchSize)].view(-1, batchSize, inSize),
        d.targets[0..$ - ($ % batchSize)].view(-1, batchSize)
        );
}


struct Model(T, alias Storage) {
    alias L = Linear!(T, Storage);
    Convolution!(float, 2, Storage) conv;
    L fc1, fc2, fc3;

    this(int nin, int nhidden, int nout) {
        this.conv = Convolution!(float, 2, Storage)(1, 32, // in-out channels
                                                    [3, 3], // kernel size
                                                    [3, 3], // stride size
                                                    [0, 0], // pad size
                                                    [1, 1], // dilation size
                                                    true); // use bias
        auto x = numir.empty!float(1, 1, nin, nin).variable.to!Storage;
        auto hshape = this.conv.outShape([1, 1, nin, nin]);
        import std.algorithm;
        auto n1 = std.algorithm.reduce!"a * b"(1, hshape[]);
        this.fc1 = L(n1, nhidden);
        this.fc2 = L(nhidden, nhidden);
        this.fc3 = L(nhidden, nout);
    }

    auto opCall(Variable!(T, 2, Storage) x) {
        import grain.chain : view;
        auto xs = x.view(x.shape[0], 1, 28, 28);
        auto h0 = relu(this.conv(xs)).view(x.shape[0], -1);
        auto h1 = relu(this.fc1(h0));
        auto h2 = relu(this.fc2(h1));
        auto h3 = this.fc3(h2);
        return h3;
    }
}

auto accuracy(Vy, Vt)(Vy y, Vt t) { // if (isVariable!Vy && isVariable!Vt) {
    auto nbatch = t.shape[0];
    auto hy = y.to!HostStorage.sliced;
    auto ht = t.to!HostStorage.sliced;
    double acc = 0.0;
    foreach (i; 0 .. nbatch) {
        auto maxid = hy[i].maxIndex[0];
        if (maxid == ht[i]) {
            ++acc;
        }
    }
    return acc / nbatch;
}

// FIXME build with "optimize" option and DeviceStorage backend causes loss=nan (non-release or CPU is OK)
version (grain_cuda) {
    alias S = DeviceStorage;
} else {
    alias S = HostStorage;
}

void main() {
    import std.file : exists;
    RNG.setSeed(0);
    grain.autograd.backprop = true;
    auto datasets = prepareDataset();
    auto batchSize = 64;
    auto inSize = 28;
    auto trainBatch = datasets.train.makeBatch(batchSize);
    auto testBatch = datasets.test.makeBatch(batchSize);
    auto model = Model!(float, S)(inSize, 256, 10);
    auto optimizer = SGD!(typeof(model))(model, 1e-2);
    if ("mnist_cnn.h5".exists) {
        model.load("mnist_cnn.h5");
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
                // writefln!"train loss: %f, acc: %f"(lossSum / i, accSum / i);
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
        model.save("mnist_cnn.h5");
    }
}
