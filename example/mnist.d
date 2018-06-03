import std.stdio;
import std.typecons : tuple;

import mir.ndslice; // : sliced, map, slice;
import numir;

import grain.autograd;
import grain.chain; // : Linear, relu;

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
            // debug print
            writeln("first image:");
            foreach (i; 0 ..  imgs.length!1) {
                import std.format;
                imgs[0, i].format!"%(%3s %)".writeln;
            }
            // normalize 0 .. 255 to 0.0 .. 1.0
            dataset.inputs = imgs.map!(i => 1.0f * i / 255).slice;
        } else { // labels
            // skip header
            decomp = decomp[8..$];
            dataset.targets = decomp.sliced.as!int.slice;
            // debug print
            writeln("first label: ", dataset.targets[0]);
        }
    }
    return tuple!("train", "test")(train, test);
}


struct MLP(T, alias Storage) {
    alias L = Linear!(T, Storage);
    L fc1, fc2, fc3;

    this(int nhidden) {
        this.fc1 = L(28*28, nhidden);
        this.fc2 = L(nhidden, nhidden);
        this.fc3 = L(nhidden, 10);
    }

    auto opCall(Variable!(T, 2, Storage) x) {
        auto h1 = relu(this.fc1(x));
        auto h2 = relu(this.fc2(h1));
        auto h3 = this.fc2(h2);
        return h3;
    }
}


void main() {
    auto datasets = prepareDataset();
    alias S = DeviceStorage;
    auto model = MLP!(float, S)(100);
    auto xs = datasets.train.inputs[0..8].view(-1, 28 * 28).variable.to!S;
    auto ts = datasets.train.targets[0..8].variable.to!S;
    xs.shape.writeln;
    auto ys = model(xs);
    ys.shape.writeln;
    auto loss = crossEntropy(ys, ts);
    loss.to!HostStorage.writeln;
    auto g = new UntypedVariable(1.0f.variable.to!S);
    loss.backward(g); // TODO test this
    ys.grad.length.writeln;
    model.fc1.bias.grad.length.writeln;
}

