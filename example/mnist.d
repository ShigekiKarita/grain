import std.stdio;
import std.typecons : tuple;

import mir.ndslice; // : sliced, map, slice;
import numir;

import grain.autograd;

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

    Dataset train, test;
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

import numir;
struct Linear(T, alias Storage) {
    import std.traits : isFloatingPoint;
    import grain.functions : MatMul;
    static assert(isFloatingPoint!T);
    Variable!(T, 2, Storage) weight;
    Variable!(T, 1, Storage) bias;

    this(int ninput, int noutput) {
        this.weight = normal!T(ninput, noutput).slice.variable.to!Storage;
        this.bias = normal!T(noutput).slice.variable.to!Storage;
    }

    auto opCall(Variable!(T, 2, Storage) x) {
        auto matmul = new MatMul!T;
        return matmul.applyForward(x, this.weight);
    }
}

struct MLP(T, alias Storage) {
    // import grain.chain : Linear;
    alias L = Linear!(T, Storage);
    L fc1, fc2, fc3;

    this(int nhidden = 1000) {
        this.fc1 = L(28*28, nhidden);
        this.fc2 = L(nhidden, nhidden);
        this.fc3 = L(nhidden, 10);
    }

    auto opCall(Variable!(T, 2, Storage) x) {
        auto h1 = this.fc1(x);
        auto h2 = this.fc2(h1);
        auto h3 = this.fc2(h2);
        return h3;
    }
}


void main() {
    auto datasets = prepareDataset();
}

