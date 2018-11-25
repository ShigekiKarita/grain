/// MNIST hand-written digit recognition dataset
module grain.dataset.mnist;

import std.typecons : tuple;
import mir.ndslice;

///
enum files = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte"
    ];

///
struct Dataset {
    Slice!(float*, 3) inputs;
    Slice!(int*, 1) targets;
}

///
auto prepareDataset() {
    import std.stdio : writeln;
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

///
auto makeBatch(Dataset* d, size_t batchSize) {
    import numir.core : view;
    auto niter = d.inputs.shape[0] / batchSize; // omit last
    auto inSize = d.inputs[0].view(-1).length!0;
    return tuple!("niter", "inputs", "targets")(
        niter,
        d.inputs.view(-1, inSize)[0..$ - ($ % batchSize)].view(-1, batchSize, inSize),
        d.targets[0..$ - ($ % batchSize)].view(-1, batchSize)
        );
}
