module example.cifar;

/**
   Image recognition example on CIFAR10/100

   TODO: implement maxpool, dropout and batch normalization for VGG
*/

import std.stdio : writeln, writefln;
import std.exception : enforce;

struct Dataset {
    import mir.ndslice : Slice, Contiguous;

    /// shape is [data, rgb, width, height]
    Slice!(float*, 4) inputs;
    Slice!(int*, 1) targets;
    Slice!(int*, 1) coarses;

    this(string[] paths) {
        import std.algorithm : sum, map, canFind;
        import mir.ndslice :  sliced, flattened, universal;
        import numir : empty;
        import std.stdio : File;

        immutable isCIFAR100 = paths[0].canFind("cifar-100");
        immutable dataSize = isCIFAR100 ? 3074 : 3073;
        auto fileSize = paths.map!(p => File(p, "rb").size).sum;
        enforce(fileSize % dataSize == 0);
        auto numData = fileSize / dataSize;
        this.inputs = empty!float(numData, 3, 32, 32);
        this.targets = empty!int(numData);
        if (isCIFAR100) {
            this.coarses = empty!int(numData);
        }

        size_t i = 0;
        immutable imageOffset = isCIFAR100 ? 2 : 1;
        foreach (p; paths) {
            foreach (chunk; File(p).byChunk(dataSize)) {
                if (isCIFAR100) {
                    this.coarses[i] = cast(int) chunk[0];
                    this.targets[i] = cast(int) chunk[1];
                } else {
                    this.targets[i] = cast(int) chunk[0];
                }

                size_t n = 0;
                foreach (ref inp; this.inputs[i].flattened) {
                    inp = cast(float) chunk[imageOffset + n];
                    ++n;
                }
                ++i;
            }
        }
    }

    auto makeBatch(size_t batchSize) {
        import numir : view;
        import std.typecons : tuple;
        auto niter = this.inputs.length!0 / batchSize; // omit last
        auto ndata = batchSize * niter;
        return tuple!("niter", "inputs", "targets")(
            niter,
            this.inputs[0..ndata].view(niter, batchSize, 3, 32, 32),
            this.targets[0..ndata].view(niter, batchSize)
            );
    }
}

auto prepareDataset(string dataset, string dir = "data") {
    import std.typecons : tuple;
    import std.array : array;
    import std.format : format;
    import std.algorithm : filter, canFind, map;
    import std.string : replace, split;
    import std.path : baseName, extension, dirName;
    import std.file : exists, read, mkdirRecurse, dirEntries, SpanMode, readText;
    import std.net.curl : download, HTTP;
    import std.range : chunks;
    import std.process : executeShell;

    immutable url = "https://www.cs.toronto.edu/~kriz/%s-binary.tar.gz".format(dataset);
    immutable root = dir ~ "/" ~ url.baseName.replace(".tar.gz", "");
    if (!root.exists) {
        immutable gz = dir ~ "/" ~ url.baseName;
        if (!gz.exists) {
            writeln("downloading ", url);
            auto conn = HTTP(url);
            download(url, gz, conn);
            auto code = conn.statusLine().code;
            // FIXME: does not work?
            enforce(code == 200, "status code: %s".format(code));
        }
        mkdirRecurse(root);
        auto cmd = "tar -xvf " ~ gz ~ " -C " ~ root;
        writeln("uncompressing ", cmd);
        auto ret = executeShell(cmd);
        enforce(ret.status == 0, ret.output);
    }

    auto bins = root
        .dirEntries("*", SpanMode.depth)
        .filter!(a => a.name.extension == ".bin")
        .map!"a.name".array;

    auto train = Dataset(bins.filter!(a => !a.canFind("test")).array);
    auto test = Dataset(bins.filter!(a => a.canFind("test")).array);
    auto meta = dataset == "cifar-10" ? "batches.meta.txt" : "fine_label_names.txt";
    auto labels = readText(bins[0].dirName ~ "/" ~ meta).split;

    string[] coarseLabels;
    if (dataset == "cifar-100") {
        coarseLabels = readText(bins[0].dirName ~ "/" ~ "coarse_label_names.txt").split;
    }
    return tuple!("train", "test", "labels", "coarses")(train, test, labels, coarseLabels);
}


void dumpPNG(Dataset dataset, string[] labels, string[] coarses, string root) {
    import std.path : dirName;
    import std.file : mkdirRecurse;
    import mir.ndslice : flattened, swapped, map, iota;
    import std.array : array, empty;
    import std.format : format;
    import imaged : Img, Px;
    import snck : snck;

    foreach (i; dataset.targets.length.iota.snck) {
        auto fs = dataset.inputs[i];
        auto label = labels[dataset.targets[i]];
        auto bs = fs.swapped!(1, 2).swapped!(0, 2).flattened.map!(a => cast(ubyte) a).array;
        auto img = new Img!(Px.R8G8B8)(32, 32, bs);
        string path;
        if (coarses.empty) {
            path = root ~ "/cifar10/%s/%s.png".format(label, i);
        } else {
            auto coarse = coarses[dataset.coarses[i]];
            path = root ~ "/cifar100/%s/%s/%s.png".format(coarse, label, i);
        }
        mkdirRecurse(path.dirName);
        img.write(path);
    }
}

import grain;

struct Model(T, alias Storage) {
    alias L = Linear!(T, Storage);
    Convolution!(float, 2, Storage) conv;
    L fc1, fc2, fc3;

    this(int nhidden, int nout) {
        import std.algorithm : reduce;
        this.conv = Convolution!(float, 2, Storage)(3, 32, // in-out channels
                                                    [3, 3], // kernel size
                                                    [3, 3], // stride size
                                                    [0, 0], // pad size
                                                    [1, 1], // dilation size
                                                    true); // use bias
        auto hshape = this.conv.outShape([1, 3, 32, 32]);
        auto n1 = reduce!"a * b"(1, hshape[]);
        this.fc1 = L(n1, nhidden);
        this.fc2 = L(nhidden, nhidden);
        this.fc3 = L(nhidden, nout);
    }

    auto opCall(Variable!(T, 4, Storage) xs) {
        auto h0 = relu(this.conv(xs)).view(xs.shape[0], -1);
        auto h1 = relu(this.fc1(h0));
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


void main(string[] args) {
    import std.getopt : getopt, defaultGetoptPrinter;
    import std.string : empty;
    import std.file : exists;
    import numir;
    import snck : snck;
    import grain.autograd;
    static import grain.config;

    auto datasetName = "cifar-10";
    auto dumpDir = "";
    auto opt = getopt(
        args,
        "dataset", "dataset name (only cifar-10 and cifar-100 are allowed)", &datasetName,
        "dumpPNG", "dump .png images from .bin files into specified dir", &dumpDir
        );

    if (opt.helpWanted) {
        defaultGetoptPrinter("Image recognition example on CIFAR10/100",
                             opt.options);
        return;
    }

    auto datasets = prepareDataset(datasetName);
    if (!dumpDir.empty) {
        dumpPNG(datasets.train, datasets.labels, datasets.coarses, dumpDir);
        dumpPNG(datasets.test, datasets.labels, datasets.coarses, dumpDir);
    }

    // TODO add training and evaluation code here
    RNG.setSeed(0);
    grain.config.backprop = true;
    auto batchSize = 64;
    auto trainBatch = datasets.train.makeBatch(batchSize);
    auto testBatch = datasets.test.makeBatch(batchSize);
    auto model = Model!(float, S)(256, cast(int) datasets.labels.length);
    auto optimizer = SGD!(typeof(model))(model, 0.001);
    // if ("cifar_cnn.h5".exists) {
    //     model.load("cifar_cnn.h5");
    // }

    foreach (epoch; 0 .. 300) {
        // TODO implement model.train();
        writeln("epoch: ", epoch);
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
        model.save("cifar_cnn.h5");
    }
}
