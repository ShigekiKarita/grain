import std.stdio : writeln, writefln;
import std.typecons : tuple;
import std.exception : enforce;

struct Dataset {
    import mir.ndslice : Slice, Contiguous;

    /// shape is [data, rgb, width, height]
    Slice!(Contiguous, [4], float*) inputs;
    Slice!(Contiguous, [1], int*) targets;
    Slice!(Contiguous, [1], int*) coarses;

    this(string[] paths) {
        import std.range;
        import std.algorithm : sum, map, canFind;
        import mir.ndslice : each, zip, sliced, flattened, universal;
        import numir : empty;
        import std.stdio : File;
        import std.conv : to;

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
}

auto prepareDataset(string dataset, string dir = "data") {
    import std.array : array;
    import std.format : format;
    import std.algorithm : filter, canFind, map;
    import std.string : replace, split;
    import std.path : baseName, extension, dirName;
    import std.file : exists, read, mkdir, dirEntries, SpanMode, readText;
    import std.algorithm : canFind;
    import std.net.curl : download, HTTP;
    import std.range : chunks;
    import std.process : executeShell;

    if (!dir.exists) {
        mkdir(dir);
    }

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
        mkdir(root);
        auto cmd = "tar -xvf " ~ gz ~ " -C " ~ root;
        writeln("uncompressing ", cmd);
        auto ret = executeShell(cmd);
        enforce(ret.status == 0, ret.output);
        enforce(root.exists, root ~ " does not exist");
    }

    auto bins = root
        .dirEntries("*", SpanMode.depth)
        .filter!(a => a.name.extension == ".bin")
        .map!"a.name".array;

    auto train = Dataset(bins.filter!(a => !a.canFind("test")).array);
    auto test = Dataset(bins.filter!(a => a.canFind("test")).array);
    auto meta = dataset == "cifar-10" ? "batches.meta.txt" : "fine_label_names.txt";
    auto labels = readText(bins[0].dirName ~ "/" ~ meta).split;
    writeln(labels);
    string[] coarseLabels;
    if (dataset == "cifar-100") {
        coarseLabels = readText(bins[0].dirName ~ "/" ~ "coarse_label_names.txt").split;
    }
    return tuple!("train", "test", "labels", "coarses")(train, test, labels, coarseLabels);
}


void dumpPNG(Dataset dataset, string[] labels, string[] coarses, string root) {
    import std.path;
    import std.file;
    import mir.ndslice;
    import std.array;
    import std.format;
    import imaged;
    import snck;

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

void main(string[] args) {
    import std.getopt;
    import std.string;

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

    auto dataset = prepareDataset(datasetName);
    if (!dumpDir.empty) {
        dumpPNG(dataset.train, dataset.labels, dataset.coarses, dumpDir);
        dumpPNG(dataset.test, dataset.labels, dataset.coarses, dumpDir);
    }

    // TODO add training and evaluation code here
}
