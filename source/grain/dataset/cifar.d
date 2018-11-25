/// CIFAR10/100 image recognition datasets
module grain.dataset.cifar;

import std.exception : enforce;

///
struct Dataset {
    import mir.ndslice : Slice, Contiguous;

    /// shape is [data, rgb, width, height]
    Slice!(float*, 4) inputs;
    Slice!(int*, 1) targets;
    Slice!(int*, 1) coarses;

    ///
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

    ///
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

///
auto prepareDataset(string dataset, string dir = "data") {
    import std.stdio : writeln;
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
