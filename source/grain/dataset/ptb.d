// PTB language modeling dataset
module grain.dataset.ptb;

import std.array;
import std.algorithm : splitter;
import std.stdio : File;
import mir.ndslice;
import mir.random.variable : UniformVariable;

import numir;

auto prepareDataset() {
    import std.format;
    import std.file : exists, read, mkdir;
    import std.net.curl : download;

    if (!exists("data")) {
        mkdir("data");
    }
    enum root = "https://github.com/tomsercu/lstm/raw/master/";
    foreach (f; ["train", "valid", "test"]) {
        auto dst = format!"data/ptb.%s.txt"(f);
        if (!exists(dst))download(root ~ dst, dst);
    }

    Corpus corpus;
    foreach (name; ["train", "valid", "test"]) {
        corpus.register("data/ptb." ~ name ~ ".txt", name);
    }
    return corpus;
}

struct Dictionary {
    enum eos = "<eos>";
    enum eosId = 0;
    string[] idx2word;
    int[string] word2idx;

    void register(string word) {
        assert(int.max > this.idx2word.length);
        if (this.idx2word.empty) { // init
            this.idx2word = [eos];
            this.word2idx[eos] = 0;
        }
        if (word !in this.word2idx) {
            this.word2idx[word] = cast(int) this.idx2word.length;
            this.idx2word ~= word;
        }
    }
}

struct Corpus {
    Dictionary dict;
    int[][string] dataset;
    size_t batchSize = 20;

    void register(string path, string name) {
        import std.string : strip;
        int[] data;
        foreach (line; File(path).byLine) {
            foreach (word; line.strip.splitter(' ')) {
                this.dict.register(word.idup);
                data ~= this.dict.word2idx[word];
            }
            data ~= Dictionary.eosId;
        }
        dataset[name] = data;
    }

    /// returns word-id 2d slice shaped (seqlen, batchsize)
    auto batchfy(string name) {
        import numir;
        auto data = this.dataset[name];
        const len = data.length / this.batchSize;
        return data[0 .. len * this.batchSize].sliced.view(this.batchSize, len).transposed.slice;
    }

    auto vocabSize() {
        return cast(int) this.dict.idx2word.length;
    }
}
