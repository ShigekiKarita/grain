module ptb;

import std.array;
import std.algorithm : splitter;
import std.stdio : File;
import mir.ndslice;
import grain;

struct Dictionary {
    enum eos = "<eos>";
    enum eosId = 0;
    string[] idx2word;
    int[string] word2idx;

    auto register(string word) {
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

    auto batchfy(string name) {
        import numir;
        const data = this.dataset[name];
        const len = data.length / this.batchSize;
        return data[0 .. len * this.batchSize].sliced.view(this.batchSize, len).transposed.slice;
    }
}

struct LSTM(T, alias Storage) {
    int inSize, outSize;
    enum gates = ["c", "i", "f", "o"];
    static foreach (i; ["x", "h"]) {
        static foreach (g; gates) {
            mixin("Linear!(T, Storage) " ~ i ~ g ~";");
        }
    }

    this(int inSize, int outSize) {
        static foreach (g; gates) {
            mixin("x" ~ g ~" = Linear!(T, Storage)(inSize, outSize);");
            mixin("h" ~ g ~" = Linear!(T, Storage)(outSize, outSize);");
        }
    }

    auto opCall(Variable!(T, 2, Storage) x, Variable!(T, 2, Storage) h, Variable!(T, 2, Storage) c) {
        import std.typecons;
        static foreach (g; gates) {
            mixin("auto " ~ g ~ "_ = x" ~ g  ~ "(x) + h" ~ g ~ "(h);");
        }
        auto cNext = sigmoid(f_) * c + sigmoid(i_) * tanh(c_);
        auto hNext = sigmoid(o_) * tanh(cNext);
        return tuple!("h", "c")(cNext, hNext);
    }
}

struct RNNLM(alias Storage) {
    size_t vocabSize, embedSize, hiddenSize;
    Embedding!(int, Storage) embed;
    Linear!(float, Storage) linear;
}

void main() {
    import std.stdio;
    Corpus corpus;
    foreach (name; ["train", "valid", "test"]) {
        corpus.register("data/ptb." ~ name ~ ".txt", name);
    }
    writeln("vocab:", corpus.dict.idx2word.length);
    // auto trainBatchList =
    // writeln(corpus.dataset["train"][0 .. 10].map!(i => corpus.dict.idx2word[i]));
    corpus.batchfy("train")[0 .. 10, 0].map!(i => corpus.dict.idx2word[i]).writeln;
    auto lstm = LSTM!(float, HostStorage)(2, 3);
}
