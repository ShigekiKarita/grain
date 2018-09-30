module ptb;

import std.array;
import std.algorithm : splitter;
import std.stdio : File;
import mir.ndslice;

import numir;
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
    enum gates = ["a", "i", "f", "o"];
    static foreach (g; gates) {
        static foreach (i; ["x", "h"]) {
            mixin("Linear!(T, Storage) " ~ i ~ g ~";");
        }
    }

    this(int inSize, int outSize) {
        static foreach (g; gates) {
            static foreach (i, isize; ["x": "inSize", "h": "outSize"]) {
                mixin(i ~ g ~" = Linear!(T, Storage)(" ~ isize ~ ", outSize);");
            }
        }
    }

    auto opCall(Variable!(T, 2, Storage) x, Variable!(T, 2, Storage) h, Variable!(T, 2, Storage) c) {
        import std.typecons : tuple;
        if (!h.defined || !c.define) {
            return this.opCall(x);
        }

        static foreach (g; gates) {
            mixin("auto " ~ g ~ "_ = x" ~ g  ~ "(x) + h" ~ g ~ "(h);");
        }
        auto cNext = sigmoid(f_) * c + sigmoid(i_) * tanh(a_);
        auto hNext = sigmoid(o_) * tanh(cNext);
        return tuple!("h", "c")(hNext, cNext);
    }

    auto opCall(Variable!(T, 2, Storage) x) {
        import std.typecons : tuple;
        static foreach (g; gates) {
            mixin("auto " ~ g ~ "_ = x" ~ g  ~ "(x);");
        }
        auto cNext = sigmoid(i_) * tanh(a_);
        auto hNext = sigmoid(o_) * tanh(cNext);
        return tuple!("h", "c")(hNext, cNext);
    }
}

void test() {
    auto lstm = LSTM!(float, HostStorage)(2, 3);
    auto x = zeros!float(5, 2).variable;

    auto state1 = lstm(x);
    assert(state1.h.shape == [5, 3]);
    assert(state1.c.shape == [5, 3]);

    auto state2 = lstm(x, state1.h, state1.c);
    assert(state2.h.shape == [5, 3]);
    assert(state2.c.shape == [5, 3]);
}

struct RNNLM(alias Storage) {
    Embedding!(float, Storage) embed;
    LSTM!(float, Storage) lstm1, lstm2;
    Linear!(float, Storage) linear;

    static void init(C, Args ...)(ref C chain, Args args) {
        chain = C(args);
    }

    this(int vocabSize, int embedSize, int hiddenSize) {
        init(embed, vocabSize, embedSize);
    }

    auto opCall(Variable!(int, 1, Storage)[] xs) {
        // TODO implement indexing
        double ret = 0;
        Variable!(float, 2, Storage) h1, h2, c1, c2;
        foreach (i, x; xs[0 .. $-1]) {
            auto h0 = this.embed(x);
            auto s1 = this.lstm1(h0, h1, c1);
            h1 = s1.h;
            c1 = s1.h;
            auto s2 = this.lstm2(h1, h2, c2);
            h2 = s2.h;
            c2 = s2.c;
            auto y = this.fc(h2);
            auto loss = crossEntropy(y, xs[i + 1]);
            loss.backward();
            ret += loss.to!HostStorage.data[0];
        }
        return ret;
    }
}

void main(string[] args) {
    debug test();

    import std.stdio;
    import std.getopt;
    import std.format;

    version (grain_cuda) alias Storage = DeviceStorage;
    else alias Storage = HostStorage;

    auto dataset = "data";
    auto embed = 200;
    auto hidden = 200;

    auto opt = getopt(
        args,
        "dataset", format!"dataset path (default %s)"(dataset), &dataset,
        "embed", format!"embed size (default %s)"(embed), &embed,
        "hidden", format!"hidden size (default %s)"(hidden), &hidden
        );

    if (opt.helpWanted) {
        defaultGetoptPrinter("Image recognition example on CIFAR10/100",
                             opt.options);
        return;
    }

    Corpus corpus;
    foreach (name; ["train", "valid", "test"]) {
        corpus.register("data/ptb." ~ name ~ ".txt", name);
    }
    immutable vocabSize = cast(int) corpus.dict.idx2word.length;
    writeln("vocab:", vocabSize);
    corpus.batchfy("train")[0 .. 10, 0].map!(i => corpus.dict.idx2word[i]).writeln;
    auto rnnlm = RNNLM!Storage(vocabSize, embed, hidden);
}
