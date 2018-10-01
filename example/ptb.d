module ptb;

import std.array;
import std.algorithm : splitter;
import std.stdio : File;
import mir.ndslice;
import mir.random.variable : UniformVariable;

import numir;
import grain;

auto prepareDataset() {
    import std.format;
    import std.file : exists, read, mkdir;
    import std.net.curl : download;

    if (!exists("data")) {
        mkdir("data");
        enum root = "https://github.com/tomsercu/lstm/raw/master/";
        enum dst = "data/ptb.%s.txt";
        foreach (f; ["train", "valid", "test"]) {
            download(format!(root ~ dst)(f), format!dst(f));
        }
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
        if (!h.defined || !c.defined) {
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

void uniform_(C)(ref C chain) {
    import std.traits : hasMember;
    static if (!hasMember!(C, "tupleof")) return;
    else {
        foreach (c; chain.tupleof) {
            static if (is(typeof(C) == Linear!(T, Storage))) {
                c.weight.sliced[] = UniformVariable!T(-0.1, 0.1).generate(c.nInput, c.nOutput);
                c.bias.sliced[] = 0;
            } else {
                uniform_(c);
            }
        }
    }
}

struct RNNLM(alias Storage) {
    Embedding!(float, Storage) embed;
    LSTM!(float, Storage) lstm1, lstm2;
    Linear!(float, Storage) linear;

    static ref init(C, Args ...)(ref C chain, Args args) {
        chain = C(args);
        return chain;
    }

    this(int vocabSize, int embedSize, int hiddenSize) {
        init(embed, vocabSize, embedSize);
        init(lstm1, embedSize, hiddenSize).uniform_;
        init(lstm2, hiddenSize, hiddenSize).uniform_;
        init(linear, hiddenSize, vocabSize).uniform_;
    }

    auto opCall(O)(Slice!(int*, 2) xslice, ref O optimizer) {
        // TODO implement indexing
        auto xs = new Variable!(int, 1, Storage)[xslice.length!0];
        foreach (i; 0 .. xslice.length!0) {
            xs[i] = xslice[i].variable.to!Storage;
        }
        double ret = 0;
        alias VF2 = Variable!(float, 2, Storage);
        auto h0 = new VF2[xs.length];
        auto h1 = new VF2[xs.length+1];
        auto h2 = new VF2[xs.length+1];
        auto c1 = new VF2[xs.length+1];
        auto c2 = new VF2[xs.length+1];
        foreach (i, x; xs[0 .. $-1]) {
            h0[i] = this.embed(x);
            auto s1 = this.lstm1(h0[i], h1[i], c1[i]);
            h1[i+1] = s1.h;
            c1[i+1] = s1.h;
            auto s2 = this.lstm2(h1[i+1], h2[i], c2[i]);
            h2[i+1] = s2.h;
            c2[i+1] = s2.c;
            auto y = this.linear(h2[i+1]);
            auto loss = crossEntropy(y, xs[i+1]);
            loss.backward();
            ret += y.shape[0] * loss.to!HostStorage.data[0];
        }
        return ret / xslice.length!0;
    }
}

void main(string[] args) {
    debug test();

    import mir.math : exp;
    import std.stdio : writeln;
    import std.getopt : getopt, defaultGetoptPrinter;
    import std.format : format;

    version (grain_cuda) alias Storage = DeviceStorage;
    else alias Storage = HostStorage;

    auto dataset = "data";
    auto embed = 200;
    auto hidden = 200;
    auto batchSize = 20;
    auto bptt = 35;
    auto lr = 0.1;

    auto opt = getopt(
        args,
        "dataset", format!"dataset path (default %s)"(dataset), &dataset,
        "embed", format!"embed size (default %s)"(embed), &embed,
        "hidden", format!"hidden size (default %s)"(hidden), &hidden,
        "batchSize", format!"batch size (default %s)"(batchSize), &batchSize,
        "bptt", format!"backprop through time size (default %s)"(bptt), &bptt,
        "lr", format!"learning rate of optimizer (default %s)"(lr), &lr
        );

    if (opt.helpWanted) {
        defaultGetoptPrinter("Image recognition example on CIFAR10/100",
                             opt.options);
        return;
    }

    auto corpus = prepareDataset();
    writeln("vocab:", corpus.vocabSize);
    auto model = RNNLM!Storage(corpus.vocabSize, embed, hidden);
    iterVariables!((k, v) { writeln(k, v.shape); })(&model);

    auto optimizer = make!SGD(model, lr);

    auto batch = corpus.batchfy("train");
    import grain.autograd;
    import std.algorithm : min;
    grain.autograd.backprop = true;
    auto total = batch.length!0;
    foreach (i; iota([total / bptt], 0, bptt)) {
        auto xs = batch[i .. min($, i + bptt)];
        model.zeroGrad();
        auto loss = model(xs, optimizer);
        writeln(loss.exp);
        optimizer.update();
    }
}
