/++
Character Recurrent Neural Networks in numir.

See_Also:

Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
https://gist.github.com/karpathy/d4dee566867f8291f086
 +/

import std.array : array;
import std.stdio;
import std.datetime.stopwatch; //  : StopWatch, seconds, TickDuration;
// import std.conv : to;
import std.file : readText, exists;
import std.algorithm : stdmap = map;
import std.typecons : tuple;
import std.net.curl : get;

import mir.math.common : log; // , exp, sqrt, fastmath;
// import mir.random : Random, unpredictableSeed;
// import mir.random.variable : discreteVar;
// import std.math : tanh;
// import mir.ndslice : slice, sliced, map, transposed, ndarray;
// import mir.math.sum : sum;
// import lubeck : mtimes;
import numir;

import mir.ndslice;
import grain.autograd;
import grain.optim;



// /++
// Params:
//     params = RNN model parameters
//     h = memory state
//     seed_ix = seed letter for first time step
// Returns:
//     a sampled sequence of integers from the model
//  +/
// //@fastmath
// auto sample(STuple, S)(STuple params, S h, size_t seed_ix, size_t n) {
//     auto gen = Random(unpredictableSeed);
//     auto x = zeros(params.Wxh.length!1, 1);
//     x[seed_ix][] = 1;
//     size_t[] ixes;
//     ixes.length = n;
//     foreach (t; 0 .. n) {
//         h[] = map!tanh(mtimes(params.Wxh, x) + mtimes(params.Whh, h) + params.bh);
//         auto y = mtimes(params.Why, h) + params.by;
//         auto p = map!exp(y).slice;
//         p[] /= p.sum;
//         auto ix = discreteVar(p.squeeze!1.ndarray)(gen);
//         x[] = 0;
//         x[ix][] = 1;
//         ixes[t] = ix;
//     }
//     return ixes;
// }


struct RNN(alias Storage, T=float) {
    import grain.chain;
    alias L = Linear!(T, Storage);
    Embedding!(T, Storage) wx;
    L wh, wy;

    this(uint nvocab, int nunits) {
        this.wx = Embedding!(T, Storage)(nvocab, nunits);
        this.wh = L(nunits, nunits);
        this.wy = L(nunits, nvocab);
    }

    /// framewise batch input
    auto opCall(Variable!(int, 1, Storage) x, Variable!(float, 2, Storage) hprev) {
        import std.typecons : tuple;
        auto h = tanh(this.wx(x) + this.wh(hprev));
        auto y = this.wy(h);
        return tuple!("y", "h")(y, h);
    }

    /// batch x frame input
    auto loss(Slice!(Universal, [2L], int*) xs, Variable!(float, 2, Storage) hprev) {
        auto loss = 0f.variable.to!Storage;
        auto ret = new Variable!(float, 0, Storage)[xs.length!1 - 1];
        foreach (t; 0 .. xs.length!1 - 1) {
            auto next = this.opCall(xs[0..$, t].variable(true).to!Storage, hprev);
            hprev = next.h;
            ret[t] = crossEntropy(next.y, xs[0..$, t+1].variable.to!Storage);
        }
        return ret;
    }
}

void main() {
    import C = std.conv;
    // data I/O
    if (!"data/".exists) {
        import std.file : mkdir;
        mkdir("data");
    }
    if (!"data/input.txt".exists) {
        import std.net.curl : download;
        download("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", "data/input.txt");
    }
    auto data = C.to!dstring(readText("data/input.txt"));
    int[dchar] char2idx;
    dchar[] idx2char;
    foreach (c; data) {
        if (c !in char2idx) {
            char2idx[c] = cast(int) idx2char.length;
            idx2char ~= [c];
        }
    }
    auto vocabSize = cast(uint) idx2char.length;
    writefln!"data has %d characters, %d unique."(data.length, vocabSize);

    // hyperparameters
    auto hiddenSize = 100; // size of hidden layer of neurons
    auto seqLength = 25;   // number of steps to unroll the RNN for
    auto learningRate = 1e-1;
    auto maxIter = 10000;
    auto logIter = maxIter / 100;
    auto batchSize = 1;

    // model parameters
    alias Storage = HostStorage;
    auto model = RNN!Storage(vocabSize, hiddenSize);

    // for optim
    SGD optim = { lr: learningRate };
    auto smoothLoss = -log(1.0 / vocabSize) * seqLength;
    size_t beginId = 0;
    auto hprev = zeros!float(batchSize, hiddenSize).variable(true).to!Storage;
    auto sw = StopWatch(AutoStart.yes);
    foreach (nIter; 0 .. maxIter) {
        // prepare inputs (we're sweeping from left to right in steps seq_length long)
        if (beginId + seqLength + 1 >= data.length || nIter == 0) {
            // reset RNN memory
            hprev = zeros!float(batchSize, hiddenSize).variable(true).to!Storage;
            beginId = 0; // go from start of data
        }
        auto ids = data[beginId .. beginId + seqLength + 1].stdmap!(c => char2idx[c]).array;
        // // sample from the model now and then
        // if (nIter % logIter == 0) {
        //     auto sample_ix = sample(params, hprev, ids[0], 200);
        //     auto txt = sample_ix.stdmap!(ix => ix_to_char[ix]).to!dstring;
        //     writeln("-----\n", txt, "\n-----");
        // }

        // forward seq_length characters through the net and fetch gradient
        auto ret = model.loss(ids.sliced.unsqueeze!0, hprev);
        model.zeroGrad();
        auto sumLoss = 0.0;
        foreach (loss; ret) {
            loss.backward();
            sumLoss += loss.to!HostStorage.data[0];
        }
        optim.update(model);
        smoothLoss = smoothLoss * 0.999 + sumLoss * 0.001;
        if (nIter % logIter == 0) {
            writefln!"iter %d, loss: %f, iter/sec: %f"(
                nIter, smoothLoss,
                cast(double) logIter / (C.to!(TickDuration)(sw.peek()).msecs * 1e-3));
            sw.reset();
        }
        // foreach (k, v; params) {
        //     memory[k][] += results.grads[k] * results.grads[k];
        //     params[k][] -= learning_rate * results.grads[k] / (memory[k] + 1e-8).map!sqrt; // adagrad update
        // }
        beginId += seqLength; // move data pointer
    }
}
