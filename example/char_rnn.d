/++
Character Recurrent Neural Networks in numir.

See_Also:

Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
https://gist.github.com/karpathy/d4dee566867f8291f086
 +/

import std.array : array;
import std.stdio;
import std.datetime.stopwatch; //  : StopWatch, seconds, TickDuration;
import std.conv : to;
import std.file : readText, exists;
import std.algorithm : stdmap = map;
import std.typecons : tuple;
import std.net.curl : get;

import mir.math.common : log, exp, sqrt, fastmath;
import mir.random : Random, unpredictableSeed;
import mir.random.variable : discreteVar;
import std.math : tanh;
import mir.ndslice : slice, sliced, map, transposed, ndarray;
import mir.math.sum : sum;
import lubeck : mtimes;
import numir;

// pragma(inline, true)
// @fastmath
void dger(S, R)(S x, S y, R A, double alpha=1.0) if (Ndim!S == 2)
in {
    assert(x.length!1 == 1);
    assert(y.length!1 == 1);
} do {
    import mir.blas : ger;
    ger(1.0, x[0..$, 0], y[0..$,0], A);
}

/++
Params:
    inputs = array of char-id integers.
    targets = array of char-id integers.
    hprev = Hx1 array of initial hidden state
Returns:
    the loss, gradients on model parameters, and last hidden state
 +/
// @fastmath
auto lossFun(STuple, X, T, H)(STuple params, X inputs, T targets, H hprev) {
    import std.algorithm : clamp;
    with (params) {
        auto xs = zeros(inputs.length, Wxh.length!1, 1);
        auto hs = zeros(inputs.length, bh.length!0, 1);
        auto ys = zeros(inputs.length, by.length!0, 1);
        auto ps = zeros_like(ys);
        double loss = 0;
        // forward pass
        foreach (t, i; inputs) {
            xs[t][i, 0] = 1; // encode in 1-of-k reps
            auto hp = t == 0 ? hprev : hs[t-1];
            hs[t][] = map!tanh(mtimes(Wxh, xs[t]) + mtimes(Whh, hp) + bh); // hidden state
            ys[t][] = mtimes(Why, hs[t]) + by; // unnormalized log probabilities for next chars
            ps[t][] = map!exp(ys[t]);
            ps[t][] /= ps[t].sum!"fast"; // probabilities for next chars
            loss += -log(ps[t][targets[t], 0]); // softmax (cross-entropy loss)
        }

        // backward pass: compute gradients of going backwards
        STuple grads;
        foreach (k, v; params) {
            grads[k] = zeros_like(v);
        }
        auto dhnext = zeros_like(hs[0]);
        foreach_reverse (t, i; inputs) {
            auto dy = ps[t];
            dy[targets[t]][] -= 1; // backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
            dger(dy, hs[t], grads.Why);
            // grads["Why"][] += mtimes(dy, hs[t].transposed);
            grads.by[] += dy;
            auto dh = mtimes(Why.transposed, dy); // backprop into h
            dh[] += dhnext;
            dh[] *= (1.0 - hs[t] * hs[t]); // backprop throgh tanh nonlinearity
            grads.bh[] += dh;
            dger(dh, xs[t], grads.Wxh);
            auto hp = t == 0 ? hprev : hs[t-1];
            dger(dh, hp, grads.Whh);
            dhnext[] = mtimes(Whh.transposed, dh);
        }
        foreach (v; grads) {
            v[] = v.map!(a => clamp(a, -5, 5)); // clip to mitigate exploding gradients
        }
        return tuple!("loss", "grads")(loss, grads);
    }
}


/++
Params:
    params = RNN model parameters
    h = memory state
    seed_ix = seed letter for first time step
Returns:
    a sampled sequence of integers from the model
 +/
//@fastmath
auto sample(STuple, S)(STuple params, S h, size_t seed_ix, size_t n) {
    auto gen = Random(unpredictableSeed);
    auto x = zeros(params.Wxh.length!1, 1);
    x[seed_ix][] = 1;
    size_t[] ixes;
    ixes.length = n;
    foreach (t; 0 .. n) {
        h[] = map!tanh(mtimes(params.Wxh, x) + mtimes(params.Whh, h) + params.bh);
        auto y = mtimes(params.Why, h) + params.by;
        auto p = map!exp(y).slice;
        p[] /= p.sum;
        auto ix = discreteVar(p.squeeze!1.ndarray)(gen);
        x[] = 0;
        x[ix][] = 1;
        ixes[t] = ix;
    }
    return ixes;
}

/**
   TODO
   1. tanh
   2. EmbedId
   3. opBinary!"+" (add)
   4. opSlice
 */
struct RNN(Storage) {
    import grain.chain;
    alias L = Linear!(float, Storage);
    EmbedId!(T, Storage) wx;
    L wh, wy;
    int nunits;

    this(int nvocab, int nunits) {
        this.nunits = nunits;
        this.wx = EmbedId!(T, Storage)(nvocab, nunits);
        this.wh = L(nunits, nunits);
        this.wy = L(nunits, nvocab);
    }

    /// framewise batch input
    auto opCall(Variable!(int, 1, Storage) x, Variable!(float, 1, Storage) hprev) {
        import std.typecons : tuple;
        auto h = tanh(this.wx(this.embed(x)) + this.wh(hprev));
        auto y = this.wy(h);
        return tuple!("y", "h")(y, h);
    }

    /// batch x frame input
    auto loss(Variable!(int, 2, Storage) xs) {
        auto hprev = zeros(xs.shape[0], nunits).variable(true).to!Storage;
        auto loss = 0f.variable(true).to!Storage;
        foreach (t; 0 .. xs.shape[1]-1) {
            auto next = this(xs[0..$, t], hprev);
            hprev = next.h;
            loss = loss + crossEntropy(next.y, xs[0..$, t+1]);
        }
        return loss;
    }
}

void main() {
    // data I/O
    if (!"data/".exists) {
        import std.file : mkdir;
        mkdir("data");
    }
    if (!"data/input.txt".exists) {
        import std.net.curl : download;
        download("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", "data/input.txt");
    }
    auto data = readText("data/input.txt").to!dstring;
    size_t[dchar] char_to_ix;
    dchar[] ix_to_char;
    foreach (c; data) {
        if (c !in char_to_ix) {
            char_to_ix[c] = ix_to_char.length;
            ix_to_char ~= [c];
        }
    }
    auto vocab_size = ix_to_char.length;
    writefln!"data has %d characters, %d unique."(data.length, vocab_size);

    // hyperparameters
    auto hidden_size = 100; // size of hidden layer of neurons
    auto seq_length = 25;   // number of steps to unroll the RNN for
    auto learning_rate = 1e-1;
    auto max_iter = 10000;
    auto log_iter = max_iter / 100;

    // model parameters
    auto params = tuple!("Wxh", "Whh", "Why", "bh", "by")
        ((normal(hidden_size, vocab_size) * 0.01).slice,  // input to hidden
         (normal(hidden_size, hidden_size) * 0.01).slice, // hidden to hidden
         (normal(vocab_size, hidden_size) * 0.01).slice,  // hidden to output
         zeros(hidden_size, 1), // hidden bias
         zeros(vocab_size, 1)   // output bias
         );

    // memory variables for Adagrad
    typeof(params) memory;
    foreach (k, v; params) {
        memory[k] = zeros_like(v);
    }
    auto smooth_loss = -log(1.0 / vocab_size) * seq_length;
    size_t begin_id = 0;
    auto hprev = zeros(hidden_size, 1);
    auto sw = StopWatch(AutoStart.yes);
    foreach (n_iter; 0 .. max_iter) {
        // prepare inputs (we're sweeping from left to right in steps seq_length long)
        if (begin_id + seq_length + 1 >= data.length || n_iter == 0) {
            hprev[] = 0; // reset RNN memory
            begin_id = 0; // go from start of data
        }
        auto raw = data[begin_id .. begin_id + seq_length + 1].stdmap!(c => char_to_ix[c]).array;
        auto inputs = raw[0 .. $-1];
        auto targets = raw[1 .. $];
        // sample from the model now and then
        if (n_iter % log_iter == 0) {
            auto sample_ix = sample(params, hprev, inputs[0], 200);
            auto txt = sample_ix.stdmap!(ix => ix_to_char[ix]).to!dstring;
            writeln("-----\n", txt, "\n-----");
        }

        // forward seq_length characters through the net and fetch gradient
        auto results = lossFun(params, inputs, targets, hprev);
        smooth_loss = smooth_loss * 0.999 + results.loss * 0.001;
        if (n_iter % log_iter == 0) {
            writefln!"iter %d, loss: %f, iter/sec: %f"(
                n_iter, smooth_loss,
                cast(double) log_iter / (sw.peek().to!TickDuration.msecs * 1e-3));
            sw.reset();
        }
        foreach (k, v; params) {
            memory[k][] += results.grads[k] * results.grads[k];
            params[k][] -= learning_rate * results.grads[k] / (memory[k] + 1e-8).map!sqrt; // adagrad update
        }
        begin_id += seq_length; // move data pointer
    }
}
