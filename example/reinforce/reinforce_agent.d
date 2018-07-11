/**
based on chainer impl https://github.com/ShigekiKarita/cupy-examples/blob/master/pole.py

TODO
- concat
- huber loss
- sum
- multinomial sampling in cuda
*/
import grain.autograd;
import grain.chain;
import grain.optim;
import grain.serializer;

import gym : Environment, Discrete;
import std.range : iota;
import std.random : choice;
import std.stdio : writefln, writeln;

struct Model(T, alias Storage) {
    alias L = Linear!(T, Storage);
    Convolution!(float, 2, Storage) conv;
    L fc1, fc2, fc3;

    this(int nin, int nhidden, int nout) {
        this.fc1 = L(nin, nhidden);
        this.fc2 = L(nhidden, nhidden);
        this.fc3 = L(nhidden, nout);
    }

    auto opCall(Variable!(T, 2, Storage) x) {
        auto h1 = relu(this.fc1(x));
        auto h2 = relu(this.fc2(h1));
        auto h3 = this.fc3(h2);
        return h3;
    }
}

void main() {
    auto env = Environment("127.0.0.1:5000", "CartPole-v0");
    env.record("/tmp/gym-d");
    scope (exit)
        env.stop();

    foreach (episode; 0 .. 10) {
        auto totalReward = 0.0;
        auto n = cast(size_t) Discrete.from(env.actionInfo);
        for (auto state = env.reset(); !state.done;) {
            auto action = choice(iota(n)); // left/right
            state = env.step(action, false); // not render
            totalReward += state.reward;
            writeln(state.observation);
        }
        writefln!"total reward %f"(totalReward);
    }
}
