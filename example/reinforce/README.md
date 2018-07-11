# D openai gym

[![Build Status](https://travis-ci.org/ShigekiKarita/d-openai-gym.svg?branch=master)](https://travis-ci.org/ShigekiKarita/d-openai-gym)
[![Dub version](https://img.shields.io/dub/v/d-openai-gym.svg)](https://code.dlang.org/packages/d-openai-gym)

D-language binding / DUB package of https://github.com/openai/gym-http-api

## usage

random agent in example.d

``` d
import gym : Environment, Discrete;
import std.range : iota;
import std.random : choice;
import std.stdio : writefln, writeln;

void main() {
    auto env = Environment("127.0.0.1:5000", "CartPole-v0");
    env.record("/tmp/gym-d");
    scope(exit) env.stop();

    foreach (episode; 0 .. 10) {
        auto totalReward = 0.0;
        auto n = cast(size_t) Discrete.from(env.actionInfo);
        for (auto state = env.reset(); !state.done;) {
            auto action = choice(iota(n)); // left/right
            state = env.step(action, true); // render
            totalReward += state.reward;
            writeln(state);
        }
        writefln!"total reward %f"(totalReward);
    }
}
```

## how to run

``` bash
$ pip install -r requirements.txt
$ python gym_http_server.py &
$ rdmd example.d
$ kill %1
```
