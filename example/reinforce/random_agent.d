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
            state = env.step(action, false); // not render
            totalReward += state.reward;
            writeln(state.observation);
        }
        writefln!"total reward %f"(totalReward);
    }
}
