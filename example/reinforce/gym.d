/// based on https://github.com/openai/gym-http-api/blob/master/binding-rust/src/lib.rs

module gym;
import std.array : array;
import std.algorithm : map;
import std.exception : enforce;
import std.json : JSONValue, parseJSON, JSON_TYPE;
import std.conv : to;
import std.stdio : writeln;
import std.variant : Algebraic, This;
import std.net.curl;

enum bool isSpace(T) = is(typeof({ T.from(JSONValue.init); }));

/// Discrete Space
struct Discrete {
    long n;
    alias n this;

    static from(T: JSONValue)(scope auto ref T info) {
        enforce(info["name"].str == "Discrete");
        return Discrete(info["n"].integer);
    }
}

///
version (gym_test) unittest {
    static assert(isSpace!Discrete);

    auto s = `{
   "name": "Discrete",
   "n": 123
}`;
    auto j = s.parseJSON;
    assert(Discrete.from(j) == Discrete(123));
}

/// Box Space
struct Box {
    long[] shape;
    double[] high;
    double[] low;

    static from(T: JSONValue)(scope auto ref T info) {
        enforce(info["name"].str == "Box");
        return Box(
            info["shape"].array.map!(a => a.integer).array.dup,
            info["high"].array.map!(a => a.floating).array.dup,
            info["low"].array.map!(a => a.floating).array.dup
            );
    }
}

///
version (gym_test) unittest {
    static assert(isSpace!Box);

    auto s = `{
   "name": "Box",
   "shape": [2],
   "high": [1.0, 2.0],
   "low": [0.0, 0.0]
}`;
    auto j = s.parseJSON;
    assert(Box.from(j) == Box([2], [1.0, 2.0], [0.0, 0.0]));
}

///
struct State {
    JSONValue info;
    alias info this;

    @property
    const ref observation() {
        return this.info["observation"];
    }

    @property
    double reward() {
        if ("reward" in this.info) {
            return this.info["reward"].floating;
        } else {
            return 0;
        }
    }

    @property
    bool done() {
        if ("done" in this.info) {
            return this.info["done"].type == JSON_TYPE.TRUE;
        } else {
            return false;
        }
    }
}

///
struct Environment {
    import std.format : format;

    string address, id;
    string instanceId;
    JSONValue actionInfo;
    JSONValue observationInfo;

    auto _post(T: JSONValue)(scope auto ref T req) const {
        auto client = HTTP();
        client.addRequestHeader("Content-Type", "application/json");
        return post(this.address ~ "/v1/envs/", req.toString, client).parseJSON;
    }

    auto _post(T: JSONValue)(string loc, scope auto ref T req) const {
        auto client = HTTP();
        client.addRequestHeader("Content-Type", "application/json");
        return post(this.address ~ "/v1/envs/" ~ this.instanceId ~ "/%s/".format(loc),
                    req.toString, client).parseJSON;
    }

    auto _get(string loc) const {
        return get(this.address ~ "/v1/envs/" ~ this.instanceId ~ "/%s/".format(loc))
            .parseJSON;
    }

    this(string address, string id) {
        this.address = address;
        this.id = id;
        this.instanceId = this._post(JSONValue(["env_id": this.id]))["instance_id"].str;
        this.observationInfo = this._get("observation_space")["info"];
        this.actionInfo = this._get("action_space")["info"];
    }

    auto reset() {
        return State(this._post("reset", JSONValue(null)));
    }

    /// step by json action e.g., 0, [1.0, 2.0, ...], etc
    auto step(A)(A action, bool render = false) {
        JSONValue req;
        req["render"] = render;
        req["action"] = action;
        auto ret = this._post("step", req);
        return State(ret);
    }

    auto record(string dir, bool force = true, bool resume = false) {
        JSONValue req;
        req["directory"] = dir;
        req["force"] = force;
        req["resume"] = resume;
        return this._post("monitor/start", req);
    }

    auto stop() {
        return this._post("monitor/close", JSONValue());
    }

    auto upload(string dir, string apiKey, string algorithmId) {
        JSONValue req = [
            "training_dir": dir,
            "api_key": apiKey,
            "algorithm_id": algorithmId
            ];
        auto client = HTTP();
        client.addRequestHeader("Content-Type", "application/json");
        return post(this.address ~ "/v1/upload/",
                    req.toString, client).parseJSON;
    }
}

/// simple integration test
version (gym_test) unittest {
    {
        auto env = Environment("127.0.0.1:5000", "CartPole-v0");
        assert(Discrete.from(env.actionInfo) == 2);
        auto o = Box.from(env.observationInfo);
        assert(o.shape == [4]);
        assert(o.low.length == 4);
        env.record("/tmp/d-gym");
        scope(exit) env.stop();

        auto state = env.reset;
        double reward = 0;
        while (!state.done) {
            state = env.step(Discrete(0));
            reward += state.reward;
        }
        assert(reward > 0);
    }
    // {
    //     auto env = Environment("127.0.0.1:5000", "MsPacman-v0");
    //     assert(Discrete.from(env.actionInfo) == 9);
    //     auto o = Box.from(env.observationInfo);
    //     assert(o.shape == [210, 160, 3]);
    //     assert(o.high.length == 210 * 160 * 3);
    //     auto a = Discrete.from(env.actionInfo);
    //     assert(a == 9);
    // }
}
