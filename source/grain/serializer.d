module grain.serializer;

import std.stdio;
import grain.autograd;

enum variableNames(C) = {
    string[] ret;
    void register(V)(string k, V v) if (isVariable!V) {
        ret ~= [k];
    }
    C chain;
    iterVariables!( (k, v) { register(k, v); })(&chain, "");
    return ret;
}();

///
unittest {
    import std.traits;
    auto mlp = MLP!(float, HostStorage)(3);
    static assert(variableNames!(typeof(mlp)) ==
                  [".fc1.weight", ".fc1.bias", ".fc2.weight", ".fc2.bias", ".fc3.weight", ".fc3.bias"]);
}

auto variableDict(C)(C chain) {
    UntypedVariable[string] ret;
    void register(V)(string k, V v) if (isVariable!V) {
        ret[k] = UntypedVariable(v);
    }
    iterVariables!( (k, v) { register(k, v); })(&chain, "");
    return ret;
}

unittest {
    import std.traits;
    auto mlp = MLP!(float, HostStorage)(3);
    auto keys = [".fc1.weight", ".fc1.bias",
                 ".fc2.weight", ".fc2.bias",
                 ".fc3.weight", ".fc3.bias"];
}

/// test .slice makes slice contiguous
unittest {
    import numir;
    import mir.ndslice;
    auto i = iota(3, 4, 5).transposed(1);
    assert(i.universal._strides == [5, 20, 1]);
    assert(i.slice.universal._strides == [15, 5, 1]);
}


version (unittest) {
    struct MLP(T, alias Storage) {
        import grain.autograd : Variable;
        import grain.chain : Linear, relu;

        alias L = Linear!(T, Storage);
        L fc1, fc2, fc3;

        this(int nhidden) {
            this.fc1 = L(2, nhidden);
            this.fc2 = L(nhidden, nhidden);
            this.fc3 = L(nhidden, 10);
        }

        auto opCall(Variable!(T, 2, Storage) x) {
            auto h1 = relu(this.fc1(x));
            auto h2 = relu(this.fc2(h1));
            auto h3 = this.fc3(h2);
            return h1;
        }
    }
}

void save(bool verbose = true, C)(C chain, string path) {
    import std.file : exists;
    import std.string : replace, endsWith;
    import mir.ndslice : slice;
    import hdf5.hdf5;
    import grain.utility : castArray;
    auto file = H5F.create(path,
                           // path.exists ? H5F_ACC_RDWR :
                           H5F_ACC_TRUNC,
                           H5P_DEFAULT, H5P_DEFAULT);
    scope(exit) H5F.close(file);
    // auto property = H5P.create (H5P_DATASET_CREATE);
    // H5P.set_alloc_time(property, H5DAllocTime.Early); // canbe Late
    // scope(exit) H5P.close(property);

    void register(V)(string k, V v) if (isVariable!V) {
        auto h = v.to!HostStorage;
        // FIXME support check contiguous
        // auto s = h.sliced.slice;
        auto data = v.to!HostStorage.data;
        auto space = H5S.create_simple(h.shape.castArray!hsize_t);
        scope(exit) H5S.close(space);
        auto h5key = "/" ~ k.replace(".", "_");
        // FIXME support non-float type
        auto dataset = H5D.create2(file, h5key, H5T_IEEE_F32LE, space,
                                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        scope(exit) H5D.close(dataset);
        H5D.write(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                  cast(ubyte*) data.ptr);
        // auto raw = new float[v.data.length];
        // H5D.read(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
        //          cast(ubyte*) raw.ptr);
    }
    iterVariables!( (k, v) { register(k, v); })(&chain, "");
}

void load(C)(ref C chain, string path) {
    import std.string : replace, endsWith;
    import mir.ndslice : slice, sliced;
    import hdf5.hdf5;
    import grain.utility : castArray;
    auto file = H5F.open(path,
                         // path.exists ? H5F_ACC_RDWR :
                         H5F_ACC_RDONLY, //
                         // H5F_ACC_RDWR,
                         H5P_DEFAULT);
    scope(exit) H5F.close(file);

    void register(T, size_t dim, alias Storage)(string k, ref Variable!(T, dim, Storage) v) {
        // writeln(k, v.sliced);
        auto h5key = "/" ~ k.replace(".", "_");
        // writeln(h5key);
        auto dataset = H5D.open2(file, h5key, H5P_DEFAULT);
        scope(exit) H5D.close(dataset);
        // FIXME support non-float type
        auto raw = new float[v.data.length];
        H5D.read(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                 cast(ubyte*) &raw[0]);
        auto src = raw.sliced(v.shape.castArray!size_t).variable;
        // TODO cuda support
        v.data[] = src.to!Storage.data;
        v.strides = src.strides;
    }
    refIterVariables!( (k, ref v) { register(k, v); })(chain, "");
}

unittest {
    import numir;
    auto model1 = MLP!(float, HostStorage)(3);
    model1.save("test_grain.h5");

    auto model2 = MLP!(float, HostStorage)(3);
    model2.load("test_grain.h5");
    assert(model1.fc1.bias.sliced == model2.fc1.bias.sliced);
}
