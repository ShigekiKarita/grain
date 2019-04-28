/**
A module for serialization. HDF5 read/write is supported
 */
module grain.serializer;

import std.stdio;
import grain.autograd;

import grain.hdf5;
import std.string : toStringz;

shared static this() {
    H5open();
}


/// enumerate the parameter names inside chain C
enum variableNames(C) = {
    string[] ret;
    void register(V)(string k, V v) if (isVariable!V) {
        ret ~= [k];
    }

    C chain;
    iterVariables!((k, v) { register(k, v); })(&chain, "");
    return ret;
}();

///
unittest {
    import std.traits;

    auto mlp = MLP!(float, HostStorage)(3);
    static assert(variableNames!(typeof(mlp)) == [".fc1.weight", ".fc1.bias",
            ".fc2.weight", ".fc2.bias", ".fc3.weight", ".fc3.bias"]);
}

// test .slice makes slice contiguous
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

/// convert D type into HDF5 type-id https://support.hdfgroup.org/HDF5/doc1.8/RM/PredefDTypes.html
auto toH5Type(T)() {
    import std.traits;
    import std.format;

    static assert(isBasicType!T);
    mixin("return H5T_%s%dLE;".format(isFloatingPoint!T ? "IEEE_F"
            : (isSigned!T ? "STD_I" : "STD_U"), T.sizeof * 8));
}

/// save chain parameters to HDF5 path
void save(bool verbose = true, C)(C chain, string path) {
    import std.file : exists;
    import std.string : replace, endsWith;
    import mir.ndslice : slice;
    import grain.utility : castArray;

    auto file = H5Fcreate(path.toStringz, // path.exists ? H5F_ACC_TRUNC :
                          // H5F_ACC_RDWR, //
                          H5F_ACC_TRUNC,
                          H5P_DEFAULT, H5P_DEFAULT);
    scope (exit)
        H5Fclose(file);

    void register(T, size_t dim, alias Storage)(string k, Variable!(T, dim, Storage) v) {
        auto h = v.to!HostStorage;
        // FIXME support check contiguous
        // auto s = h.sliced.slice;
        auto data = v.to!HostStorage.data;
        auto dims = h.shape.castArray!hsize_t;
        auto space = H5Screate_simple(cast(int) dims.length, dims.ptr, dims.ptr);
        scope (exit)
            H5Sclose(space);
        auto dataset = H5Dcreate2(file, toStringz("/" ~ k), toH5Type!T, space,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        scope (exit)
            H5Dclose(dataset);
        H5Dwrite(dataset, toH5Type!T, H5S_ALL, H5S_ALL, H5P_DEFAULT, cast(void*) data
                .ptr);
    }

    iterVariables!((k, v) { register(k, v); })(&chain, "");
}

/// load chain parameters from HDF5 path
void load(C)(ref C chain, string path) {
    import std.string : replace, endsWith;
    import mir.ndslice : slice, sliced;

    import grain.utility : castArray;

    auto file = H5Fopen(path.toStringz, // path.exists ? H5F_ACC_RDWR :
            H5F_ACC_RDONLY, //
            // H5F_ACC_RDWR,
            H5P_DEFAULT);
    scope (exit)
        H5Fclose(file);

    void register(T, size_t dim, alias Storage)(string k, ref Variable!(T, dim, Storage) v) {
        auto dataset = H5Dopen2(file, toStringz("/" ~ k), H5P_DEFAULT);
        scope (exit)
            H5Dclose(dataset);

        auto raw = new T[v.data.length];
        H5Dread(dataset, toH5Type!T, H5S_ALL, H5S_ALL, H5P_DEFAULT, cast(void*)&raw[
                0]);

        auto src = raw.sliced(v.shape.castArray!size_t).variable;
        static if (is(Storage!T == HostStorage!T)) {
            v.sliced[] = src.sliced;
        }
        else {
            import grain.cudnn : transform;

            transform(src.to!Storage, v);
        }
    }

    refIterVariables!((k, ref v) { register(k, v); })(chain, "");
}

///
unittest {
    auto model1 = MLP!(float, HostStorage)(3);
    model1.save("/tmp/test_grain0.h5");

    auto model2 = MLP!(float, HostStorage)(3);
    model2.load("/tmp/test_grain0.h5");
    assert(model1.fc1.bias.sliced == model2.fc1.bias.sliced);

    import numir;
    import mir.ndslice;

    auto x = uniform!float(3, 2).slice.variable;
    assert(model1(x).sliced == model2(x).sliced);
}

///
version (grain_cuda) unittest {
    auto model1 = MLP!(float, DeviceStorage)(3);
    model1.save("/tmp/test_grain1.h5");

    auto model2 = MLP!(float, DeviceStorage)(3);
    model2.load("/tmp/test_grain1.h5");
    assert(model1.fc1.bias.to!HostStorage.sliced == model2.fc1.bias.to!HostStorage
            .sliced);

    import numir;
    import mir.ndslice;

    auto x = uniform!float(3, 2).slice.variable.to!DeviceStorage;
    assert(model1(x).to!HostStorage.sliced == model2(x).to!HostStorage.sliced);
}

///
version (grain_cuda) unittest {
    auto model1 = MLP!(float, HostStorage)(3);
    model1.save("/tmp/test_grain2.h5");

    auto model2 = MLP!(float, DeviceStorage)(3);
    model2.load("/tmp/test_grain2.h5");
    assert(model1.fc1.bias.to!HostStorage.sliced == model2.fc1.bias.to!HostStorage
            .sliced);
}

///
version (grain_cuda) unittest {
    auto model1 = MLP!(float, DeviceStorage)(3);
    model1.save("/tmp/test_grain3.h5");

    auto model2 = MLP!(float, HostStorage)(3);
    model2.load("/tmp/test_grain3.h5");
    assert(model1.fc1.bias.to!HostStorage.sliced == model2.fc1.bias.to!HostStorage
            .sliced);
}
