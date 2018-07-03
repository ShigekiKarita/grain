module grain.functions.topology;

import std.stdio;

import numir;
import mir.ndslice;

import grain.autograd;
import grain.utility;
import grain.testing : gradCheck;
import grain.functions.common;

/*
  mir.ndslice shape/strides test
 */
unittest {
    auto info(S, P)(S s, P base) {
        writefln!"Slice(ptr: %s, shape: %s, strides: %s)"(s._iterator - base, s._lengths, s._strides);
    }

    auto s = iota(3, 4, 5).slice.universal;
    auto ptr = s._iterator;

    assert(s._iterator == ptr);
    assert(s._lengths == [3, 4, 5]);
    assert(s._strides == [20, 5, 1]);

    auto t = s.swapped(0, 1);
    assert(t._iterator == ptr);
    assert(t._lengths == [4, 3, 5]);
    assert(t._strides == [5, 20, 1]);

    auto s0 = s[0];
    assert(s0._iterator == ptr);
    assert(s0._lengths == [4, 5]);
    assert(s0._strides == [5, 1]);

    auto s1 = s[0..$, 0];
    assert(s1._iterator == ptr);
    assert(s1._lengths == [3, 5]);
    assert(s1._strides == [20, 1]);

    auto s2 = s[0..$, 0..$, 0];
    assert(s2._iterator == ptr);
    assert(s2._lengths == [3, 4]);
    assert(s2._strides == [20, 5]);

    auto r0 = s.reversed!0;
    assert(r0._iterator == ptr + 40);
    assert(r0._lengths == [3, 4, 5]);
    assert(r0._strides == [-20, 5, 1]);

    auto ra = s.allReversed;
    assert(ra._iterator == ptr + 59);
    assert(ra._lengths == [3, 4, 5]);
    assert(ra._strides == [-20, -5, -1]);

    auto v = s.view(3, -1);
    assert(v._iterator == ptr);
    assert(v._lengths == [3, 20]);
    assert(v._strides == [20, 1]);
}

auto prod(T)(T x) {
    return reduce!"a * b"(1L, x.sliced);
}


// Reshaping or viewing to the other shape
struct View(T, size_t sourceDim, size_t targetDim, alias Storage) {
    import numir : view;
    ptrdiff_t[targetDim] targetShape;
    ptrdiff_t[sourceDim] sourceShape;

    auto forward(Variable!(T, sourceDim, Storage) x) {
        // assert(x.shape[].prod == targetShape[].prod);
        this.sourceShape = x.shape.castArray!ptrdiff_t; // TODO if train
        auto y = x.sliced.view(targetShape);
        return Variable!(T, targetDim, Storage)(
            x.requiresGrad,
            y.shape.castArray!uint,
            y.strides.castArray!int,
            x.data
            );
    }

    auto backward(Variable!(T, targetDim, Storage) gy) {
        auto gx = gy.sliced.view(this.sourceShape);
        return Variable!(T, sourceDim, Storage)(
            gy.requiresGrad,
            gx.shape.castArray!uint,
            gx.strides.castArray!int,
            gy.data
            );
    }

    mixin FunctionCommon;
}

///
unittest {
    auto f = View!(float, 3, 2, HostStorage)([3, -1]);
    auto x = iota(3, 4, 5).as!float.slice.variable;
    auto y = f.forward(x);
    assert(y.sliced == iota(3, 20));
    auto hgy = uniform!float(3, 20).slice.variable;
    auto hgx = f.backward(hgy);
    assert(hgy.sliced.view(3, 4, 5) == hgx.sliced);
    // gradCheck(f, x, hgy);

    version (grain_cuda) {
        auto df = View!(float, 3, 2, DeviceStorage)([3, -1]);
        auto dy = df.forward(x.to!DeviceStorage);
        assert(dy.to!HostStorage.sliced == iota(3, 20));
        auto dgx = df.backward(hgy.to!DeviceStorage);
        assert(dgx.to!HostStorage.sliced.view(3, 4, 5) == hgx.sliced);
    }
}


