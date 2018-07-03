/**
   A module of utility functions
 */
module grain.utility;

import std.typecons  : isTuple, tuple;

/// non-tuple to tuple. tuple to tuple
auto toTuple(T)(T t) {
    static if (isTuple!T) {
        return t;
    } else {
        return tuple(t);
    }
}

/// single element tuple to element. the other tuple to tuple
auto fromTuple(T)(T t) {
    static if (t.length == 0) {
        return t[0];
    } else {
        return t;
    }
}

/// unsafe cast of array (e.g., int[] -> size_t[])
Dst[N] castArray(Dst, Src, size_t N)(Src[N] src) {
    Dst[N] dst;
    static foreach (i; 0 .. N) {
        dst[i] = cast(Dst) src[i];
    }
    return dst;
}

version (LDC) {
    public import ldc.attributes : optStrategy;
} else {
    /// usage @optStrategy("none")
    struct optStrategy {
        string s;
    }
}
