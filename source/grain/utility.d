module grain.utility;

import std.typecons  : isTuple, tuple;


auto toTuple(T)(T t) {
    static if (isTuple!T) {
        return t;
    } else {
        return tuple(t);
    }
}

auto fromTuple(T)(T t) {
    static if (t.length == 0) {
        return t[0];
    } else {
        return t;
    }
}

Dst[N] castArray(Dst, Src, size_t N)(Src[N] src) {
    Dst[N] dst;
    static foreach (i; 0 .. N) {
        dst[i] = src[i];
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
