module grain.utility;


Dst[N] castArray(Dst, Src, size_t N)(Src[N] src) {
    Dst[N] dst;
    static foreach (i; 0 .. N) {
        dst[i] = src[i];
    }
    return dst;
}

