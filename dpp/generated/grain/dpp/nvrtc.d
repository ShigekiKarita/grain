module grain.dpp.nvrtc;


        import core.stdc.config;
        import core.stdc.stdarg: va_list;
        static import core.simd;
        static import std.conv;

        struct Int128 { long lower; long upper; }
        struct UInt128 { ulong lower; ulong upper; }

        struct __locale_data { int dummy; }



alias _Bool = bool;
struct dpp {
    static struct Opaque(int N) {
        void[N] bytes;
    }

    static bool isEmpty(T)() {
        return T.tupleof.length == 0;
    }
    static struct Move(T) {
        T* ptr;
    }


    static auto move(T)(ref T value) {
        return Move!T(&value);
    }
    mixin template EnumD(string name, T, string prefix) if(is(T == enum)) {
        private static string _memberMixinStr(string member) {
            import std.conv: text;
            import std.array: replace;
            return text(` `, member.replace(prefix, ""), ` = `, T.stringof, `.`, member, `,`);
        }
        private static string _enumMixinStr() {
            import std.array: join;
            string[] ret;
            ret ~= "enum " ~ name ~ "{";
            static foreach(member; __traits(allMembers, T)) {
                ret ~= _memberMixinStr(member);
            }
            ret ~= "}";
            return ret.join("\n");
        }
        mixin(_enumMixinStr());
    }
}

extern(C)
{
    nvrtcResult nvrtcGetLoweredName(_nvrtcProgram*, const(const(char)*), const(char)**) @nogc nothrow;
    nvrtcResult nvrtcAddNameExpression(_nvrtcProgram*, const(const(char)*)) @nogc nothrow;
    nvrtcResult nvrtcGetProgramLog(_nvrtcProgram*, char*) @nogc nothrow;
    nvrtcResult nvrtcGetProgramLogSize(_nvrtcProgram*, c_ulong*) @nogc nothrow;
    nvrtcResult nvrtcGetPTX(_nvrtcProgram*, char*) @nogc nothrow;
    nvrtcResult nvrtcGetPTXSize(_nvrtcProgram*, c_ulong*) @nogc nothrow;
    nvrtcResult nvrtcCompileProgram(_nvrtcProgram*, int, const(const(char)*)*) @nogc nothrow;
    nvrtcResult nvrtcDestroyProgram(_nvrtcProgram**) @nogc nothrow;
    nvrtcResult nvrtcCreateProgram(_nvrtcProgram**, const(char)*, const(char)*, int, const(const(char)*)*, const(const(char)*)*) @nogc nothrow;
    struct _nvrtcProgram;
    alias nvrtcProgram = _nvrtcProgram*;
    nvrtcResult nvrtcVersion(int*, int*) @nogc nothrow;
    const(char)* nvrtcGetErrorString(nvrtcResult) @nogc nothrow;
    enum _Anonymous_0
    {
        NVRTC_SUCCESS = 0,
        NVRTC_ERROR_OUT_OF_MEMORY = 1,
        NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
        NVRTC_ERROR_INVALID_INPUT = 3,
        NVRTC_ERROR_INVALID_PROGRAM = 4,
        NVRTC_ERROR_INVALID_OPTION = 5,
        NVRTC_ERROR_COMPILATION = 6,
        NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
        NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
        NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
        NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
        NVRTC_ERROR_INTERNAL_ERROR = 11,
    }
    enum NVRTC_SUCCESS = _Anonymous_0.NVRTC_SUCCESS;
    enum NVRTC_ERROR_OUT_OF_MEMORY = _Anonymous_0.NVRTC_ERROR_OUT_OF_MEMORY;
    enum NVRTC_ERROR_PROGRAM_CREATION_FAILURE = _Anonymous_0.NVRTC_ERROR_PROGRAM_CREATION_FAILURE;
    enum NVRTC_ERROR_INVALID_INPUT = _Anonymous_0.NVRTC_ERROR_INVALID_INPUT;
    enum NVRTC_ERROR_INVALID_PROGRAM = _Anonymous_0.NVRTC_ERROR_INVALID_PROGRAM;
    enum NVRTC_ERROR_INVALID_OPTION = _Anonymous_0.NVRTC_ERROR_INVALID_OPTION;
    enum NVRTC_ERROR_COMPILATION = _Anonymous_0.NVRTC_ERROR_COMPILATION;
    enum NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = _Anonymous_0.NVRTC_ERROR_BUILTIN_OPERATION_FAILURE;
    enum NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = _Anonymous_0.NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION;
    enum NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = _Anonymous_0.NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION;
    enum NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = _Anonymous_0.NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID;
    enum NVRTC_ERROR_INTERNAL_ERROR = _Anonymous_0.NVRTC_ERROR_INTERNAL_ERROR;
    alias nvrtcResult = _Anonymous_0;
    alias fsfilcnt_t = c_ulong;
    alias fsblkcnt_t = c_ulong;
    alias blkcnt_t = c_long;
    alias blksize_t = c_long;
    alias register_t = c_long;
    alias u_int64_t = c_ulong;
    alias u_int32_t = uint;
    alias u_int16_t = ushort;
    alias u_int8_t = ubyte;
    alias key_t = int;
    alias caddr_t = char*;
    alias daddr_t = int;
    alias ssize_t = c_long;
    alias id_t = uint;
    alias pid_t = int;
    alias off_t = c_long;
    alias uid_t = uint;
    alias nlink_t = c_ulong;
    alias mode_t = uint;
    alias gid_t = uint;
    alias dev_t = c_ulong;
    alias ino_t = c_ulong;
    alias loff_t = c_long;
    alias fsid_t = __fsid_t;
    alias u_quad_t = c_ulong;
    alias quad_t = c_long;
    alias u_long = c_ulong;
    alias u_int = uint;
    alias u_short = ushort;
    alias u_char = ubyte;
    c_ulong gnu_dev_makedev(uint, uint) @nogc nothrow;
    uint gnu_dev_minor(c_ulong) @nogc nothrow;
    uint gnu_dev_major(c_ulong) @nogc nothrow;
    int pselect(int, fd_set*, fd_set*, fd_set*, const(timespec)*, const(__sigset_t)*) @nogc nothrow;
    int select(int, fd_set*, fd_set*, fd_set*, timeval*) @nogc nothrow;
    alias fd_mask = c_long;
    struct fd_set
    {
        c_long[16] __fds_bits;
    }
    alias __fd_mask = c_long;
    alias suseconds_t = c_long;
    enum _Anonymous_1
    {
        P_ALL = 0,
        P_PID = 1,
        P_PGID = 2,
    }
    enum P_ALL = _Anonymous_1.P_ALL;
    enum P_PID = _Anonymous_1.P_PID;
    enum P_PGID = _Anonymous_1.P_PGID;
    alias idtype_t = _Anonymous_1;
    static c_ulong __uint64_identity(c_ulong) @nogc nothrow;
    static uint __uint32_identity(uint) @nogc nothrow;
    static ushort __uint16_identity(ushort) @nogc nothrow;
    alias timer_t = void*;
    alias time_t = c_long;
    struct timeval
    {
        c_long tv_sec;
        c_long tv_usec;
    }
    struct timespec
    {
        c_long tv_sec;
        c_long tv_nsec;
    }
    alias sigset_t = __sigset_t;
    alias clockid_t = int;
    alias clock_t = c_long;
    struct __sigset_t
    {
        c_ulong[16] __val;
    }
    alias __sig_atomic_t = int;
    alias __socklen_t = uint;
    alias __intptr_t = c_long;
    alias __caddr_t = char*;
    alias __loff_t = c_long;
    alias __syscall_ulong_t = c_ulong;
    alias __syscall_slong_t = c_long;
    alias __ssize_t = c_long;
    alias __fsword_t = c_long;
    alias __fsfilcnt64_t = c_ulong;
    alias __fsfilcnt_t = c_ulong;
    alias __fsblkcnt64_t = c_ulong;
    alias __fsblkcnt_t = c_ulong;
    alias __blkcnt64_t = c_long;
    alias __blkcnt_t = c_long;
    alias __blksize_t = c_long;
    alias __timer_t = void*;
    alias __clockid_t = int;
    alias __key_t = int;
    alias __daddr_t = int;
    alias __suseconds_t = c_long;
    alias __useconds_t = uint;
    alias __time_t = c_long;
    alias __id_t = uint;
    alias __rlim64_t = c_ulong;
    alias __rlim_t = c_ulong;
    alias __clock_t = c_long;
    struct __fsid_t
    {
        int[2] __val;
    }
    alias __pid_t = int;
    alias __off64_t = c_long;
    alias __off_t = c_long;
    alias __nlink_t = c_ulong;
    alias __mode_t = uint;
    alias __ino64_t = c_ulong;
    alias __ino_t = c_ulong;
    alias __gid_t = uint;
    alias __uid_t = uint;
    alias __dev_t = c_ulong;
    alias __uintmax_t = c_ulong;
    alias __intmax_t = c_long;
    alias __u_quad_t = c_ulong;
    alias __quad_t = c_long;
    alias __uint64_t = c_ulong;
    alias __int64_t = c_long;
    alias __uint32_t = uint;
    alias __int32_t = int;
    alias __uint16_t = ushort;
    alias __int16_t = short;
    alias __uint8_t = ubyte;
    alias __int8_t = byte;
    alias __u_long = c_ulong;
    alias __u_int = uint;
    alias __u_short = ushort;
    alias __u_char = ubyte;
    pragma(mangle, "alloca") void* alloca_(c_ulong) @nogc nothrow;
    alias size_t = c_ulong;
    alias wchar_t = int;
    struct __pthread_cond_s
    {
        static union _Anonymous_2
        {
            ulong __wseq;
            static struct _Anonymous_3
            {
                uint __low;
                uint __high;
            }
            _Anonymous_3 __wseq32;
        }
        _Anonymous_2 _anonymous_4;
        auto __wseq() @property @nogc pure nothrow { return _anonymous_4.__wseq; }
        void __wseq(_T_)(auto ref _T_ val) @property @nogc pure nothrow { _anonymous_4.__wseq = val; }
        auto __wseq32() @property @nogc pure nothrow { return _anonymous_4.__wseq32; }
        void __wseq32(_T_)(auto ref _T_ val) @property @nogc pure nothrow { _anonymous_4.__wseq32 = val; }
        static union _Anonymous_5
        {
            ulong __g1_start;
            static struct _Anonymous_6
            {
                uint __low;
                uint __high;
            }
            _Anonymous_6 __g1_start32;
        }
        _Anonymous_5 _anonymous_7;
        auto __g1_start() @property @nogc pure nothrow { return _anonymous_7.__g1_start; }
        void __g1_start(_T_)(auto ref _T_ val) @property @nogc pure nothrow { _anonymous_7.__g1_start = val; }
        auto __g1_start32() @property @nogc pure nothrow { return _anonymous_7.__g1_start32; }
        void __g1_start32(_T_)(auto ref _T_ val) @property @nogc pure nothrow { _anonymous_7.__g1_start32 = val; }
        uint[2] __g_refs;
        uint[2] __g_size;
        uint __g1_orig_size;
        uint __wrefs;
        uint[2] __g_signals;
    }
    struct __pthread_mutex_s
    {
        int __lock;
        uint __count;
        int __owner;
        uint __nusers;
        int __kind;
        short __spins;
        short __elision;
        __pthread_internal_list __list;
    }
    struct __pthread_internal_list
    {
        __pthread_internal_list* __prev;
        __pthread_internal_list* __next;
    }
    alias __pthread_list_t = __pthread_internal_list;
    alias int64_t = c_long;
    alias int32_t = int;
    alias int16_t = short;
    alias int8_t = byte;
    union pthread_barrierattr_t
    {
        char[4] __size;
        int __align;
    }
    union pthread_barrier_t
    {
        char[32] __size;
        c_long __align;
    }
    alias pthread_spinlock_t = int;
    union pthread_rwlockattr_t
    {
        char[8] __size;
        c_long __align;
    }
    union pthread_rwlock_t
    {
        __pthread_rwlock_arch_t __data;
        char[56] __size;
        c_long __align;
    }
    union pthread_cond_t
    {
        __pthread_cond_s __data;
        char[48] __size;
        long __align;
    }
    union pthread_mutex_t
    {
        __pthread_mutex_s __data;
        char[40] __size;
        c_long __align;
    }
    union pthread_attr_t
    {
        char[56] __size;
        c_long __align;
    }
    alias pthread_once_t = int;
    alias pthread_key_t = uint;
    union pthread_condattr_t
    {
        char[4] __size;
        int __align;
    }
    union pthread_mutexattr_t
    {
        char[4] __size;
        int __align;
    }
    alias pthread_t = c_ulong;
    struct __pthread_rwlock_arch_t
    {
        uint __readers;
        uint __writers;
        uint __wrphase_futex;
        uint __writers_futex;
        uint __pad3;
        uint __pad4;
        int __cur_writer;
        int __shared;
        byte __rwelision;
        ubyte[7] __pad1;
        c_ulong __pad2;
        uint __flags;
    }
    struct div_t
    {
        int quot;
        int rem;
    }
    struct ldiv_t
    {
        c_long quot;
        c_long rem;
    }
    struct lldiv_t
    {
        long quot;
        long rem;
    }
    c_ulong __ctype_get_mb_cur_max() @nogc nothrow;
    double atof(const(char)*) @nogc nothrow;
    int atoi(const(char)*) @nogc nothrow;
    c_long atol(const(char)*) @nogc nothrow;
    long atoll(const(char)*) @nogc nothrow;
    double strtod(const(char)*, char**) @nogc nothrow;
    float strtof(const(char)*, char**) @nogc nothrow;
    real strtold(const(char)*, char**) @nogc nothrow;
    c_long strtol(const(char)*, char**, int) @nogc nothrow;
    c_ulong strtoul(const(char)*, char**, int) @nogc nothrow;
    long strtoq(const(char)*, char**, int) @nogc nothrow;
    ulong strtouq(const(char)*, char**, int) @nogc nothrow;
    long strtoll(const(char)*, char**, int) @nogc nothrow;
    ulong strtoull(const(char)*, char**, int) @nogc nothrow;
    char* l64a(c_long) @nogc nothrow;
    c_long a64l(const(char)*) @nogc nothrow;
    c_long random() @nogc nothrow;
    void srandom(uint) @nogc nothrow;
    char* initstate(uint, char*, c_ulong) @nogc nothrow;
    char* setstate(char*) @nogc nothrow;
    struct random_data
    {
        int* fptr;
        int* rptr;
        int* state;
        int rand_type;
        int rand_deg;
        int rand_sep;
        int* end_ptr;
    }
    int random_r(random_data*, int*) @nogc nothrow;
    int srandom_r(uint, random_data*) @nogc nothrow;
    int initstate_r(uint, char*, c_ulong, random_data*) @nogc nothrow;
    int setstate_r(char*, random_data*) @nogc nothrow;
    int rand() @nogc nothrow;
    void srand(uint) @nogc nothrow;
    int rand_r(uint*) @nogc nothrow;
    double drand48() @nogc nothrow;
    double erand48(ushort*) @nogc nothrow;
    c_long lrand48() @nogc nothrow;
    c_long nrand48(ushort*) @nogc nothrow;
    c_long mrand48() @nogc nothrow;
    c_long jrand48(ushort*) @nogc nothrow;
    void srand48(c_long) @nogc nothrow;
    ushort* seed48(ushort*) @nogc nothrow;
    void lcong48(ushort*) @nogc nothrow;
    struct drand48_data
    {
        ushort[3] __x;
        ushort[3] __old_x;
        ushort __c;
        ushort __init;
        ulong __a;
    }
    int drand48_r(drand48_data*, double*) @nogc nothrow;
    int erand48_r(ushort*, drand48_data*, double*) @nogc nothrow;
    int lrand48_r(drand48_data*, c_long*) @nogc nothrow;
    int nrand48_r(ushort*, drand48_data*, c_long*) @nogc nothrow;
    int mrand48_r(drand48_data*, c_long*) @nogc nothrow;
    int jrand48_r(ushort*, drand48_data*, c_long*) @nogc nothrow;
    int srand48_r(c_long, drand48_data*) @nogc nothrow;
    int seed48_r(ushort*, drand48_data*) @nogc nothrow;
    int lcong48_r(ushort*, drand48_data*) @nogc nothrow;
    void* malloc(c_ulong) @nogc nothrow;
    void* calloc(c_ulong, c_ulong) @nogc nothrow;
    void* realloc(void*, c_ulong) @nogc nothrow;
    void free(void*) @nogc nothrow;
    void* valloc(c_ulong) @nogc nothrow;
    int posix_memalign(void**, c_ulong, c_ulong) @nogc nothrow;
    void* aligned_alloc(c_ulong, c_ulong) @nogc nothrow;
    void abort() @nogc nothrow;
    int atexit(void function()) @nogc nothrow;
    int at_quick_exit(void function()) @nogc nothrow;
    int on_exit(void function(int, void*), void*) @nogc nothrow;
    void exit(int) @nogc nothrow;
    void quick_exit(int) @nogc nothrow;
    void _Exit(int) @nogc nothrow;
    char* getenv(const(char)*) @nogc nothrow;
    int putenv(char*) @nogc nothrow;
    int setenv(const(char)*, const(char)*, int) @nogc nothrow;
    int unsetenv(const(char)*) @nogc nothrow;
    int clearenv() @nogc nothrow;
    char* mktemp(char*) @nogc nothrow;
    int mkstemp(char*) @nogc nothrow;
    int mkstemps(char*, int) @nogc nothrow;
    char* mkdtemp(char*) @nogc nothrow;
    int system(const(char)*) @nogc nothrow;
    char* realpath(const(char)*, char*) @nogc nothrow;
    alias __compar_fn_t = int function(const(void)*, const(void)*);
    void* bsearch(const(void)*, const(void)*, c_ulong, c_ulong, int function(const(void)*, const(void)*)) @nogc nothrow;
    void qsort(void*, c_ulong, c_ulong, int function(const(void)*, const(void)*)) @nogc nothrow;
    int abs(int) @nogc nothrow;
    c_long labs(c_long) @nogc nothrow;
    long llabs(long) @nogc nothrow;
    div_t div(int, int) @nogc nothrow;
    ldiv_t ldiv(c_long, c_long) @nogc nothrow;
    lldiv_t lldiv(long, long) @nogc nothrow;
    char* ecvt(double, int, int*, int*) @nogc nothrow;
    char* fcvt(double, int, int*, int*) @nogc nothrow;
    char* gcvt(double, int, char*) @nogc nothrow;
    char* qecvt(real, int, int*, int*) @nogc nothrow;
    char* qfcvt(real, int, int*, int*) @nogc nothrow;
    char* qgcvt(real, int, char*) @nogc nothrow;
    int ecvt_r(double, int, int*, int*, char*, c_ulong) @nogc nothrow;
    int fcvt_r(double, int, int*, int*, char*, c_ulong) @nogc nothrow;
    int qecvt_r(real, int, int*, int*, char*, c_ulong) @nogc nothrow;
    int qfcvt_r(real, int, int*, int*, char*, c_ulong) @nogc nothrow;
    int mblen(const(char)*, c_ulong) @nogc nothrow;
    int mbtowc(int*, const(char)*, c_ulong) @nogc nothrow;
    int wctomb(char*, int) @nogc nothrow;
    c_ulong mbstowcs(int*, const(char)*, c_ulong) @nogc nothrow;
    c_ulong wcstombs(char*, const(int)*, c_ulong) @nogc nothrow;
    int rpmatch(const(char)*) @nogc nothrow;
    int getsubopt(char**, char**, char**) @nogc nothrow;
    int getloadavg(double*, int) @nogc nothrow;
    alias _Float64x = real;
    alias _Float32x = double;
    alias _Float64 = double;
    alias _Float32 = float;



    static if(!is(typeof(__CFLOAT32X))) {
        private enum enumMixinStr___CFLOAT32X = `enum __CFLOAT32X = _Complex double;`;
        static if(is(typeof({ mixin(enumMixinStr___CFLOAT32X); }))) {
            mixin(enumMixinStr___CFLOAT32X);
        }
    }




    static if(!is(typeof(__CFLOAT64))) {
        private enum enumMixinStr___CFLOAT64 = `enum __CFLOAT64 = _Complex double;`;
        static if(is(typeof({ mixin(enumMixinStr___CFLOAT64); }))) {
            mixin(enumMixinStr___CFLOAT64);
        }
    }




    static if(!is(typeof(__CFLOAT32))) {
        private enum enumMixinStr___CFLOAT32 = `enum __CFLOAT32 = _Complex float;`;
        static if(is(typeof({ mixin(enumMixinStr___CFLOAT32); }))) {
            mixin(enumMixinStr___CFLOAT32);
        }
    }




    static if(!is(typeof(__CFLOAT64X))) {
        private enum enumMixinStr___CFLOAT64X = `enum __CFLOAT64X = _Complex long double;`;
        static if(is(typeof({ mixin(enumMixinStr___CFLOAT64X); }))) {
            mixin(enumMixinStr___CFLOAT64X);
        }
    }
    static if(!is(typeof(__HAVE_FLOATN_NOT_TYPEDEF))) {
        private enum enumMixinStr___HAVE_FLOATN_NOT_TYPEDEF = `enum __HAVE_FLOATN_NOT_TYPEDEF = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_FLOATN_NOT_TYPEDEF); }))) {
            mixin(enumMixinStr___HAVE_FLOATN_NOT_TYPEDEF);
        }
    }




    static if(!is(typeof(__HAVE_DISTINCT_FLOAT128X))) {
        private enum enumMixinStr___HAVE_DISTINCT_FLOAT128X = `enum __HAVE_DISTINCT_FLOAT128X = __HAVE_FLOAT128X;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_DISTINCT_FLOAT128X); }))) {
            mixin(enumMixinStr___HAVE_DISTINCT_FLOAT128X);
        }
    }
    static if(!is(typeof(__HAVE_DISTINCT_FLOAT64X))) {
        private enum enumMixinStr___HAVE_DISTINCT_FLOAT64X = `enum __HAVE_DISTINCT_FLOAT64X = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_DISTINCT_FLOAT64X); }))) {
            mixin(enumMixinStr___HAVE_DISTINCT_FLOAT64X);
        }
    }




    static if(!is(typeof(__HAVE_DISTINCT_FLOAT32X))) {
        private enum enumMixinStr___HAVE_DISTINCT_FLOAT32X = `enum __HAVE_DISTINCT_FLOAT32X = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_DISTINCT_FLOAT32X); }))) {
            mixin(enumMixinStr___HAVE_DISTINCT_FLOAT32X);
        }
    }




    static if(!is(typeof(__HAVE_DISTINCT_FLOAT64))) {
        private enum enumMixinStr___HAVE_DISTINCT_FLOAT64 = `enum __HAVE_DISTINCT_FLOAT64 = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_DISTINCT_FLOAT64); }))) {
            mixin(enumMixinStr___HAVE_DISTINCT_FLOAT64);
        }
    }




    static if(!is(typeof(__HAVE_DISTINCT_FLOAT32))) {
        private enum enumMixinStr___HAVE_DISTINCT_FLOAT32 = `enum __HAVE_DISTINCT_FLOAT32 = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_DISTINCT_FLOAT32); }))) {
            mixin(enumMixinStr___HAVE_DISTINCT_FLOAT32);
        }
    }
    static if(!is(typeof(__HAVE_DISTINCT_FLOAT16))) {
        private enum enumMixinStr___HAVE_DISTINCT_FLOAT16 = `enum __HAVE_DISTINCT_FLOAT16 = __HAVE_FLOAT16;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_DISTINCT_FLOAT16); }))) {
            mixin(enumMixinStr___HAVE_DISTINCT_FLOAT16);
        }
    }




    static if(!is(typeof(__HAVE_FLOAT128X))) {
        private enum enumMixinStr___HAVE_FLOAT128X = `enum __HAVE_FLOAT128X = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_FLOAT128X); }))) {
            mixin(enumMixinStr___HAVE_FLOAT128X);
        }
    }




    static if(!is(typeof(__HAVE_FLOAT32X))) {
        private enum enumMixinStr___HAVE_FLOAT32X = `enum __HAVE_FLOAT32X = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_FLOAT32X); }))) {
            mixin(enumMixinStr___HAVE_FLOAT32X);
        }
    }




    static if(!is(typeof(__HAVE_FLOAT64))) {
        private enum enumMixinStr___HAVE_FLOAT64 = `enum __HAVE_FLOAT64 = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_FLOAT64); }))) {
            mixin(enumMixinStr___HAVE_FLOAT64);
        }
    }
    static if(!is(typeof(__HAVE_FLOAT32))) {
        private enum enumMixinStr___HAVE_FLOAT32 = `enum __HAVE_FLOAT32 = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_FLOAT32); }))) {
            mixin(enumMixinStr___HAVE_FLOAT32);
        }
    }




    static if(!is(typeof(__HAVE_FLOAT16))) {
        private enum enumMixinStr___HAVE_FLOAT16 = `enum __HAVE_FLOAT16 = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_FLOAT16); }))) {
            mixin(enumMixinStr___HAVE_FLOAT16);
        }
    }






    static if(!is(typeof(__BYTE_ORDER))) {
        private enum enumMixinStr___BYTE_ORDER = `enum __BYTE_ORDER = __LITTLE_ENDIAN;`;
        static if(is(typeof({ mixin(enumMixinStr___BYTE_ORDER); }))) {
            mixin(enumMixinStr___BYTE_ORDER);
        }
    }
    static if(!is(typeof(_BITS_BYTESWAP_H))) {
        private enum enumMixinStr__BITS_BYTESWAP_H = `enum _BITS_BYTESWAP_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__BITS_BYTESWAP_H); }))) {
            mixin(enumMixinStr__BITS_BYTESWAP_H);
        }
    }






    static if(!is(typeof(__HAVE_FLOAT128))) {
        private enum enumMixinStr___HAVE_FLOAT128 = `enum __HAVE_FLOAT128 = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_FLOAT128); }))) {
            mixin(enumMixinStr___HAVE_FLOAT128);
        }
    }






    static if(!is(typeof(__HAVE_DISTINCT_FLOAT128))) {
        private enum enumMixinStr___HAVE_DISTINCT_FLOAT128 = `enum __HAVE_DISTINCT_FLOAT128 = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_DISTINCT_FLOAT128); }))) {
            mixin(enumMixinStr___HAVE_DISTINCT_FLOAT128);
        }
    }




    static if(!is(typeof(__HAVE_FLOAT64X))) {
        private enum enumMixinStr___HAVE_FLOAT64X = `enum __HAVE_FLOAT64X = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_FLOAT64X); }))) {
            mixin(enumMixinStr___HAVE_FLOAT64X);
        }
    }




    static if(!is(typeof(__HAVE_FLOAT64X_LONG_DOUBLE))) {
        private enum enumMixinStr___HAVE_FLOAT64X_LONG_DOUBLE = `enum __HAVE_FLOAT64X_LONG_DOUBLE = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_FLOAT64X_LONG_DOUBLE); }))) {
            mixin(enumMixinStr___HAVE_FLOAT64X_LONG_DOUBLE);
        }
    }




    static if(!is(typeof(MB_CUR_MAX))) {
        private enum enumMixinStr_MB_CUR_MAX = `enum MB_CUR_MAX = ( __ctype_get_mb_cur_max ( ) );`;
        static if(is(typeof({ mixin(enumMixinStr_MB_CUR_MAX); }))) {
            mixin(enumMixinStr_MB_CUR_MAX);
        }
    }




    static if(!is(typeof(EXIT_SUCCESS))) {
        private enum enumMixinStr_EXIT_SUCCESS = `enum EXIT_SUCCESS = 0;`;
        static if(is(typeof({ mixin(enumMixinStr_EXIT_SUCCESS); }))) {
            mixin(enumMixinStr_EXIT_SUCCESS);
        }
    }




    static if(!is(typeof(EXIT_FAILURE))) {
        private enum enumMixinStr_EXIT_FAILURE = `enum EXIT_FAILURE = 1;`;
        static if(is(typeof({ mixin(enumMixinStr_EXIT_FAILURE); }))) {
            mixin(enumMixinStr_EXIT_FAILURE);
        }
    }




    static if(!is(typeof(RAND_MAX))) {
        private enum enumMixinStr_RAND_MAX = `enum RAND_MAX = 2147483647;`;
        static if(is(typeof({ mixin(enumMixinStr_RAND_MAX); }))) {
            mixin(enumMixinStr_RAND_MAX);
        }
    }




    static if(!is(typeof(__lldiv_t_defined))) {
        private enum enumMixinStr___lldiv_t_defined = `enum __lldiv_t_defined = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___lldiv_t_defined); }))) {
            mixin(enumMixinStr___lldiv_t_defined);
        }
    }




    static if(!is(typeof(__ldiv_t_defined))) {
        private enum enumMixinStr___ldiv_t_defined = `enum __ldiv_t_defined = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___ldiv_t_defined); }))) {
            mixin(enumMixinStr___ldiv_t_defined);
        }
    }




    static if(!is(typeof(__GLIBC_USE_LIB_EXT2))) {
        private enum enumMixinStr___GLIBC_USE_LIB_EXT2 = `enum __GLIBC_USE_LIB_EXT2 = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___GLIBC_USE_LIB_EXT2); }))) {
            mixin(enumMixinStr___GLIBC_USE_LIB_EXT2);
        }
    }




    static if(!is(typeof(__GLIBC_USE_IEC_60559_BFP_EXT))) {
        private enum enumMixinStr___GLIBC_USE_IEC_60559_BFP_EXT = `enum __GLIBC_USE_IEC_60559_BFP_EXT = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___GLIBC_USE_IEC_60559_BFP_EXT); }))) {
            mixin(enumMixinStr___GLIBC_USE_IEC_60559_BFP_EXT);
        }
    }




    static if(!is(typeof(__GLIBC_USE_IEC_60559_FUNCS_EXT))) {
        private enum enumMixinStr___GLIBC_USE_IEC_60559_FUNCS_EXT = `enum __GLIBC_USE_IEC_60559_FUNCS_EXT = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___GLIBC_USE_IEC_60559_FUNCS_EXT); }))) {
            mixin(enumMixinStr___GLIBC_USE_IEC_60559_FUNCS_EXT);
        }
    }




    static if(!is(typeof(__GLIBC_USE_IEC_60559_TYPES_EXT))) {
        private enum enumMixinStr___GLIBC_USE_IEC_60559_TYPES_EXT = `enum __GLIBC_USE_IEC_60559_TYPES_EXT = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___GLIBC_USE_IEC_60559_TYPES_EXT); }))) {
            mixin(enumMixinStr___GLIBC_USE_IEC_60559_TYPES_EXT);
        }
    }




    static if(!is(typeof(_BITS_PTHREADTYPES_ARCH_H))) {
        private enum enumMixinStr__BITS_PTHREADTYPES_ARCH_H = `enum _BITS_PTHREADTYPES_ARCH_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__BITS_PTHREADTYPES_ARCH_H); }))) {
            mixin(enumMixinStr__BITS_PTHREADTYPES_ARCH_H);
        }
    }
    static if(!is(typeof(__SIZEOF_PTHREAD_MUTEX_T))) {
        private enum enumMixinStr___SIZEOF_PTHREAD_MUTEX_T = `enum __SIZEOF_PTHREAD_MUTEX_T = 40;`;
        static if(is(typeof({ mixin(enumMixinStr___SIZEOF_PTHREAD_MUTEX_T); }))) {
            mixin(enumMixinStr___SIZEOF_PTHREAD_MUTEX_T);
        }
    }




    static if(!is(typeof(__SIZEOF_PTHREAD_ATTR_T))) {
        private enum enumMixinStr___SIZEOF_PTHREAD_ATTR_T = `enum __SIZEOF_PTHREAD_ATTR_T = 56;`;
        static if(is(typeof({ mixin(enumMixinStr___SIZEOF_PTHREAD_ATTR_T); }))) {
            mixin(enumMixinStr___SIZEOF_PTHREAD_ATTR_T);
        }
    }




    static if(!is(typeof(__SIZEOF_PTHREAD_RWLOCK_T))) {
        private enum enumMixinStr___SIZEOF_PTHREAD_RWLOCK_T = `enum __SIZEOF_PTHREAD_RWLOCK_T = 56;`;
        static if(is(typeof({ mixin(enumMixinStr___SIZEOF_PTHREAD_RWLOCK_T); }))) {
            mixin(enumMixinStr___SIZEOF_PTHREAD_RWLOCK_T);
        }
    }




    static if(!is(typeof(__SIZEOF_PTHREAD_BARRIER_T))) {
        private enum enumMixinStr___SIZEOF_PTHREAD_BARRIER_T = `enum __SIZEOF_PTHREAD_BARRIER_T = 32;`;
        static if(is(typeof({ mixin(enumMixinStr___SIZEOF_PTHREAD_BARRIER_T); }))) {
            mixin(enumMixinStr___SIZEOF_PTHREAD_BARRIER_T);
        }
    }




    static if(!is(typeof(__SIZEOF_PTHREAD_MUTEXATTR_T))) {
        private enum enumMixinStr___SIZEOF_PTHREAD_MUTEXATTR_T = `enum __SIZEOF_PTHREAD_MUTEXATTR_T = 4;`;
        static if(is(typeof({ mixin(enumMixinStr___SIZEOF_PTHREAD_MUTEXATTR_T); }))) {
            mixin(enumMixinStr___SIZEOF_PTHREAD_MUTEXATTR_T);
        }
    }




    static if(!is(typeof(__SIZEOF_PTHREAD_COND_T))) {
        private enum enumMixinStr___SIZEOF_PTHREAD_COND_T = `enum __SIZEOF_PTHREAD_COND_T = 48;`;
        static if(is(typeof({ mixin(enumMixinStr___SIZEOF_PTHREAD_COND_T); }))) {
            mixin(enumMixinStr___SIZEOF_PTHREAD_COND_T);
        }
    }




    static if(!is(typeof(__SIZEOF_PTHREAD_CONDATTR_T))) {
        private enum enumMixinStr___SIZEOF_PTHREAD_CONDATTR_T = `enum __SIZEOF_PTHREAD_CONDATTR_T = 4;`;
        static if(is(typeof({ mixin(enumMixinStr___SIZEOF_PTHREAD_CONDATTR_T); }))) {
            mixin(enumMixinStr___SIZEOF_PTHREAD_CONDATTR_T);
        }
    }




    static if(!is(typeof(__SIZEOF_PTHREAD_RWLOCKATTR_T))) {
        private enum enumMixinStr___SIZEOF_PTHREAD_RWLOCKATTR_T = `enum __SIZEOF_PTHREAD_RWLOCKATTR_T = 8;`;
        static if(is(typeof({ mixin(enumMixinStr___SIZEOF_PTHREAD_RWLOCKATTR_T); }))) {
            mixin(enumMixinStr___SIZEOF_PTHREAD_RWLOCKATTR_T);
        }
    }




    static if(!is(typeof(__SIZEOF_PTHREAD_BARRIERATTR_T))) {
        private enum enumMixinStr___SIZEOF_PTHREAD_BARRIERATTR_T = `enum __SIZEOF_PTHREAD_BARRIERATTR_T = 4;`;
        static if(is(typeof({ mixin(enumMixinStr___SIZEOF_PTHREAD_BARRIERATTR_T); }))) {
            mixin(enumMixinStr___SIZEOF_PTHREAD_BARRIERATTR_T);
        }
    }
    static if(!is(typeof(__PTHREAD_MUTEX_LOCK_ELISION))) {
        private enum enumMixinStr___PTHREAD_MUTEX_LOCK_ELISION = `enum __PTHREAD_MUTEX_LOCK_ELISION = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___PTHREAD_MUTEX_LOCK_ELISION); }))) {
            mixin(enumMixinStr___PTHREAD_MUTEX_LOCK_ELISION);
        }
    }






    static if(!is(typeof(__PTHREAD_MUTEX_NUSERS_AFTER_KIND))) {
        private enum enumMixinStr___PTHREAD_MUTEX_NUSERS_AFTER_KIND = `enum __PTHREAD_MUTEX_NUSERS_AFTER_KIND = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___PTHREAD_MUTEX_NUSERS_AFTER_KIND); }))) {
            mixin(enumMixinStr___PTHREAD_MUTEX_NUSERS_AFTER_KIND);
        }
    }




    static if(!is(typeof(__PTHREAD_MUTEX_USE_UNION))) {
        private enum enumMixinStr___PTHREAD_MUTEX_USE_UNION = `enum __PTHREAD_MUTEX_USE_UNION = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___PTHREAD_MUTEX_USE_UNION); }))) {
            mixin(enumMixinStr___PTHREAD_MUTEX_USE_UNION);
        }
    }
    static if(!is(typeof(__PTHREAD_RWLOCK_ELISION_EXTRA))) {
        private enum enumMixinStr___PTHREAD_RWLOCK_ELISION_EXTRA = `enum __PTHREAD_RWLOCK_ELISION_EXTRA = 0 , { 0 , 0 , 0 , 0 , 0 , 0 , 0 };`;
        static if(is(typeof({ mixin(enumMixinStr___PTHREAD_RWLOCK_ELISION_EXTRA); }))) {
            mixin(enumMixinStr___PTHREAD_RWLOCK_ELISION_EXTRA);
        }
    }




    static if(!is(typeof(__PTHREAD_RWLOCK_INT_FLAGS_SHARED))) {
        private enum enumMixinStr___PTHREAD_RWLOCK_INT_FLAGS_SHARED = `enum __PTHREAD_RWLOCK_INT_FLAGS_SHARED = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___PTHREAD_RWLOCK_INT_FLAGS_SHARED); }))) {
            mixin(enumMixinStr___PTHREAD_RWLOCK_INT_FLAGS_SHARED);
        }
    }






    static if(!is(typeof(_BITS_PTHREADTYPES_COMMON_H))) {
        private enum enumMixinStr__BITS_PTHREADTYPES_COMMON_H = `enum _BITS_PTHREADTYPES_COMMON_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__BITS_PTHREADTYPES_COMMON_H); }))) {
            mixin(enumMixinStr__BITS_PTHREADTYPES_COMMON_H);
        }
    }




    static if(!is(typeof(_STDLIB_H))) {
        private enum enumMixinStr__STDLIB_H = `enum _STDLIB_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__STDLIB_H); }))) {
            mixin(enumMixinStr__STDLIB_H);
        }
    }
    static if(!is(typeof(_STDC_PREDEF_H))) {
        private enum enumMixinStr__STDC_PREDEF_H = `enum _STDC_PREDEF_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__STDC_PREDEF_H); }))) {
            mixin(enumMixinStr__STDC_PREDEF_H);
        }
    }






    static if(!is(typeof(__GLIBC_MINOR__))) {
        private enum enumMixinStr___GLIBC_MINOR__ = `enum __GLIBC_MINOR__ = 27;`;
        static if(is(typeof({ mixin(enumMixinStr___GLIBC_MINOR__); }))) {
            mixin(enumMixinStr___GLIBC_MINOR__);
        }
    }




    static if(!is(typeof(__GLIBC__))) {
        private enum enumMixinStr___GLIBC__ = `enum __GLIBC__ = 2;`;
        static if(is(typeof({ mixin(enumMixinStr___GLIBC__); }))) {
            mixin(enumMixinStr___GLIBC__);
        }
    }




    static if(!is(typeof(__GNU_LIBRARY__))) {
        private enum enumMixinStr___GNU_LIBRARY__ = `enum __GNU_LIBRARY__ = 6;`;
        static if(is(typeof({ mixin(enumMixinStr___GNU_LIBRARY__); }))) {
            mixin(enumMixinStr___GNU_LIBRARY__);
        }
    }




    static if(!is(typeof(__have_pthread_attr_t))) {
        private enum enumMixinStr___have_pthread_attr_t = `enum __have_pthread_attr_t = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___have_pthread_attr_t); }))) {
            mixin(enumMixinStr___have_pthread_attr_t);
        }
    }




    static if(!is(typeof(__GLIBC_USE_DEPRECATED_GETS))) {
        private enum enumMixinStr___GLIBC_USE_DEPRECATED_GETS = `enum __GLIBC_USE_DEPRECATED_GETS = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___GLIBC_USE_DEPRECATED_GETS); }))) {
            mixin(enumMixinStr___GLIBC_USE_DEPRECATED_GETS);
        }
    }




    static if(!is(typeof(__USE_FORTIFY_LEVEL))) {
        private enum enumMixinStr___USE_FORTIFY_LEVEL = `enum __USE_FORTIFY_LEVEL = 0;`;
        static if(is(typeof({ mixin(enumMixinStr___USE_FORTIFY_LEVEL); }))) {
            mixin(enumMixinStr___USE_FORTIFY_LEVEL);
        }
    }




    static if(!is(typeof(__USE_ATFILE))) {
        private enum enumMixinStr___USE_ATFILE = `enum __USE_ATFILE = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___USE_ATFILE); }))) {
            mixin(enumMixinStr___USE_ATFILE);
        }
    }




    static if(!is(typeof(__USE_MISC))) {
        private enum enumMixinStr___USE_MISC = `enum __USE_MISC = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___USE_MISC); }))) {
            mixin(enumMixinStr___USE_MISC);
        }
    }




    static if(!is(typeof(_ATFILE_SOURCE))) {
        private enum enumMixinStr__ATFILE_SOURCE = `enum _ATFILE_SOURCE = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__ATFILE_SOURCE); }))) {
            mixin(enumMixinStr__ATFILE_SOURCE);
        }
    }




    static if(!is(typeof(__USE_XOPEN2K8))) {
        private enum enumMixinStr___USE_XOPEN2K8 = `enum __USE_XOPEN2K8 = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___USE_XOPEN2K8); }))) {
            mixin(enumMixinStr___USE_XOPEN2K8);
        }
    }




    static if(!is(typeof(__USE_ISOC99))) {
        private enum enumMixinStr___USE_ISOC99 = `enum __USE_ISOC99 = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___USE_ISOC99); }))) {
            mixin(enumMixinStr___USE_ISOC99);
        }
    }




    static if(!is(typeof(__USE_ISOC95))) {
        private enum enumMixinStr___USE_ISOC95 = `enum __USE_ISOC95 = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___USE_ISOC95); }))) {
            mixin(enumMixinStr___USE_ISOC95);
        }
    }




    static if(!is(typeof(__USE_XOPEN2K))) {
        private enum enumMixinStr___USE_XOPEN2K = `enum __USE_XOPEN2K = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___USE_XOPEN2K); }))) {
            mixin(enumMixinStr___USE_XOPEN2K);
        }
    }




    static if(!is(typeof(__USE_POSIX199506))) {
        private enum enumMixinStr___USE_POSIX199506 = `enum __USE_POSIX199506 = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___USE_POSIX199506); }))) {
            mixin(enumMixinStr___USE_POSIX199506);
        }
    }




    static if(!is(typeof(__USE_POSIX199309))) {
        private enum enumMixinStr___USE_POSIX199309 = `enum __USE_POSIX199309 = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___USE_POSIX199309); }))) {
            mixin(enumMixinStr___USE_POSIX199309);
        }
    }




    static if(!is(typeof(__USE_POSIX2))) {
        private enum enumMixinStr___USE_POSIX2 = `enum __USE_POSIX2 = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___USE_POSIX2); }))) {
            mixin(enumMixinStr___USE_POSIX2);
        }
    }




    static if(!is(typeof(__USE_POSIX))) {
        private enum enumMixinStr___USE_POSIX = `enum __USE_POSIX = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___USE_POSIX); }))) {
            mixin(enumMixinStr___USE_POSIX);
        }
    }




    static if(!is(typeof(_POSIX_C_SOURCE))) {
        private enum enumMixinStr__POSIX_C_SOURCE = `enum _POSIX_C_SOURCE = 200809L;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_C_SOURCE); }))) {
            mixin(enumMixinStr__POSIX_C_SOURCE);
        }
    }




    static if(!is(typeof(_POSIX_SOURCE))) {
        private enum enumMixinStr__POSIX_SOURCE = `enum _POSIX_SOURCE = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__POSIX_SOURCE); }))) {
            mixin(enumMixinStr__POSIX_SOURCE);
        }
    }




    static if(!is(typeof(__USE_POSIX_IMPLICITLY))) {
        private enum enumMixinStr___USE_POSIX_IMPLICITLY = `enum __USE_POSIX_IMPLICITLY = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___USE_POSIX_IMPLICITLY); }))) {
            mixin(enumMixinStr___USE_POSIX_IMPLICITLY);
        }
    }




    static if(!is(typeof(__USE_ISOC11))) {
        private enum enumMixinStr___USE_ISOC11 = `enum __USE_ISOC11 = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___USE_ISOC11); }))) {
            mixin(enumMixinStr___USE_ISOC11);
        }
    }




    static if(!is(typeof(_DEFAULT_SOURCE))) {
        private enum enumMixinStr__DEFAULT_SOURCE = `enum _DEFAULT_SOURCE = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__DEFAULT_SOURCE); }))) {
            mixin(enumMixinStr__DEFAULT_SOURCE);
        }
    }
    static if(!is(typeof(_FEATURES_H))) {
        private enum enumMixinStr__FEATURES_H = `enum _FEATURES_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__FEATURES_H); }))) {
            mixin(enumMixinStr__FEATURES_H);
        }
    }






    static if(!is(typeof(__FD_ZERO_STOS))) {
        private enum enumMixinStr___FD_ZERO_STOS = `enum __FD_ZERO_STOS = "stosq";`;
        static if(is(typeof({ mixin(enumMixinStr___FD_ZERO_STOS); }))) {
            mixin(enumMixinStr___FD_ZERO_STOS);
        }
    }
    static if(!is(typeof(_BITS_STDINT_INTN_H))) {
        private enum enumMixinStr__BITS_STDINT_INTN_H = `enum _BITS_STDINT_INTN_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__BITS_STDINT_INTN_H); }))) {
            mixin(enumMixinStr__BITS_STDINT_INTN_H);
        }
    }
    static if(!is(typeof(_BITS_SYSMACROS_H))) {
        private enum enumMixinStr__BITS_SYSMACROS_H = `enum _BITS_SYSMACROS_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__BITS_SYSMACROS_H); }))) {
            mixin(enumMixinStr__BITS_SYSMACROS_H);
        }
    }






    static if(!is(typeof(__SYSMACROS_DECLARE_MAJOR))) {
        private enum enumMixinStr___SYSMACROS_DECLARE_MAJOR = `enum __SYSMACROS_DECLARE_MAJOR = ( DECL_TEMPL ) DECL_TEMPL ( unsigned int , major , ( __dev_t __dev ) );`;
        static if(is(typeof({ mixin(enumMixinStr___SYSMACROS_DECLARE_MAJOR); }))) {
            mixin(enumMixinStr___SYSMACROS_DECLARE_MAJOR);
        }
    }




    static if(!is(typeof(__SYSMACROS_DEFINE_MAJOR))) {
        private enum enumMixinStr___SYSMACROS_DEFINE_MAJOR = `enum __SYSMACROS_DEFINE_MAJOR = ( DECL_TEMPL ) ( DECL_TEMPL ) DECL_TEMPL ( unsigned int , major , ( __dev_t __dev ) ) ( DECL_TEMPL ) { unsigned int __major ; __major = ( ( __dev & cast( __dev_t ) 0x00000000000fff00u ) >> 8 ) ; __major |= ( ( __dev & cast( __dev_t ) 0xfffff00000000000u ) >> 32 ) ; return __major ; };`;
        static if(is(typeof({ mixin(enumMixinStr___SYSMACROS_DEFINE_MAJOR); }))) {
            mixin(enumMixinStr___SYSMACROS_DEFINE_MAJOR);
        }
    }




    static if(!is(typeof(__SYSMACROS_DECLARE_MINOR))) {
        private enum enumMixinStr___SYSMACROS_DECLARE_MINOR = `enum __SYSMACROS_DECLARE_MINOR = ( DECL_TEMPL ) DECL_TEMPL ( unsigned int , minor , ( __dev_t __dev ) );`;
        static if(is(typeof({ mixin(enumMixinStr___SYSMACROS_DECLARE_MINOR); }))) {
            mixin(enumMixinStr___SYSMACROS_DECLARE_MINOR);
        }
    }




    static if(!is(typeof(__SYSMACROS_DEFINE_MINOR))) {
        private enum enumMixinStr___SYSMACROS_DEFINE_MINOR = `enum __SYSMACROS_DEFINE_MINOR = ( DECL_TEMPL ) ( DECL_TEMPL ) DECL_TEMPL ( unsigned int , minor , ( __dev_t __dev ) ) ( DECL_TEMPL ) { unsigned int __minor ; __minor = ( ( __dev & cast( __dev_t ) 0x00000000000000ffu ) >> 0 ) ; __minor |= ( ( __dev & cast( __dev_t ) 0x00000ffffff00000u ) >> 12 ) ; return __minor ; };`;
        static if(is(typeof({ mixin(enumMixinStr___SYSMACROS_DEFINE_MINOR); }))) {
            mixin(enumMixinStr___SYSMACROS_DEFINE_MINOR);
        }
    }




    static if(!is(typeof(__SYSMACROS_DECLARE_MAKEDEV))) {
        private enum enumMixinStr___SYSMACROS_DECLARE_MAKEDEV = `enum __SYSMACROS_DECLARE_MAKEDEV = ( DECL_TEMPL ) DECL_TEMPL ( __dev_t , makedev , ( unsigned int __major , unsigned int __minor ) );`;
        static if(is(typeof({ mixin(enumMixinStr___SYSMACROS_DECLARE_MAKEDEV); }))) {
            mixin(enumMixinStr___SYSMACROS_DECLARE_MAKEDEV);
        }
    }




    static if(!is(typeof(__SYSMACROS_DEFINE_MAKEDEV))) {
        private enum enumMixinStr___SYSMACROS_DEFINE_MAKEDEV = `enum __SYSMACROS_DEFINE_MAKEDEV = ( DECL_TEMPL ) ( DECL_TEMPL ) DECL_TEMPL ( __dev_t , makedev , ( unsigned int __major , unsigned int __minor ) ) ( DECL_TEMPL ) { __dev_t __dev ; __dev = ( ( cast( __dev_t ) ( __major & 0x00000fffu ) ) << 8 ) ; __dev |= ( ( cast( __dev_t ) ( __major & 0xfffff000u ) ) << 32 ) ; __dev |= ( ( cast( __dev_t ) ( __minor & 0x000000ffu ) ) << 0 ) ; __dev |= ( ( cast( __dev_t ) ( __minor & 0xffffff00u ) ) << 12 ) ; return __dev ; };`;
        static if(is(typeof({ mixin(enumMixinStr___SYSMACROS_DEFINE_MAKEDEV); }))) {
            mixin(enumMixinStr___SYSMACROS_DEFINE_MAKEDEV);
        }
    }




    static if(!is(typeof(_THREAD_SHARED_TYPES_H))) {
        private enum enumMixinStr__THREAD_SHARED_TYPES_H = `enum _THREAD_SHARED_TYPES_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__THREAD_SHARED_TYPES_H); }))) {
            mixin(enumMixinStr__THREAD_SHARED_TYPES_H);
        }
    }
    static if(!is(typeof(BYTE_ORDER))) {
        private enum enumMixinStr_BYTE_ORDER = `enum BYTE_ORDER = __LITTLE_ENDIAN;`;
        static if(is(typeof({ mixin(enumMixinStr_BYTE_ORDER); }))) {
            mixin(enumMixinStr_BYTE_ORDER);
        }
    }




    static if(!is(typeof(__PTHREAD_SPINS_DATA))) {
        private enum enumMixinStr___PTHREAD_SPINS_DATA = `enum __PTHREAD_SPINS_DATA = short __spins ; short __elision;`;
        static if(is(typeof({ mixin(enumMixinStr___PTHREAD_SPINS_DATA); }))) {
            mixin(enumMixinStr___PTHREAD_SPINS_DATA);
        }
    }




    static if(!is(typeof(__PTHREAD_SPINS))) {
        private enum enumMixinStr___PTHREAD_SPINS = `enum __PTHREAD_SPINS = 0 , 0;`;
        static if(is(typeof({ mixin(enumMixinStr___PTHREAD_SPINS); }))) {
            mixin(enumMixinStr___PTHREAD_SPINS);
        }
    }




    static if(!is(typeof(PDP_ENDIAN))) {
        private enum enumMixinStr_PDP_ENDIAN = `enum PDP_ENDIAN = __PDP_ENDIAN;`;
        static if(is(typeof({ mixin(enumMixinStr_PDP_ENDIAN); }))) {
            mixin(enumMixinStr_PDP_ENDIAN);
        }
    }




    static if(!is(typeof(BIG_ENDIAN))) {
        private enum enumMixinStr_BIG_ENDIAN = `enum BIG_ENDIAN = __BIG_ENDIAN;`;
        static if(is(typeof({ mixin(enumMixinStr_BIG_ENDIAN); }))) {
            mixin(enumMixinStr_BIG_ENDIAN);
        }
    }




    static if(!is(typeof(LITTLE_ENDIAN))) {
        private enum enumMixinStr_LITTLE_ENDIAN = `enum LITTLE_ENDIAN = __LITTLE_ENDIAN;`;
        static if(is(typeof({ mixin(enumMixinStr_LITTLE_ENDIAN); }))) {
            mixin(enumMixinStr_LITTLE_ENDIAN);
        }
    }




    static if(!is(typeof(__FLOAT_WORD_ORDER))) {
        private enum enumMixinStr___FLOAT_WORD_ORDER = `enum __FLOAT_WORD_ORDER = __LITTLE_ENDIAN;`;
        static if(is(typeof({ mixin(enumMixinStr___FLOAT_WORD_ORDER); }))) {
            mixin(enumMixinStr___FLOAT_WORD_ORDER);
        }
    }




    static if(!is(typeof(__PDP_ENDIAN))) {
        private enum enumMixinStr___PDP_ENDIAN = `enum __PDP_ENDIAN = 3412;`;
        static if(is(typeof({ mixin(enumMixinStr___PDP_ENDIAN); }))) {
            mixin(enumMixinStr___PDP_ENDIAN);
        }
    }




    static if(!is(typeof(__BIG_ENDIAN))) {
        private enum enumMixinStr___BIG_ENDIAN = `enum __BIG_ENDIAN = 4321;`;
        static if(is(typeof({ mixin(enumMixinStr___BIG_ENDIAN); }))) {
            mixin(enumMixinStr___BIG_ENDIAN);
        }
    }




    static if(!is(typeof(__LITTLE_ENDIAN))) {
        private enum enumMixinStr___LITTLE_ENDIAN = `enum __LITTLE_ENDIAN = 1234;`;
        static if(is(typeof({ mixin(enumMixinStr___LITTLE_ENDIAN); }))) {
            mixin(enumMixinStr___LITTLE_ENDIAN);
        }
    }




    static if(!is(typeof(__PTHREAD_MUTEX_HAVE_PREV))) {
        private enum enumMixinStr___PTHREAD_MUTEX_HAVE_PREV = `enum __PTHREAD_MUTEX_HAVE_PREV = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___PTHREAD_MUTEX_HAVE_PREV); }))) {
            mixin(enumMixinStr___PTHREAD_MUTEX_HAVE_PREV);
        }
    }




    static if(!is(typeof(_ENDIAN_H))) {
        private enum enumMixinStr__ENDIAN_H = `enum _ENDIAN_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__ENDIAN_H); }))) {
            mixin(enumMixinStr__ENDIAN_H);
        }
    }




    static if(!is(typeof(NULL))) {
        private enum enumMixinStr_NULL = `enum NULL = ( cast( void * ) 0 );`;
        static if(is(typeof({ mixin(enumMixinStr_NULL); }))) {
            mixin(enumMixinStr_NULL);
        }
    }






    static if(!is(typeof(_BITS_TYPES_H))) {
        private enum enumMixinStr__BITS_TYPES_H = `enum _BITS_TYPES_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__BITS_TYPES_H); }))) {
            mixin(enumMixinStr__BITS_TYPES_H);
        }
    }
    static if(!is(typeof(_ALLOCA_H))) {
        private enum enumMixinStr__ALLOCA_H = `enum _ALLOCA_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__ALLOCA_H); }))) {
            mixin(enumMixinStr__ALLOCA_H);
        }
    }




    static if(!is(typeof(__S16_TYPE))) {
        private enum enumMixinStr___S16_TYPE = `enum __S16_TYPE = short int;`;
        static if(is(typeof({ mixin(enumMixinStr___S16_TYPE); }))) {
            mixin(enumMixinStr___S16_TYPE);
        }
    }




    static if(!is(typeof(__U16_TYPE))) {
        private enum enumMixinStr___U16_TYPE = `enum __U16_TYPE = unsigned short int;`;
        static if(is(typeof({ mixin(enumMixinStr___U16_TYPE); }))) {
            mixin(enumMixinStr___U16_TYPE);
        }
    }




    static if(!is(typeof(__S32_TYPE))) {
        private enum enumMixinStr___S32_TYPE = `enum __S32_TYPE = int;`;
        static if(is(typeof({ mixin(enumMixinStr___S32_TYPE); }))) {
            mixin(enumMixinStr___S32_TYPE);
        }
    }




    static if(!is(typeof(__U32_TYPE))) {
        private enum enumMixinStr___U32_TYPE = `enum __U32_TYPE = unsigned int;`;
        static if(is(typeof({ mixin(enumMixinStr___U32_TYPE); }))) {
            mixin(enumMixinStr___U32_TYPE);
        }
    }




    static if(!is(typeof(__SLONGWORD_TYPE))) {
        private enum enumMixinStr___SLONGWORD_TYPE = `enum __SLONGWORD_TYPE = long int;`;
        static if(is(typeof({ mixin(enumMixinStr___SLONGWORD_TYPE); }))) {
            mixin(enumMixinStr___SLONGWORD_TYPE);
        }
    }




    static if(!is(typeof(__ULONGWORD_TYPE))) {
        private enum enumMixinStr___ULONGWORD_TYPE = `enum __ULONGWORD_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___ULONGWORD_TYPE); }))) {
            mixin(enumMixinStr___ULONGWORD_TYPE);
        }
    }




    static if(!is(typeof(__SQUAD_TYPE))) {
        private enum enumMixinStr___SQUAD_TYPE = `enum __SQUAD_TYPE = long int;`;
        static if(is(typeof({ mixin(enumMixinStr___SQUAD_TYPE); }))) {
            mixin(enumMixinStr___SQUAD_TYPE);
        }
    }




    static if(!is(typeof(__UQUAD_TYPE))) {
        private enum enumMixinStr___UQUAD_TYPE = `enum __UQUAD_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___UQUAD_TYPE); }))) {
            mixin(enumMixinStr___UQUAD_TYPE);
        }
    }




    static if(!is(typeof(__SWORD_TYPE))) {
        private enum enumMixinStr___SWORD_TYPE = `enum __SWORD_TYPE = long int;`;
        static if(is(typeof({ mixin(enumMixinStr___SWORD_TYPE); }))) {
            mixin(enumMixinStr___SWORD_TYPE);
        }
    }




    static if(!is(typeof(__UWORD_TYPE))) {
        private enum enumMixinStr___UWORD_TYPE = `enum __UWORD_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___UWORD_TYPE); }))) {
            mixin(enumMixinStr___UWORD_TYPE);
        }
    }




    static if(!is(typeof(__SLONG32_TYPE))) {
        private enum enumMixinStr___SLONG32_TYPE = `enum __SLONG32_TYPE = int;`;
        static if(is(typeof({ mixin(enumMixinStr___SLONG32_TYPE); }))) {
            mixin(enumMixinStr___SLONG32_TYPE);
        }
    }




    static if(!is(typeof(__ULONG32_TYPE))) {
        private enum enumMixinStr___ULONG32_TYPE = `enum __ULONG32_TYPE = unsigned int;`;
        static if(is(typeof({ mixin(enumMixinStr___ULONG32_TYPE); }))) {
            mixin(enumMixinStr___ULONG32_TYPE);
        }
    }




    static if(!is(typeof(__S64_TYPE))) {
        private enum enumMixinStr___S64_TYPE = `enum __S64_TYPE = long int;`;
        static if(is(typeof({ mixin(enumMixinStr___S64_TYPE); }))) {
            mixin(enumMixinStr___S64_TYPE);
        }
    }




    static if(!is(typeof(__U64_TYPE))) {
        private enum enumMixinStr___U64_TYPE = `enum __U64_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___U64_TYPE); }))) {
            mixin(enumMixinStr___U64_TYPE);
        }
    }




    static if(!is(typeof(__STD_TYPE))) {
        private enum enumMixinStr___STD_TYPE = `enum __STD_TYPE = typedef;`;
        static if(is(typeof({ mixin(enumMixinStr___STD_TYPE); }))) {
            mixin(enumMixinStr___STD_TYPE);
        }
    }






    static if(!is(typeof(_SIGSET_NWORDS))) {
        private enum enumMixinStr__SIGSET_NWORDS = `enum _SIGSET_NWORDS = ( 1024 / ( 8 * ( unsigned long int ) .sizeof ) );`;
        static if(is(typeof({ mixin(enumMixinStr__SIGSET_NWORDS); }))) {
            mixin(enumMixinStr__SIGSET_NWORDS);
        }
    }




    static if(!is(typeof(__clock_t_defined))) {
        private enum enumMixinStr___clock_t_defined = `enum __clock_t_defined = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___clock_t_defined); }))) {
            mixin(enumMixinStr___clock_t_defined);
        }
    }




    static if(!is(typeof(__clockid_t_defined))) {
        private enum enumMixinStr___clockid_t_defined = `enum __clockid_t_defined = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___clockid_t_defined); }))) {
            mixin(enumMixinStr___clockid_t_defined);
        }
    }




    static if(!is(typeof(__sigset_t_defined))) {
        private enum enumMixinStr___sigset_t_defined = `enum __sigset_t_defined = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___sigset_t_defined); }))) {
            mixin(enumMixinStr___sigset_t_defined);
        }
    }




    static if(!is(typeof(__timespec_defined))) {
        private enum enumMixinStr___timespec_defined = `enum __timespec_defined = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___timespec_defined); }))) {
            mixin(enumMixinStr___timespec_defined);
        }
    }




    static if(!is(typeof(__timeval_defined))) {
        private enum enumMixinStr___timeval_defined = `enum __timeval_defined = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___timeval_defined); }))) {
            mixin(enumMixinStr___timeval_defined);
        }
    }




    static if(!is(typeof(__time_t_defined))) {
        private enum enumMixinStr___time_t_defined = `enum __time_t_defined = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___time_t_defined); }))) {
            mixin(enumMixinStr___time_t_defined);
        }
    }




    static if(!is(typeof(__timer_t_defined))) {
        private enum enumMixinStr___timer_t_defined = `enum __timer_t_defined = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___timer_t_defined); }))) {
            mixin(enumMixinStr___timer_t_defined);
        }
    }




    static if(!is(typeof(_BITS_TYPESIZES_H))) {
        private enum enumMixinStr__BITS_TYPESIZES_H = `enum _BITS_TYPESIZES_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__BITS_TYPESIZES_H); }))) {
            mixin(enumMixinStr__BITS_TYPESIZES_H);
        }
    }




    static if(!is(typeof(__SYSCALL_SLONG_TYPE))) {
        private enum enumMixinStr___SYSCALL_SLONG_TYPE = `enum __SYSCALL_SLONG_TYPE = long int;`;
        static if(is(typeof({ mixin(enumMixinStr___SYSCALL_SLONG_TYPE); }))) {
            mixin(enumMixinStr___SYSCALL_SLONG_TYPE);
        }
    }




    static if(!is(typeof(__SYSCALL_ULONG_TYPE))) {
        private enum enumMixinStr___SYSCALL_ULONG_TYPE = `enum __SYSCALL_ULONG_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___SYSCALL_ULONG_TYPE); }))) {
            mixin(enumMixinStr___SYSCALL_ULONG_TYPE);
        }
    }




    static if(!is(typeof(__DEV_T_TYPE))) {
        private enum enumMixinStr___DEV_T_TYPE = `enum __DEV_T_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___DEV_T_TYPE); }))) {
            mixin(enumMixinStr___DEV_T_TYPE);
        }
    }




    static if(!is(typeof(__UID_T_TYPE))) {
        private enum enumMixinStr___UID_T_TYPE = `enum __UID_T_TYPE = unsigned int;`;
        static if(is(typeof({ mixin(enumMixinStr___UID_T_TYPE); }))) {
            mixin(enumMixinStr___UID_T_TYPE);
        }
    }




    static if(!is(typeof(__GID_T_TYPE))) {
        private enum enumMixinStr___GID_T_TYPE = `enum __GID_T_TYPE = unsigned int;`;
        static if(is(typeof({ mixin(enumMixinStr___GID_T_TYPE); }))) {
            mixin(enumMixinStr___GID_T_TYPE);
        }
    }




    static if(!is(typeof(__INO_T_TYPE))) {
        private enum enumMixinStr___INO_T_TYPE = `enum __INO_T_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___INO_T_TYPE); }))) {
            mixin(enumMixinStr___INO_T_TYPE);
        }
    }




    static if(!is(typeof(__INO64_T_TYPE))) {
        private enum enumMixinStr___INO64_T_TYPE = `enum __INO64_T_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___INO64_T_TYPE); }))) {
            mixin(enumMixinStr___INO64_T_TYPE);
        }
    }




    static if(!is(typeof(__MODE_T_TYPE))) {
        private enum enumMixinStr___MODE_T_TYPE = `enum __MODE_T_TYPE = unsigned int;`;
        static if(is(typeof({ mixin(enumMixinStr___MODE_T_TYPE); }))) {
            mixin(enumMixinStr___MODE_T_TYPE);
        }
    }




    static if(!is(typeof(__NLINK_T_TYPE))) {
        private enum enumMixinStr___NLINK_T_TYPE = `enum __NLINK_T_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___NLINK_T_TYPE); }))) {
            mixin(enumMixinStr___NLINK_T_TYPE);
        }
    }




    static if(!is(typeof(__FSWORD_T_TYPE))) {
        private enum enumMixinStr___FSWORD_T_TYPE = `enum __FSWORD_T_TYPE = long int;`;
        static if(is(typeof({ mixin(enumMixinStr___FSWORD_T_TYPE); }))) {
            mixin(enumMixinStr___FSWORD_T_TYPE);
        }
    }




    static if(!is(typeof(__OFF_T_TYPE))) {
        private enum enumMixinStr___OFF_T_TYPE = `enum __OFF_T_TYPE = long int;`;
        static if(is(typeof({ mixin(enumMixinStr___OFF_T_TYPE); }))) {
            mixin(enumMixinStr___OFF_T_TYPE);
        }
    }




    static if(!is(typeof(__OFF64_T_TYPE))) {
        private enum enumMixinStr___OFF64_T_TYPE = `enum __OFF64_T_TYPE = long int;`;
        static if(is(typeof({ mixin(enumMixinStr___OFF64_T_TYPE); }))) {
            mixin(enumMixinStr___OFF64_T_TYPE);
        }
    }




    static if(!is(typeof(__PID_T_TYPE))) {
        private enum enumMixinStr___PID_T_TYPE = `enum __PID_T_TYPE = int;`;
        static if(is(typeof({ mixin(enumMixinStr___PID_T_TYPE); }))) {
            mixin(enumMixinStr___PID_T_TYPE);
        }
    }




    static if(!is(typeof(__RLIM_T_TYPE))) {
        private enum enumMixinStr___RLIM_T_TYPE = `enum __RLIM_T_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___RLIM_T_TYPE); }))) {
            mixin(enumMixinStr___RLIM_T_TYPE);
        }
    }




    static if(!is(typeof(__RLIM64_T_TYPE))) {
        private enum enumMixinStr___RLIM64_T_TYPE = `enum __RLIM64_T_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___RLIM64_T_TYPE); }))) {
            mixin(enumMixinStr___RLIM64_T_TYPE);
        }
    }




    static if(!is(typeof(__BLKCNT_T_TYPE))) {
        private enum enumMixinStr___BLKCNT_T_TYPE = `enum __BLKCNT_T_TYPE = long int;`;
        static if(is(typeof({ mixin(enumMixinStr___BLKCNT_T_TYPE); }))) {
            mixin(enumMixinStr___BLKCNT_T_TYPE);
        }
    }




    static if(!is(typeof(__BLKCNT64_T_TYPE))) {
        private enum enumMixinStr___BLKCNT64_T_TYPE = `enum __BLKCNT64_T_TYPE = long int;`;
        static if(is(typeof({ mixin(enumMixinStr___BLKCNT64_T_TYPE); }))) {
            mixin(enumMixinStr___BLKCNT64_T_TYPE);
        }
    }




    static if(!is(typeof(__FSBLKCNT_T_TYPE))) {
        private enum enumMixinStr___FSBLKCNT_T_TYPE = `enum __FSBLKCNT_T_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___FSBLKCNT_T_TYPE); }))) {
            mixin(enumMixinStr___FSBLKCNT_T_TYPE);
        }
    }




    static if(!is(typeof(__FSBLKCNT64_T_TYPE))) {
        private enum enumMixinStr___FSBLKCNT64_T_TYPE = `enum __FSBLKCNT64_T_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___FSBLKCNT64_T_TYPE); }))) {
            mixin(enumMixinStr___FSBLKCNT64_T_TYPE);
        }
    }




    static if(!is(typeof(__FSFILCNT_T_TYPE))) {
        private enum enumMixinStr___FSFILCNT_T_TYPE = `enum __FSFILCNT_T_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___FSFILCNT_T_TYPE); }))) {
            mixin(enumMixinStr___FSFILCNT_T_TYPE);
        }
    }




    static if(!is(typeof(__FSFILCNT64_T_TYPE))) {
        private enum enumMixinStr___FSFILCNT64_T_TYPE = `enum __FSFILCNT64_T_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___FSFILCNT64_T_TYPE); }))) {
            mixin(enumMixinStr___FSFILCNT64_T_TYPE);
        }
    }




    static if(!is(typeof(__ID_T_TYPE))) {
        private enum enumMixinStr___ID_T_TYPE = `enum __ID_T_TYPE = unsigned int;`;
        static if(is(typeof({ mixin(enumMixinStr___ID_T_TYPE); }))) {
            mixin(enumMixinStr___ID_T_TYPE);
        }
    }




    static if(!is(typeof(__CLOCK_T_TYPE))) {
        private enum enumMixinStr___CLOCK_T_TYPE = `enum __CLOCK_T_TYPE = long int;`;
        static if(is(typeof({ mixin(enumMixinStr___CLOCK_T_TYPE); }))) {
            mixin(enumMixinStr___CLOCK_T_TYPE);
        }
    }




    static if(!is(typeof(__TIME_T_TYPE))) {
        private enum enumMixinStr___TIME_T_TYPE = `enum __TIME_T_TYPE = long int;`;
        static if(is(typeof({ mixin(enumMixinStr___TIME_T_TYPE); }))) {
            mixin(enumMixinStr___TIME_T_TYPE);
        }
    }




    static if(!is(typeof(__USECONDS_T_TYPE))) {
        private enum enumMixinStr___USECONDS_T_TYPE = `enum __USECONDS_T_TYPE = unsigned int;`;
        static if(is(typeof({ mixin(enumMixinStr___USECONDS_T_TYPE); }))) {
            mixin(enumMixinStr___USECONDS_T_TYPE);
        }
    }




    static if(!is(typeof(__SUSECONDS_T_TYPE))) {
        private enum enumMixinStr___SUSECONDS_T_TYPE = `enum __SUSECONDS_T_TYPE = long int;`;
        static if(is(typeof({ mixin(enumMixinStr___SUSECONDS_T_TYPE); }))) {
            mixin(enumMixinStr___SUSECONDS_T_TYPE);
        }
    }




    static if(!is(typeof(__DADDR_T_TYPE))) {
        private enum enumMixinStr___DADDR_T_TYPE = `enum __DADDR_T_TYPE = int;`;
        static if(is(typeof({ mixin(enumMixinStr___DADDR_T_TYPE); }))) {
            mixin(enumMixinStr___DADDR_T_TYPE);
        }
    }




    static if(!is(typeof(__KEY_T_TYPE))) {
        private enum enumMixinStr___KEY_T_TYPE = `enum __KEY_T_TYPE = int;`;
        static if(is(typeof({ mixin(enumMixinStr___KEY_T_TYPE); }))) {
            mixin(enumMixinStr___KEY_T_TYPE);
        }
    }




    static if(!is(typeof(__CLOCKID_T_TYPE))) {
        private enum enumMixinStr___CLOCKID_T_TYPE = `enum __CLOCKID_T_TYPE = int;`;
        static if(is(typeof({ mixin(enumMixinStr___CLOCKID_T_TYPE); }))) {
            mixin(enumMixinStr___CLOCKID_T_TYPE);
        }
    }




    static if(!is(typeof(__TIMER_T_TYPE))) {
        private enum enumMixinStr___TIMER_T_TYPE = `enum __TIMER_T_TYPE = void *;`;
        static if(is(typeof({ mixin(enumMixinStr___TIMER_T_TYPE); }))) {
            mixin(enumMixinStr___TIMER_T_TYPE);
        }
    }




    static if(!is(typeof(__BLKSIZE_T_TYPE))) {
        private enum enumMixinStr___BLKSIZE_T_TYPE = `enum __BLKSIZE_T_TYPE = long int;`;
        static if(is(typeof({ mixin(enumMixinStr___BLKSIZE_T_TYPE); }))) {
            mixin(enumMixinStr___BLKSIZE_T_TYPE);
        }
    }




    static if(!is(typeof(__FSID_T_TYPE))) {
        private enum enumMixinStr___FSID_T_TYPE = `enum __FSID_T_TYPE = { int __val [ 2 ] ; };`;
        static if(is(typeof({ mixin(enumMixinStr___FSID_T_TYPE); }))) {
            mixin(enumMixinStr___FSID_T_TYPE);
        }
    }




    static if(!is(typeof(__SSIZE_T_TYPE))) {
        private enum enumMixinStr___SSIZE_T_TYPE = `enum __SSIZE_T_TYPE = long int;`;
        static if(is(typeof({ mixin(enumMixinStr___SSIZE_T_TYPE); }))) {
            mixin(enumMixinStr___SSIZE_T_TYPE);
        }
    }




    static if(!is(typeof(__CPU_MASK_TYPE))) {
        private enum enumMixinStr___CPU_MASK_TYPE = `enum __CPU_MASK_TYPE = unsigned long int;`;
        static if(is(typeof({ mixin(enumMixinStr___CPU_MASK_TYPE); }))) {
            mixin(enumMixinStr___CPU_MASK_TYPE);
        }
    }




    static if(!is(typeof(__OFF_T_MATCHES_OFF64_T))) {
        private enum enumMixinStr___OFF_T_MATCHES_OFF64_T = `enum __OFF_T_MATCHES_OFF64_T = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___OFF_T_MATCHES_OFF64_T); }))) {
            mixin(enumMixinStr___OFF_T_MATCHES_OFF64_T);
        }
    }




    static if(!is(typeof(__INO_T_MATCHES_INO64_T))) {
        private enum enumMixinStr___INO_T_MATCHES_INO64_T = `enum __INO_T_MATCHES_INO64_T = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___INO_T_MATCHES_INO64_T); }))) {
            mixin(enumMixinStr___INO_T_MATCHES_INO64_T);
        }
    }




    static if(!is(typeof(__RLIM_T_MATCHES_RLIM64_T))) {
        private enum enumMixinStr___RLIM_T_MATCHES_RLIM64_T = `enum __RLIM_T_MATCHES_RLIM64_T = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___RLIM_T_MATCHES_RLIM64_T); }))) {
            mixin(enumMixinStr___RLIM_T_MATCHES_RLIM64_T);
        }
    }




    static if(!is(typeof(__FD_SETSIZE))) {
        private enum enumMixinStr___FD_SETSIZE = `enum __FD_SETSIZE = 1024;`;
        static if(is(typeof({ mixin(enumMixinStr___FD_SETSIZE); }))) {
            mixin(enumMixinStr___FD_SETSIZE);
        }
    }




    static if(!is(typeof(_BITS_UINTN_IDENTITY_H))) {
        private enum enumMixinStr__BITS_UINTN_IDENTITY_H = `enum _BITS_UINTN_IDENTITY_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__BITS_UINTN_IDENTITY_H); }))) {
            mixin(enumMixinStr__BITS_UINTN_IDENTITY_H);
        }
    }




    static if(!is(typeof(WNOHANG))) {
        private enum enumMixinStr_WNOHANG = `enum WNOHANG = 1;`;
        static if(is(typeof({ mixin(enumMixinStr_WNOHANG); }))) {
            mixin(enumMixinStr_WNOHANG);
        }
    }




    static if(!is(typeof(WUNTRACED))) {
        private enum enumMixinStr_WUNTRACED = `enum WUNTRACED = 2;`;
        static if(is(typeof({ mixin(enumMixinStr_WUNTRACED); }))) {
            mixin(enumMixinStr_WUNTRACED);
        }
    }




    static if(!is(typeof(WSTOPPED))) {
        private enum enumMixinStr_WSTOPPED = `enum WSTOPPED = 2;`;
        static if(is(typeof({ mixin(enumMixinStr_WSTOPPED); }))) {
            mixin(enumMixinStr_WSTOPPED);
        }
    }




    static if(!is(typeof(WEXITED))) {
        private enum enumMixinStr_WEXITED = `enum WEXITED = 4;`;
        static if(is(typeof({ mixin(enumMixinStr_WEXITED); }))) {
            mixin(enumMixinStr_WEXITED);
        }
    }




    static if(!is(typeof(WCONTINUED))) {
        private enum enumMixinStr_WCONTINUED = `enum WCONTINUED = 8;`;
        static if(is(typeof({ mixin(enumMixinStr_WCONTINUED); }))) {
            mixin(enumMixinStr_WCONTINUED);
        }
    }




    static if(!is(typeof(WNOWAIT))) {
        private enum enumMixinStr_WNOWAIT = `enum WNOWAIT = 0x01000000;`;
        static if(is(typeof({ mixin(enumMixinStr_WNOWAIT); }))) {
            mixin(enumMixinStr_WNOWAIT);
        }
    }




    static if(!is(typeof(__WNOTHREAD))) {
        private enum enumMixinStr___WNOTHREAD = `enum __WNOTHREAD = 0x20000000;`;
        static if(is(typeof({ mixin(enumMixinStr___WNOTHREAD); }))) {
            mixin(enumMixinStr___WNOTHREAD);
        }
    }




    static if(!is(typeof(__WALL))) {
        private enum enumMixinStr___WALL = `enum __WALL = 0x40000000;`;
        static if(is(typeof({ mixin(enumMixinStr___WALL); }))) {
            mixin(enumMixinStr___WALL);
        }
    }




    static if(!is(typeof(__WCLONE))) {
        private enum enumMixinStr___WCLONE = `enum __WCLONE = 0x80000000;`;
        static if(is(typeof({ mixin(enumMixinStr___WCLONE); }))) {
            mixin(enumMixinStr___WCLONE);
        }
    }




    static if(!is(typeof(__ENUM_IDTYPE_T))) {
        private enum enumMixinStr___ENUM_IDTYPE_T = `enum __ENUM_IDTYPE_T = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___ENUM_IDTYPE_T); }))) {
            mixin(enumMixinStr___ENUM_IDTYPE_T);
        }
    }
    static if(!is(typeof(__W_CONTINUED))) {
        private enum enumMixinStr___W_CONTINUED = `enum __W_CONTINUED = 0xffff;`;
        static if(is(typeof({ mixin(enumMixinStr___W_CONTINUED); }))) {
            mixin(enumMixinStr___W_CONTINUED);
        }
    }




    static if(!is(typeof(__WCOREFLAG))) {
        private enum enumMixinStr___WCOREFLAG = `enum __WCOREFLAG = 0x80;`;
        static if(is(typeof({ mixin(enumMixinStr___WCOREFLAG); }))) {
            mixin(enumMixinStr___WCOREFLAG);
        }
    }




    static if(!is(typeof(__WORDSIZE))) {
        private enum enumMixinStr___WORDSIZE = `enum __WORDSIZE = 64;`;
        static if(is(typeof({ mixin(enumMixinStr___WORDSIZE); }))) {
            mixin(enumMixinStr___WORDSIZE);
        }
    }




    static if(!is(typeof(__WORDSIZE_TIME64_COMPAT32))) {
        private enum enumMixinStr___WORDSIZE_TIME64_COMPAT32 = `enum __WORDSIZE_TIME64_COMPAT32 = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___WORDSIZE_TIME64_COMPAT32); }))) {
            mixin(enumMixinStr___WORDSIZE_TIME64_COMPAT32);
        }
    }




    static if(!is(typeof(__SYSCALL_WORDSIZE))) {
        private enum enumMixinStr___SYSCALL_WORDSIZE = `enum __SYSCALL_WORDSIZE = 64;`;
        static if(is(typeof({ mixin(enumMixinStr___SYSCALL_WORDSIZE); }))) {
            mixin(enumMixinStr___SYSCALL_WORDSIZE);
        }
    }
    static if(!is(typeof(_SYS_CDEFS_H))) {
        private enum enumMixinStr__SYS_CDEFS_H = `enum _SYS_CDEFS_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__SYS_CDEFS_H); }))) {
            mixin(enumMixinStr__SYS_CDEFS_H);
        }
    }
    static if(!is(typeof(__THROW))) {
        private enum enumMixinStr___THROW = `enum __THROW = __attribute__ ( ( __nothrow__ ) );`;
        static if(is(typeof({ mixin(enumMixinStr___THROW); }))) {
            mixin(enumMixinStr___THROW);
        }
    }




    static if(!is(typeof(__THROWNL))) {
        private enum enumMixinStr___THROWNL = `enum __THROWNL = __attribute__ ( ( __nothrow__ ) );`;
        static if(is(typeof({ mixin(enumMixinStr___THROWNL); }))) {
            mixin(enumMixinStr___THROWNL);
        }
    }
    static if(!is(typeof(__ptr_t))) {
        private enum enumMixinStr___ptr_t = `enum __ptr_t = void *;`;
        static if(is(typeof({ mixin(enumMixinStr___ptr_t); }))) {
            mixin(enumMixinStr___ptr_t);
        }
    }
    static if(!is(typeof(__flexarr))) {
        private enum enumMixinStr___flexarr = `enum __flexarr = [ ];`;
        static if(is(typeof({ mixin(enumMixinStr___flexarr); }))) {
            mixin(enumMixinStr___flexarr);
        }
    }




    static if(!is(typeof(__glibc_c99_flexarr_available))) {
        private enum enumMixinStr___glibc_c99_flexarr_available = `enum __glibc_c99_flexarr_available = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___glibc_c99_flexarr_available); }))) {
            mixin(enumMixinStr___glibc_c99_flexarr_available);
        }
    }
    static if(!is(typeof(__attribute_malloc__))) {
        private enum enumMixinStr___attribute_malloc__ = `enum __attribute_malloc__ = __attribute__ ( ( __malloc__ ) );`;
        static if(is(typeof({ mixin(enumMixinStr___attribute_malloc__); }))) {
            mixin(enumMixinStr___attribute_malloc__);
        }
    }






    static if(!is(typeof(__attribute_pure__))) {
        private enum enumMixinStr___attribute_pure__ = `enum __attribute_pure__ = __attribute__ ( ( __pure__ ) );`;
        static if(is(typeof({ mixin(enumMixinStr___attribute_pure__); }))) {
            mixin(enumMixinStr___attribute_pure__);
        }
    }




    static if(!is(typeof(__attribute_const__))) {
        private enum enumMixinStr___attribute_const__ = `enum __attribute_const__ = __attribute__ ( cast( __const__ ) );`;
        static if(is(typeof({ mixin(enumMixinStr___attribute_const__); }))) {
            mixin(enumMixinStr___attribute_const__);
        }
    }




    static if(!is(typeof(__attribute_used__))) {
        private enum enumMixinStr___attribute_used__ = `enum __attribute_used__ = __attribute__ ( ( __used__ ) );`;
        static if(is(typeof({ mixin(enumMixinStr___attribute_used__); }))) {
            mixin(enumMixinStr___attribute_used__);
        }
    }




    static if(!is(typeof(__attribute_noinline__))) {
        private enum enumMixinStr___attribute_noinline__ = `enum __attribute_noinline__ = __attribute__ ( ( __noinline__ ) );`;
        static if(is(typeof({ mixin(enumMixinStr___attribute_noinline__); }))) {
            mixin(enumMixinStr___attribute_noinline__);
        }
    }




    static if(!is(typeof(__attribute_deprecated__))) {
        private enum enumMixinStr___attribute_deprecated__ = `enum __attribute_deprecated__ = __attribute__ ( ( __deprecated__ ) );`;
        static if(is(typeof({ mixin(enumMixinStr___attribute_deprecated__); }))) {
            mixin(enumMixinStr___attribute_deprecated__);
        }
    }
    static if(!is(typeof(__attribute_warn_unused_result__))) {
        private enum enumMixinStr___attribute_warn_unused_result__ = `enum __attribute_warn_unused_result__ = __attribute__ ( ( __warn_unused_result__ ) );`;
        static if(is(typeof({ mixin(enumMixinStr___attribute_warn_unused_result__); }))) {
            mixin(enumMixinStr___attribute_warn_unused_result__);
        }
    }






    static if(!is(typeof(__always_inline))) {
        private enum enumMixinStr___always_inline = `enum __always_inline = __inline __attribute__ ( ( __always_inline__ ) );`;
        static if(is(typeof({ mixin(enumMixinStr___always_inline); }))) {
            mixin(enumMixinStr___always_inline);
        }
    }






    static if(!is(typeof(__extern_inline))) {
        private enum enumMixinStr___extern_inline = `enum __extern_inline = extern __inline __attribute__ ( ( __gnu_inline__ ) );`;
        static if(is(typeof({ mixin(enumMixinStr___extern_inline); }))) {
            mixin(enumMixinStr___extern_inline);
        }
    }




    static if(!is(typeof(__extern_always_inline))) {
        private enum enumMixinStr___extern_always_inline = `enum __extern_always_inline = extern __inline __attribute__ ( ( __always_inline__ ) ) __attribute__ ( ( __gnu_inline__ ) );`;
        static if(is(typeof({ mixin(enumMixinStr___extern_always_inline); }))) {
            mixin(enumMixinStr___extern_always_inline);
        }
    }




    static if(!is(typeof(__fortify_function))) {
        private enum enumMixinStr___fortify_function = `enum __fortify_function = extern __inline __attribute__ ( ( __always_inline__ ) ) __attribute__ ( ( __gnu_inline__ ) ) ;`;
        static if(is(typeof({ mixin(enumMixinStr___fortify_function); }))) {
            mixin(enumMixinStr___fortify_function);
        }
    }




    static if(!is(typeof(__restrict_arr))) {
        private enum enumMixinStr___restrict_arr = `enum __restrict_arr = __restrict;`;
        static if(is(typeof({ mixin(enumMixinStr___restrict_arr); }))) {
            mixin(enumMixinStr___restrict_arr);
        }
    }
    static if(!is(typeof(__HAVE_GENERIC_SELECTION))) {
        private enum enumMixinStr___HAVE_GENERIC_SELECTION = `enum __HAVE_GENERIC_SELECTION = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___HAVE_GENERIC_SELECTION); }))) {
            mixin(enumMixinStr___HAVE_GENERIC_SELECTION);
        }
    }




    static if(!is(typeof(_SYS_SELECT_H))) {
        private enum enumMixinStr__SYS_SELECT_H = `enum _SYS_SELECT_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__SYS_SELECT_H); }))) {
            mixin(enumMixinStr__SYS_SELECT_H);
        }
    }






    static if(!is(typeof(__NFDBITS))) {
        private enum enumMixinStr___NFDBITS = `enum __NFDBITS = ( 8 * cast( int ) ( __fd_mask ) .sizeof );`;
        static if(is(typeof({ mixin(enumMixinStr___NFDBITS); }))) {
            mixin(enumMixinStr___NFDBITS);
        }
    }
    static if(!is(typeof(FD_SETSIZE))) {
        private enum enumMixinStr_FD_SETSIZE = `enum FD_SETSIZE = 1024;`;
        static if(is(typeof({ mixin(enumMixinStr_FD_SETSIZE); }))) {
            mixin(enumMixinStr_FD_SETSIZE);
        }
    }




    static if(!is(typeof(NFDBITS))) {
        private enum enumMixinStr_NFDBITS = `enum NFDBITS = ( 8 * cast( int ) ( __fd_mask ) .sizeof );`;
        static if(is(typeof({ mixin(enumMixinStr_NFDBITS); }))) {
            mixin(enumMixinStr_NFDBITS);
        }
    }
    static if(!is(typeof(_SYS_SYSMACROS_H))) {
        private enum enumMixinStr__SYS_SYSMACROS_H = `enum _SYS_SYSMACROS_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__SYS_SYSMACROS_H); }))) {
            mixin(enumMixinStr__SYS_SYSMACROS_H);
        }
    }
    static if(!is(typeof(__SYSMACROS_DECL_TEMPL))) {
        private enum enumMixinStr___SYSMACROS_DECL_TEMPL = `enum __SYSMACROS_DECL_TEMPL = ( rtype , name , proto ) extern rtype gnu_dev_ ## name proto __attribute__ ( ( __nothrow__ ) ) __attribute__ ( cast( __const__ ) ) ;;`;
        static if(is(typeof({ mixin(enumMixinStr___SYSMACROS_DECL_TEMPL); }))) {
            mixin(enumMixinStr___SYSMACROS_DECL_TEMPL);
        }
    }




    static if(!is(typeof(__SYSMACROS_IMPL_TEMPL))) {
        private enum enumMixinStr___SYSMACROS_IMPL_TEMPL = `enum __SYSMACROS_IMPL_TEMPL = ( rtype , name , proto ) __extension__ extern __inline __attribute__ ( ( __gnu_inline__ ) ) __attribute__ ( cast( __const__ ) ) rtype __attribute__ ( ( __nothrow__ ) ) gnu_dev_ ## name proto;`;
        static if(is(typeof({ mixin(enumMixinStr___SYSMACROS_IMPL_TEMPL); }))) {
            mixin(enumMixinStr___SYSMACROS_IMPL_TEMPL);
        }
    }
    static if(!is(typeof(_SYS_TYPES_H))) {
        private enum enumMixinStr__SYS_TYPES_H = `enum _SYS_TYPES_H = 1;`;
        static if(is(typeof({ mixin(enumMixinStr__SYS_TYPES_H); }))) {
            mixin(enumMixinStr__SYS_TYPES_H);
        }
    }
    static if(!is(typeof(__BIT_TYPES_DEFINED__))) {
        private enum enumMixinStr___BIT_TYPES_DEFINED__ = `enum __BIT_TYPES_DEFINED__ = 1;`;
        static if(is(typeof({ mixin(enumMixinStr___BIT_TYPES_DEFINED__); }))) {
            mixin(enumMixinStr___BIT_TYPES_DEFINED__);
        }
    }
}
