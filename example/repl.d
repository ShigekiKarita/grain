/**
   TODO: test cuda
 */

import std.experimental.logger : Logger, LogLevel, sharedLog;

class REPLLogger : Logger {
    this(LogLevel lv) @safe {
        super(lv);
    }

    override void writeLogMsg(ref LogEntry payload) {
        import colorize : color, fg;
        payload.msg = color(payload.msg, fg.blue);
        sharedLog.forwardMsg(payload);
    }
}

REPLLogger logger;

struct CompilerOpt {
    import std.string : split;

    string compiler, ver, pic, mode;
    string[string] build;
    string[] flags;

    this(string compiler, string mode, string flags_) {
        this.mode = mode;
        this.compiler = compiler;
        this.flags = flags_.split;
        // FIXME use dub library instead of this reinvention
        if (compiler == "dmd") {
            ver = "-version";
            pic = "-fPIC";
            build["debug"] = "-g -debug";
            build["release"] = "-release -inline";
            build["optimize"] = build["release"] ~ " -O";
        }
        else if (compiler == "ldc2") {
            ver = "-d-version";
            pic = "-relocation-model=pic";
            build["debug"] = "-g -d-debug -vcolumns";
            build["release"] = "-release";
            build["optimize"] = build["release"] ~ " -O";
        }
        else {
            throw new Exception("unsupported compiler: " ~ compiler);
        }
    }

    auto cmd() {
        return [compiler, pic] ~ build[mode].split;
    }
}

struct DUBEngine {
    import drepl.engines : EngineResult;

    import std.array : array;
    import std.container : RedBlackTree;
    import std.typecons : Tuple;
    import std.stdio : File;

    RedBlackTree!string _dubFlags;
    CompilerOpt _compiler;
    string _tmpDir;
    size_t _id;

    @disable this(this);
    @disable new(size_t);

    ~this() {
        import std.file : rmdirRecurse;
        if (_tmpDir)
            rmdirRecurse(_tmpDir);
    }

    this(string[] packages, CompilerOpt compiler) {
        import core.sys.posix.unistd, std.random;
        import drepl : mkdtemp;

        auto tmpDir = mkdtemp();
        this(packages, compiler, tmpDir);
    }

    this(string[] packages, CompilerOpt compiler, string tmpDir) {
        import std.algorithm : map, each;
        import std.file : exists, rmdirRecurse, mkdirRecurse;

        _compiler = compiler;
        _tmpDir = tmpDir;
        _dubFlags = new RedBlackTree!string;

        if (_tmpDir.exists) {
            rmdirRecurse(_tmpDir);
        }
        mkdirRecurse(_tmpDir);

        packages.each!(p => this.registerPackage(p));
    }

    void registerPackage(string _package) {
        import std.file : exists;
        import std.algorithm : map;
        import std.json : parseJSON;
        import std.range : chain;
        import std.process : execute;

        // FIXME use dub library instead of subprocess
        auto dubDescribe = execute(["dub", "describe", _package,
                "--compiler=" ~ _compiler.compiler]);
        if (dubDescribe.status != 0) {
            throw new Exception("failed: $ dub describe " ~ _package ~ "\n"
                    ~ dubDescribe.output ~ "\n\nsuggest: $ dub fetch " ~ _package);
        }
        auto dubInfo = parseJSON(dubDescribe.output);
        auto target = dubInfo["targets"][0];
        assert(target["rootPackage"].str == _package);
        auto build = target["buildSettings"];
        auto read(string key) {
            return build[key].array.map!"a.str".array;
        }

        _dubFlags.insert(chain(read("importPaths").map!(a => "-I" ~ a),
                               read("libs").map!(a => "-L-l" ~ a),
                               read("versions").map!(a => _compiler.ver ~ "=" ~ a),
                               read("linkerFiles")));

        // TODO re-think how to read _package's static lib
        immutable targetFileName = build["targetPath"].str ~ "/lib" ~ build["targetName"]
            .str ~ ".a";
        if (targetFileName.exists) {
            _dubFlags.insert(targetFileName);
        }
    }

    string compileModule(string path) {
        import std.process : execute;
        import std.format : format;
        import std.file : exists;
        import std.regex : ctRegex, replaceAll;
        import std.range : chain;

        logger.trace("compile path: ", path);
        // TODO support build mode (debug, release, optimize)
        auto args = chain(_compiler.cmd,
                          ["-I" ~ _tmpDir, "-of" ~ path ~ ".so", "-shared", path, "-L-l:libphobos2.so"],
                          _dubFlags[]).array;

        foreach (i; 0 .. _id)
            args ~= "-L" ~ _tmpDir ~ format("/_mod%s.so", i);

        logger.trace("compile with: ", args);
        auto dmd = execute(args);
        enum cleanErr = ctRegex!(`^.*Error: `, "m");
        if (dmd.status != 0)
            return dmd.output.replaceAll(cleanErr, "");
        if (!exists(path ~ ".so"))
            return path ~ ".so not found";
        return null;
    }

    EngineResult evalDecl(in char[] decl) {
        import std.string : outdent;
        auto m = newModule();
        m.f.writefln(q{
            // for public imports
            public %1$s

            extern(C) void _decls()
            {
                import std.algorithm, std.stdio;
                writef("%%-(%%s, %%)", [__traits(allMembers, _mod%2$s)][1 .. $]
                       .filter!(d => !d.startsWith("_")));
            }
            }.outdent(), decl, _id);
        m.f.close();

        if (auto err = compileModule(m.path))
            return EngineResult(false, "", err);

        ++_id;

        auto func = cast(void function()) loadFunc(m.path, "_decls");
        return captureOutput(func);
    }

    EngineResult evalExpr(in char[] expr) {
        import std.string : outdent;
        auto m = newModule();
        m.f.writefln(q{
                extern(C) void _expr()
                {
                    import std.stdio;
                    static if (is(typeof((() => (%1$s))()) == void))
                        (%1$s), write("void");
                    else
                        write((%1$s));
                }
            }.outdent(), expr);
        m.f.close();

        if (auto err = compileModule(m.path))
            return EngineResult(false, "", err);

        ++_id;

        auto func = cast(void function()) loadFunc(m.path, "_expr");
        return captureOutput(func);
    }

    EngineResult evalStmt(in char[] stmt) {
        auto m = newModule();
        m.f.writefln(q{
                extern(C) void _run()
                {
                    %s
                }
            }, stmt);
        m.f.close();

        if (auto err = compileModule(m.path))
            return EngineResult(false, "", err);

        ++_id;

        auto func = cast(void function()) loadFunc(m.path, "_run");
        return captureOutput(func);
    }

private:
    EngineResult captureOutput(void function() dg) {
        import std.stdio : stdout, stderr;
        import std.string : toStringz;
        import std.path : buildPath;
        import std.file : readText;

        // TODO: cleanup, error checking...
        import core.sys.posix.fcntl, core.sys.posix.unistd, std.conv : octal;

        stdout.flush();
        stderr.flush();
        immutable saveOut = dup(stdout.fileno), saveErr = dup(stderr.fileno),
            capOut = open(toStringz(_tmpDir ~ "/_stdout"),
                    O_WRONLY | O_CREAT | O_TRUNC, octal!600),
            capErr = open(toStringz(_tmpDir ~ "/_stderr"),
                    O_WRONLY | O_CREAT | O_TRUNC, octal!600);
        dup2(capOut, stdout.fileno);
        dup2(capErr, stderr.fileno);

        bool success = true;
        try {
            dg();
        }
        catch (Exception e) {
            success = false;
            stderr.writeln(e.toString());
        }
        stdout.flush();
        stderr.flush();
        close(stdout.fileno);
        close(stderr.fileno);
        dup2(saveOut, stdout.fileno);
        dup2(saveErr, stderr.fileno);
        close(saveOut);
        close(saveErr);
        return EngineResult(success, readText(_tmpDir ~ "/_stdout"), readText(_tmpDir ~ "/_stderr"));
    }

    Tuple!(File, "f", string, "path") newModule() {
        import std.path : buildPath;
        import std.format : format;
        auto path = buildPath(_tmpDir, format("_mod%s", _id));
        auto f = File(path ~ ".d", "w");
        writeHeader(f);
        return typeof(return)(f, path);
    }

    void writeHeader(ref File f) {
        if (_id > 0) {
            f.write("import _mod0");
            foreach (i; 1 .. _id)
                f.writef(", _mod%s", i);
            f.write(";");
        }
    }

    void* loadFunc(string path, string name) {
        import std.string : toStringz;
        import core.runtime, core.demangle, core.sys.posix.dlfcn;

        auto lib = Runtime.loadLibrary(path ~ ".so");
        if (lib is null) {
            auto msg = dlerror();
            import core.stdc.string : strlen;

            throw new Exception("failed to load " ~ path ~ ".so (" ~ msg[0 .. strlen(
                    msg)].idup ~ ")");
        }
        return dlsym(lib, toStringz(name));
    }
}


import drepl.engines : isEngine;
static assert(isEngine!DUBEngine);


void main(string[] args) {
    import std.stdio : writeln;
    import std.getopt : getopt, defaultGetoptPrinter;
    import std.string : split, toStringz;
    import std.path : buildPath;
    import std.process : environment;
    import core.stdc.string : strlen;
    import deimos.linenoise;
    import drepl : interpreter, InterpreterResult;
    import colorize : color, cwriteln, fg;

    bool verbose = false;
    string build = "debug";
    string compiler = "dmd";
    string flags = "-debug -g -w";
    string packages = "grain";
    auto opt = getopt(args, "verbose|V", "verbose mode for logging", &verbose,
            "flags|F", "compiler flags", &flags, "compiler|C",
            "compiler binary", &compiler, "build|B",
            "build mode (debug, release, optimize)", &build, "packages|P",
            "list of DUB packages", &packages);

    if (opt.helpWanted) {
        defaultGetoptPrinter("playground of grain", opt.options);
        return;
    }

    logger = new REPLLogger(verbose ? LogLevel.trace : LogLevel.warning);

    writeln("Welcome to grain REPL.");
    auto history = buildPath(environment.get("HOME", ""), ".drepl_history")
        .toStringz();
    linenoiseHistoryLoad(history);

    auto intp = interpreter(DUBEngine(packages.split, CompilerOpt(compiler, build,
            flags)));

    char* line;
    const(char)* prompt = "grain> ";
    while ((line = linenoise(prompt)) !is null) {
        linenoiseHistoryAdd(line);
        linenoiseHistorySave(history);

        auto res = intp.interpret(line[0 .. strlen(line)]);
        final switch (res.state) with (InterpreterResult.State) {
        case incomplete:
            prompt = " | ";
            break;

        case success:
        case error:
            if (res.stderr.length)
                cwriteln(res.stderr.color(fg.red));
            if (res.stdout.length)
                cwriteln(res.stdout.color(fg.green));
            prompt = "grain> ";
            break;
        }
    }
}
