import dub_engine;


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
