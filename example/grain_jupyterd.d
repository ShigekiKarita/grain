import juypterd.interpreter;
import juypterd.kernel;
import zmqd;
import std.stdio;
import dub_engine;
import std.experimental.logger;
import drepl.interpreter : InterpreterResult;
import drepl.engines;


struct DUBInterpreter(Engine) if (isEngine!Engine)
{
    import std.algorithm, std.array, std.conv, std.string, std.typecons;

    alias IR = InterpreterResult;

    IR interpret(const(char)[] line, bool writeExpr=false)
    {
        // ignore empty lines or comment without incomplete input
        if (!_incomplete.data.length && (!line.length || byToken(cast(ubyte[])line).empty))
            return IR(IR.State.success);

        _incomplete.put(line);
        _incomplete.put('\n');
        auto input = _incomplete.data;

        // dismiss buffer after two consecutive empty lines
        if (input.endsWith("\n\n\n"))
        {
            _incomplete.clear();
            return IR(IR.State.error, "", "You typed two blank lines. Starting a new command.");
        }

        immutable kind = classify(input);
        EngineResult res;
        final switch (kind)
        {
        case Kind.Decl:
            auto r = _engine.evalDecl(input);
            res.success = r.success;
            break;
        case Kind.Stmt:
            res = _engine.evalStmt(input);
            break;
        case Kind.Expr:
            auto r = _engine.evalExpr(input);
            if (writeExpr) {
                res = r;
            } else {
                res.success = r.success;
            }
            break;

        case Kind.WhiteSpace:
            return IR(IR.State.success);

        case Kind.Incomplete:
            return IR(IR.State.incomplete);

        case Kind.Error:
            _incomplete.clear();
            return IR(IR.State.error, "", "Error parsing '"~input.strip.idup~"'.");
        }
        _incomplete.clear();
        return IR(res.success ? IR.State.success : IR.State.error, res.stdout, res.stderr);
    }

    enum Kind { Decl, Stmt, Expr, WhiteSpace, Incomplete, Error, }

    import dparse.lexer, dparse.parser, dparse.rollback_allocator;

    Kind classify(in char[] input)
    {
        scope cache = new StringCache(StringCache.defaultBucketCount);
        auto tokens = getTokensForParser(cast(ubyte[])input, LexerConfig(), cache);
        if (tokens.empty) return Kind.WhiteSpace;

        auto tokenIds = tokens.map!(t => t.type)();
        if (!tokenIds.balancedParens(tok!"{", tok!"}") ||
            !tokenIds.balancedParens(tok!"(", tok!")") ||
            !tokenIds.balancedParens(tok!"[", tok!"]"))
            return Kind.Incomplete;

        import std.typetuple : TypeTuple;
        foreach (kind; TypeTuple!(Kind.Decl, Kind.Stmt, Kind.Expr))
            if (parse!kind(tokens))
                return kind;
        return Kind.Error;
    }

    bool parse(Kind kind)(in Token[] tokens)
    {
        import dparse.rollback_allocator : RollbackAllocator;
        scope parser = new Parser();
        RollbackAllocator allocator;
        static bool hasErr;
        hasErr = false;
        parser.fileName = "drepl";
        parser.setTokens(tokens);
        parser.allocator = &allocator;
        parser.messageDg = delegate(string file, size_t ln, size_t col, string msg, bool isErr) {
            if (isErr)
                hasErr = true;
        };
        static if (kind == Kind.Decl)
        {
            do
            {
                if (!parser.parseDeclaration()) return false;
            } while (parser.moreTokens());
        }
        else static if (kind == Kind.Stmt)
        {
            do
            {
                if (!parser.parseStatement()) return false;
            } while (parser.moreTokens());
        }
        else static if (kind == Kind.Expr)
        {
            if (!parser.parseExpression() || parser.moreTokens())
                return false;
        }
        return !hasErr;
    }

    // unittest
    // {
    //     auto intp = interpreter(echoEngine());
    //     assert(intp.classify("3+2") == Kind.Expr);
    //     // only single expressions
    //     assert(intp.classify("3+2 foo()") == Kind.Error);
    //     assert(intp.classify("3+2;") == Kind.Stmt);
    //     // multiple statements
    //     assert(intp.classify("3+2; foo();") == Kind.Stmt);
    //     assert(intp.classify("struct Foo {}") == Kind.Decl);
    //     // multiple declarations
    //     assert(intp.classify("void foo() {} void bar() {}") == Kind.Decl);
    //     // can't currently mix declarations and statements
    //     assert(intp.classify("void foo() {} foo();") == Kind.Error);
    //     // or declarations and expressions
    //     assert(intp.classify("void foo() {} foo()") == Kind.Error);
    //     // or statments and expressions
    //     assert(intp.classify("foo(); foo()") == Kind.Error);

    //     assert(intp.classify("import std.stdio;") == Kind.Decl);
    // }

    Engine _engine;
    Appender!(char[]) _incomplete;

}


final class DynamicDUBInterpreter : Interpreter
{
    LanguageInfo li = LanguageInfo("D","2.081.1",".d", "text/plain");
    
    private import drepl.engines;
    
    DUBInterpreter!DUBEngine intp;
    InterpreterResult last;
    
    this(DUBEngine engine)
    {
        import std.algorithm : move;
        intp = DUBInterpreter!DUBEngine(move(engine));
        // intp.interpret(`import std.experimental.all;`);
    }
    override InterpreterResult interpret(const(char)[] code)
    {
	    import std.string:splitLines, join, empty;
	    import std.algorithm;
	    import std.array:array;
	    import std.range:back;
	    import std.stdio:stderr,writeln;
	    InterpreterResult.State state;
        auto codeLines = code.splitLines.array;
        InterpreterResult[] result;
	    // auto result = codeLines.map!(line=>intp.interpret(line)).array;
        foreach (i, line; codeLines) {
            result ~= intp.interpret(line, i == codeLines.length - 1);
        }
	    auto success = (result.all!(line=>line.state != InterpreterResult.State.error));
	    if (!success)
		   state = InterpreterResult.State.error;
	    else 
		    state = ((result.length==0) || (result.back.state != InterpreterResult.State.incomplete) )? InterpreterResult.State.success : InterpreterResult.State.incomplete;
	    auto errorOutput = result.map!(line => line.stderr).join("\n");
	    auto stdOutput = result.map!(line => line.state == InterpreterResult.State.success
                                     && !line.stdout.empty
                                     ? line.stdout ~ "\n" : "").join;
	    stderr.writeln(state,stdOutput,errorOutput);
	    return InterpreterResult(state,stdOutput,errorOutput);
    }
    
    override ref const(LanguageInfo) languageInfo()
    {
        return li;
    }
}



int main(string[] args) {
    import std.stdio : writeln;
    import std.getopt : getopt, defaultGetoptPrinter;
    import std.string : split, toStringz;
    import std.path : buildPath;
    import std.process : environment;
    import core.stdc.string : strlen;
    import deimos.linenoise;
    import drepl : interpreter, InterpreterResult;
    import colorize : color, cwriteln, fg;

    bool verbose = true;
    string build = "debug";
    string compiler = "dmd";
    string flags = "-debug -g -w";
    string packages = "grain";
    // auto opt = getopt(args, "verbose|V", "verbose mode for logging", &verbose,
    //         "flags|F", "compiler flags", &flags, "compiler|C",
    //         "compiler binary", &compiler, "build|B",
    //         "build mode (debug, release, optimize)", &build, "packages|P",
    //         "list of DUB packages", &packages);

    // if (opt.helpWanted) {
    //     defaultGetoptPrinter("playground of grain", opt.options);
    //     return;
    // }
    logger = new REPLLogger(verbose ? LogLevel.trace : LogLevel.warning);
    Interpreter i = new DynamicDUBInterpreter(
        DUBEngine(packages.split,
                  CompilerOpt(compiler, build, flags))
        );
    // );

    // Interpreter i;
    // switch(args[1])
    // {
    // case "echo":
    //     i = new EchoInterpreter();
    //     break;
    // case "d":
    //     i = new DInterpreter();
    //     break;
    // default:
    //     return 1;
    // }

    auto k = Kernel(i,/*connection string=*/args[2]);
    k.mainloop();
    return 0;
}

