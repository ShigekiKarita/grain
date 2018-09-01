import juypterd.interpreter;
import juypterd.kernel;
import zmqd;
import std.stdio;
import dub_engine;


final class DUBInterpreter : Interpreter
{
    LanguageInfo li = LanguageInfo("D","2.081.1",".d", "text/plain");
    
    private import drepl.engines;
    
    typeof(interpreter(DUBEngine.init)) intp;
    InterpreterResult last;
    
    this(DUBEngine engine)
    {
        import std.algorithm : move;
        intp = interpreter(move(engine));
        // intp.interpret(`import std.experimental.all;`);
    }
    override InterpreterResult interpret(const(char)[] code)
    {
	    import std.string:splitLines, join;
	    import std.algorithm;
	    import std.array:array;
	    import std.range:back;
	    import std.stdio:stderr,writeln;
	    InterpreterResult.State state;
	    auto result = code.splitLines.map!(line=>intp.interpret(line)).array;
	    auto success = (result.all!(line=>line.state != InterpreterResult.State.error));
	    if (!success)
		   state = InterpreterResult.State.error;
	    else 
		    state = ((result.length==0) || (result.back.state != InterpreterResult.State.incomplete) )? InterpreterResult.State.success : InterpreterResult.State.incomplete;
	    auto errorOutput = result.map!(line=>line.stderr).join("\n");
	    auto stdOutput = result.map!(line=>line.stdout).join("\n");
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
    Interpreter i = new DUBInterpreter(
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

