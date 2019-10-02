/// type check
module grain.traits;


/// https://github.com/libmir/mir-algorithm/blob/0432da3869cbbffdb7cb3cc97522f30f318673ba/source/mir/type_info.d#L94
package template hasDestructor(T)
{
    import std.traits : Unqual;

    static if (is(T == struct))
    {
        static if (__traits(hasMember, Unqual!T, "__xdtor"))
            enum hasDestructor = __traits(isSame, Unqual!T, __traits(parent, T.init.__xdtor));
        else
            enum hasDestructor = false;
    }
    else
    static if (is(T == class))
    {
        enum hasDestructor = __traits(hasMember, Unqual!T, "__xdtor");
    }
    else
    {
        enum hasDestructor = false;
    }
}
