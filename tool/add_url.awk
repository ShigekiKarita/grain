{
    if ($0 ~ /"/ && $0 !~ /;/) {
        print $0 " [URL=" substr($1,0,length($1)) ".html\"]"
    }
    else print $0
}
