.PHONY: test clean kernel

test: kernel/kernel.ptx
	dub test --compiler=ldc2

kernel/%.ptx: kernel/%.d
	ldc2 $< -march=nvptx64 -betterC -c -of=- -H -Hd source/grain \
	| sed "s/.target generic/.target sm_30/g" \
	| sed "s/.func/.entry/g" > $@

clean:
	rm -rfv **/*.di **/*.ptx

