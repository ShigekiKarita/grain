.PHONY: test clean kernel

CUDA_COMPUTE_CAPABILITY := `tool/compute_capability.out 0`
CUDA_BIT := $(shell getconf LONG_BIT)

test: source/grain/kernel.di tool/compute_capability.out
	dub test --compiler=ldc2

kernel/%.ptx: kernel/%.d tool/compute_capability.out
	ldc2 $< --mdcompute-targets=cuda-$(CUDA_COMPUTE_CAPABILITY)0 -H -Hd kernel -mdcompute-file-prefix=$(shell basename -s .d $<)
	mv $(shell basename -s .d $<)_cuda$(CUDA_COMPUTE_CAPABILITY)0_$(CUDA_BIT).ptx $@

source/grain/%.di: kernel/%.ptx
	cat kernel/$(shell basename -s .ptx $<).di     > $@
	@echo "/**"                                   >> $@
	@echo " * generated PTX (see Makefile %.di) " >> $@
	@echo "**/"                                   >> $@
	@echo 'enum ptx = q"EOS'                      >> $@
	@cat $<                                       >> $@
	@echo 'EOS";'                                 >> $@

tool/%.out: tool/%.cu
	nvcc $< -o $@ -lcuda -std=c++11

clean:
	rm -rfv **/*.di **/*.ptx **/*.out
