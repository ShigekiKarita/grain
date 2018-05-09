.PHONY: test clean kernel

CUDA_COMPUTE_CAPABILITY := `tool/compute_capability.out 0`
CUDA_BIT := $(shell getconf LONG_BIT)
NO_CUDA := false
DUB_BUILD := unittest

ifeq ($(NO_CUDA),true)
	DUB_OPTS = -b=$(DUB_BUILD)
else
	CUDA_DEPS = tool/compute_capability.out source/grain/kernel.di kernel/kernel.ptx
	DUB_OPTS = -b=cuda-$(DUB_BUILD)
endif

test: $(CUDA_DEPS)
	dub test --compiler=ldc2 $(DUB_OPTS)

kernel/%.ptx: kernel/%.d
	ldc2 $< --mdcompute-targets=cuda-$(CUDA_COMPUTE_CAPABILITY)0 -H -Hd kernel -mdcompute-file-prefix=$(shell basename -s .d $<) -I=source
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
	find . -type f -name "*.ptx" -print -delete
	find . -type f -name "*.di" -print -delete
	find . -type f -name "*.out" -print -delete

