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

kernel/kernel_lib.ptx: kernel/kernel_lib.cu
	# clang-6.0 -c -S -emit-llvm $< --cuda-gpu-arch=sm_$(CUDA_COMPUTE_CAPABILITY)
	# llc-6.0 -mcpu=sm_$(CUDA_COMPUTE_CAPABILITY) $(shell basename -s .cu $<)-cuda-nvptx64-nvidia-cuda-sm_$(CUDA_COMPUTE_CAPABILITY).ll -o $@
	nvcc -ptx -arch=sm_$(CUDA_COMPUTE_CAPABILITY) $< -o $@ -std=c++11

kernel/kernel.ptx: kernel/kernel.d kernel/kernel_lib.ptx
	ldc2 $< --mdcompute-targets=cuda-$(CUDA_COMPUTE_CAPABILITY)0 -H -Hd kernel -mdcompute-file-prefix=$(shell basename -s .d $<) -I=source
	mv $(shell basename -s .d $<)_cuda$(CUDA_COMPUTE_CAPABILITY)0_$(CUDA_BIT).ptx $@


source/grain/kernel.di: kernel/kernel.ptx kernel/kernel_lib.ptx
	cat kernel/$(shell basename -s .ptx $<).di     > $@
	@echo "/**"                                   >> $@
	@echo " * generated PTX (see Makefile %.di) " >> $@
	@echo "**/"                                   >> $@
	@echo 'enum ptx = q"EOS'                      >> $@
	@cat kernel/kernel.ptx                        >> $@
	@echo 'EOS";'                                 >> $@
	@echo "/**"                                   >> $@
	@echo " * generated PTX (see Makefile %.di) " >> $@
	@echo "**/"                                   >> $@
	@echo 'enum cxxptx = q"EOS'                   >> $@
	@cat kernel/kernel_lib.ptx                    >> $@
	@echo 'EOS";'                                 >> $@

tool/%.out: tool/%.cu
	nvcc $< -o $@ -lcuda -std=c++11

clean:
	find . -type f -name "*.ll" -print -delete
	find . -type f -name "*.ptx" -print -delete
	find . -type f -name "*.di" -print -delete
	find . -type f -name "*.out" -print -delete

