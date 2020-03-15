DPPROOT := generated/grain/dpp
CUDA_ROOT := /usr/local/cuda
CUSRC := cublas.d cudnn.d cuda_driver.d nvrtc.d cuda_runtime_api.d
CLSRC := cl.d
DPP := dpp/bin/d++

.PHONY: all clean cuda cl

all: cuda cl

clean:
	rm -rf $(DPPROOT) *.d

cuda: $(addprefix $(DPPROOT)/,$(CUSRC))

cl: $(addprefix $(DPPROOT)/,$(CLSRC))

dpp/bin/d++:
	git clone https://github.com/atilaneves/dpp
	cd dpp; git checkout ffec304fa963a75c12bf96b500b407a747415b1b; dub build -b=release

$(DPPROOT)/%.d: %.dpp dpp/bin/d++
	mkdir -p $(DPPROOT)
	echo "module grain.dpp.$(basename $<);" > tmp.$(<)
	cat $< >> tmp.$(<)
	$(DPP) --include-path $(CUDA_ROOT)/include --preprocess-only tmp.$(<)
	mv tmp.$(basename $<).d $@
	rm tmp.$(<)
