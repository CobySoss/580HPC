#
# Makefile
#


CUCSRSRC=spmm_csr_driver.cu

CPPSRC=mm_helper.cpp

EXECCSR=spmm_csr_driver

OBJSCSR=$(CUCSRSRC:.cu=.o)
OBJSCSR+=$(CPPSRC:.cpp=.o)

NVCCFLAGS=-arch=sm_60 -O3 -lgomp

CC=nvcc
HCC=gcc

all: $(EXECCSR) $(EXECCSC) $(EXECOPT) 
	@echo "Change NVCCFLAGS to set the correct arch version"
	@echo "Change line the above line in Makefile to \"all: \$$(EXECSR) \$$(EXECSC)\" to build both CSR and CSC versions"
	@echo "Change line the above line in Makefile to \"all: \$$(EXECSR) \$$(EXECSC) \$$(EXEOPT)\" to build both CSR, CSC and OPT versions"

$(EXECCSR): $(OBJSCSR)
	$(CC) $(NVCCFLAGS) $^ -o $@

%.o : %.cu
	$(CC)  $(NVCCFLAGS) -c $< -o $@

%.o : %.cpp
	$(HCC) -O3 -c $< -o $@ -std=c++11

clean:
	rm -f $(EXECCSR) $(OBJSCSR) 


# vim:ft=make
#

