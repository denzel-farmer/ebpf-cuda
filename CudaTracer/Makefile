CLANG ?= clang
ROOT := $(abspath ..)
OUTPUT := $(ROOT)/CudaTracer/obj
SUBMOD_OUTPUT := $(ROOT)/CudaTracer/submodules_obj
SRCDIR = src
TESTDIR = test

LIBBPF_SRC := $(abspath $(ROOT)/libbpf/src)
BPFTOOL_SRC := $(abspath $(ROOT)/bpftool/src)
LIBBPF_OBJ := $(abspath $(SUBMOD_OUTPUT)/libbpf.a)
BPFTOOL_OUTPUT ?= $(abspath $(SUBMOD_OUTPUT)/bpftool)
BPFTOOL ?= $(BPFTOOL_OUTPUT)/bootstrap/bpftool

LIBBLAZESYM_SRC := $(abspath ../blazesym/)
LIBBLAZESYM_INC := $(abspath $(LIBBLAZESYM_SRC)/capi/include)
LIBBLAZESYM_OBJ := $(abspath $(SUBMOD_OUTPUT)/libblazesym_c.a)

ARCH ?= $(shell uname -m | sed 's/x86_64/x86/' \
			 | sed 's/arm.*/arm/' \
			 | sed 's/aarch64/arm64/' \
			 | sed 's/ppc64le/powerpc/' \
			 | sed 's/mips.*/mips/' \
			 | sed 's/riscv64/riscv/' \
			 | sed 's/loongarch64/loongarch/')

VMLINUX := $(ROOT)/vmlinux.h/include/$(ARCH)/vmlinux.h

CXX = g++
CXXFLAGS = -Wall -g -std=c++20
LDFLAGS = -lpthread -lfmt -lelf -L$(dir $(LIBBLAZESYM_OBJ)) -lblazesym_c -L$(dir $(LIBBPF_OBJ)) -lbpf -lz -ldl -L/usr/local/cuda/lib64 -lcudart
NVCC_FLAGS = -g -O2 -std=c++17

INCLUDE_DIRS = $(shell find $(SRCDIR) -type d)
INCLUDES := $(addprefix -I,$(INCLUDE_DIRS)) -I$(OUTPUT) -I$(SUBMOD_OUTPUT) -I$(ROOT)/libbpf/include/uapi -I$(dir $(VMLINUX)) -I$(LIBBLAZESYM_INC) -I/usr/local/cuda/include

# All main source files
SRC = $(shell find $(SRCDIR) -type f -name '*.cpp')
SRC_OBJ = $(patsubst $(SRCDIR)/%.cpp, $(OUTPUT)/%.o, $(SRC))
SRC_DEPS = $(SRC_OBJ:.o=.d)

# Hardcoded test executables from .cpp files
TESTS := TestTracerAgent TestMemHistory
TEST_SRC = $(addprefix $(TESTDIR)/,$(patsubst %,%.cpp,$(TESTS)))
TEST_OBJ = $(patsubst $(TESTDIR)/%.cpp, $(OUTPUT)/test_%.o, $(TEST_SRC))
TEST_DEPS = $(TEST_OBJ:.o=.d)

# Hardcoded CUDA test executables from .cu files
TESTS_CU := cuda_benchmark_application simple_cuda_memcpy matrix_mult_smart matrix_mult_pinned matrix_mult_unpinned
TEST_CU_SRC = $(addprefix $(TESTDIR)/,$(patsubst %,%.cu,$(TESTS_CU)))
TEST_CU_OBJ = $(patsubst $(TESTDIR)/%.cu, $(OUTPUT)/test_%.o, $(TEST_CU_SRC))
TEST_CU_DEPS = $(TEST_CU_OBJ:.o=.d)

all: $(LIBBPF_OBJ) $(BPFTOOL) $(LIBBLAZESYM_OBJ) $(TESTS) $(TESTS_CU) 

##############################
# Main source compilation
##############################
$(OUTPUT)/%.o: $(SRCDIR)/%.cpp $(BPFTOOL) $(LIBBPF_OBJ) $(SRCDIR)/probe/CudaTracerProbe.skel.h $(LIBBLAZESYM_OBJ) $(SRC_DEPS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

$(OUTPUT)/%.d: $(SRCDIR)/%.cpp $(BPFTOOL) $(LIBBPF_OBJ) $(SRCDIR)/probe/CudaTracerProbe.skel.h $(LIBBLAZESYM_OBJ)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -M $< > $@

##############################
# C++ Test compilation
##############################
$(OUTPUT)/test_%.o: $(TESTDIR)/%.cpp $(LIBBPF_OBJ) $(SRCDIR)/probe/CudaTracerProbe.skel.h $(TEST_DEPS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

$(OUTPUT)/test_%.d: $(TESTDIR)/%.cpp $(LIBBPF_OBJ)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -M $< > $@

$(TESTS): %: $(OUTPUT)/test_%.o $(SRC_OBJ) $(LIBBPF_OBJ) $(SRCDIR)/probe/CudaTracerProbe.skel.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

##############################
# CUDA Test compilation
##############################
$(OUTPUT)/test_%.o: $(TESTDIR)/%.cu $(SRCDIR)/probe/CudaTracerProbe.skel.h $(TEST_CU_DEPS)
	@mkdir -p $(dir $@)
	nvcc $(NVCC_FLAGS) $(INCLUDES) -c -o $@ $<

$(OUTPUT)/test_%.d: $(TESTDIR)/%.cu $(LIBBPF_OBJ)
	@mkdir -p $(dir $@)
	nvcc $(NVCC_FLAGS) $(INCLUDES) -M $< > $@

$(TESTS_CU): %: $(OUTPUT)/test_%.o $(SRC_OBJ) | $(SRCDIR)/probe/CudaTracerProbe.skel.h
	nvcc -o $@ $^ $(LDFLAGS)

##############################
# BPF code build rules
##############################
$(OUTPUT)/%.bpf.o: $(SRCDIR)/%.bpf.c $(LIBBPF_OBJ) $(wildcard %.h) $(VMLINUX) | $(OUTPUT) $(SUBMOD_OUTPUT) $(BPFTOOL)
	$(call msg,BPF,$@)
	@mkdir -p $(dir $@)
	$(Q)$(CLANG) -g -O2 -target bpf -D__TARGET_ARCH_$(ARCH) \
		     $(INCLUDES) $(CLANG_BPF_SYS_INCLUDES) \
		     -c $(filter %.c,$^) -o $(patsubst %.bpf.o,%.tmp.bpf.o,$@)
	$(Q)$(BPFTOOL) gen object $@ $(patsubst %.bpf.o,%.tmp.bpf.o,$@)

$(SRCDIR)/%.skel.h: $(OUTPUT)/%.bpf.o | $(OUTPUT) $(SUBMOD_OUTPUT) $(BPFTOOL)
	$(call msg,GEN-SKEL,$@)
	$(Q)$(BPFTOOL) gen skeleton $< > $@

##############################
# Tool build rules
##############################

$(LIBBPF_OBJ): $(wildcard $(LIBBPF_SRC)/*.[ch] $(LIBBPF_SRC)/Makefile) | $(SUBMOD_OUTPUT)/libbpf
	$(call msg,LIB,$@)
	$(Q)$(MAKE) -C $(LIBBPF_SRC) BUILD_STATIC_ONLY=1 \
		    OBJDIR=$(dir $@)/libbpf DESTDIR=$(dir $@) \
		    INCLUDEDIR= LIBDIR= UAPIDIR= \
		    install

$(BPFTOOL): | $(BPFTOOL_OUTPUT)
	$(call msg,BPFTOOL,$@)
	$(Q)$(MAKE) ARCH= CROSS_COMPILE= OUTPUT=$(BPFTOOL_OUTPUT)/ -C $(BPFTOOL_SRC) bootstrap

$(LIBBLAZESYM_SRC)/target/release/libblazesym_c.a:
	$(Q)cd $(LIBBLAZESYM_SRC) && $(CARGO) build --package=blazesym-c --release

$(LIBBLAZESYM_OBJ): $(LIBBLAZESYM_SRC)/target/release/libblazesym_c.a | $(SUBMOD_OUTPUT)
	$(call msg,LIB, $@)
	$(Q)sudo cp $(LIBBLAZESYM_SRC)/target/release/libblazesym_c.a $@

$(OUTPUT) $(SUBMOD_OUTPUT) $(SUBMOD_OUTPUT)/libbpf $(BPFTOOL_OUTPUT):
	$(call msg,MKDIR,$@)
	$(Q)mkdir -p $@

CARGO ?= $(shell which cargo)
CLANG_BPF_SYS_INCLUDES ?= $(shell $(CLANG) -v -E - </dev/null 2>&1 \
	| sed -n '/<...> search starts here:/,/End of search list./{ s| \(/.*\)|-idirafter \1|p }')

ifeq ($(V),1)
	Q =
	msg =
else
	Q = @
	msg = @printf '  %-8s %s%s\n' \
		      "$(1)" \
		      "$(patsubst $(abspath $(OUTPUT))/%,%,$(2))" \
		      "$(if $(3), $(3))";
	MAKEFLAGS += --no-print-directory
endif

define allow-override
  $(if $(or $(findstring environment,$(origin $(1))),\
            $(findstring command line,$(origin $(1)))),,\
    $(eval $(1) = $(2)))
endef

$(call allow-override,CC,$(CROSS_COMPILE)cc)
$(call allow-override,LD,$(CROSS_COMPILE)ld)


##############################
# Include dependencies (except for clean)
##############################
# ifneq ($(MAKECMDGOALS),clean)
# ifneq ($(MAKECMDGOALS),cleanall)
# -include $(SRC_DEPS) $(TEST_DEPS) $(TEST_CU_DEPS)
# endif
# endif

clean:
	rm -rf $(OUTPUT) $(TESTS) $(TESTS_CU)
	find $(SRCDIR) -name '*skel.h' -delete

cleanall: clean
	rm -rf $(SUBMOD_OUTPUT)

.PHONY: clean cleanall all


























# # Path variables

# ROOT := $(abspath ..)
# # Output for object files from our code and tests (not submodules)
# OUTPUT := $(ROOT)/CudaTracer/object
# # Output for object files from submodules
# SUBMOD_OUTPUT := $(ROOT)/CudaTracer/submodules_object

# SRCDIR = src
# TESTDIR = test

# # Paths for libbpf submodule
# LIBBPF_SRC := $(ROOT)/libbpf/src
# LIBBPF_OBJ := $(SUBMOD_OUTPUT)/libbpf.a

# # Paths for bpftool submodule
# BPFTOOL_SRC := $(ROOT)/bpftool/src
# BPFTOOL_OUTPUT ?= $(SUBMOD_OUTPUT)/bpftool
# BPFTOOL ?= $(BPFTOOL_OUTPUT)/bootstrap/bpftool

# # Paths for blazesym submodule
# LIBBLAZESYM_SRC := $(ROOT)/blazesym
# LIBBLAZESYM_INC := $(LIBBLAZESYM_SRC)/capi/include
# LIBBLAZESYM_OBJ := $(SUBMOD_OUTPUT)/libblazesym_c.a

# # Path for vmlinux.h
# ARCH ?= $(shell uname -m | sed 's/x86_64/x86/' \
# 			 | sed 's/arm.*/arm/' \
# 			 | sed 's/aarch64/arm64/' \
# 			 | sed 's/ppc64le/powerpc/' \
# 			 | sed 's/mips.*/mips/' \
# 			 | sed 's/riscv64/riscv/' \
# 			 | sed 's/loongarch64/loongarch/')
# VMLINUX := $(ROOT)/vmlinux.h/include/$(ARCH)/vmlinux.h


# # Compiler and linker flags
# CXX = g++
# NVCC = nvcc
# COMPILE_FLAGS = -O2 -g -std=c++20
# LDFLAGS = -lpthread -lfmt -lelf -L$(dir $(LIBBLAZESYM_OBJ)) -lblazesym_c -L$(dir $(LIBBPF_OBJ)) -lbpf -lz -ldl -L/usr/local/cuda/lib64 -lcudart

# INCLUDE_DIRS = $(shell find $(SRCDIR) -type d)
# INCLUDES := $(addprefix -I,$(INCLUDE_DIRS)) -I$(OUTPUT) -I$(SUBMOD_OUTPUT) -I$(ROOT)/libbpf/include/uapi -I$(dir $(VMLINUX)) -I$(LIBBLAZESYM_INC) -I/usr/local/cuda/include

# # Test Targets
# CXX_TARGETS = $(TESTDIR)/TestMemHistory.cpp $(TESTDIR)/TestTracerAgent.cpp
# CU_TARGETS = $(TESTDIR)/cuda_benchmark_application.cu $(TESTDIR)/simple_cuda_memcpy.cu

# # Source files
# SRC = $(shell find $(SRCDIR) -type f -name '*.cpp')
