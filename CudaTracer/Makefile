### Paths and variables

CLANG ?= clang
ROOT := $(abspath ..)
# OUTPUT := .output
OUTPUT := $(ROOT)/CudaTracer/obj
SUBMOD_OUTPUT := $(ROOT)/CudaTracer/submodules_obj
SRCDIR = src
TESTDIR = test
LIBBPF_SRC := $(abspath $(ROOT)/libbpf/src)
BPFTOOL_SRC := $(abspath $(ROOT)/bpftool/src)
# LIBBPF_SRC := $(abspath ../../../../libbpf/src)
# BPFTOOL_SRC := $(abspath ../../../../bpftool/src)
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
# VMLINUX := ../vmlinux.h/include/$(ARCH)/vmlinux.h
VMLINUX := $(ROOT)/vmlinux.h/include/$(ARCH)/vmlinux.h

CXX = g++
CXXFLAGS = -Wall -g -std=c++20
LDFLAGS = -lpthread -lfmt -lelf -L$(dir $(LIBBLAZESYM_OBJ)) -lblazesym_c -L$(dir $(LIBBPF_OBJ)) -lbpf -lz -ldl

# Use our own libbpf API headers and Linux UAPI headers distributed with
# libbpf to avoid dependency on system-wide headers, which could be missing or
# outdated
INCLUDE_DIRS = $(shell find $(SRCDIR) -type d)
INCLUDES := $(addprefix -I,$(INCLUDE_DIRS)) -I$(OUTPUT) -I$(SUBMOD_OUTPUT) -I$(ROOT)/libbpf/include/uapi -I$(dir $(VMLINUX)) -I$(LIBBLAZESYM_INC)
#CFLAGS := -g -Wall
# ALL_LDFLAGS := $(LDFLAGS) $(EXTRA_LDFLAGS)
# ALL_LDFLAGS += -lrt -ldl -lpthread -lm -lfmt 

# Find all source files under src
SRC = $(shell find $(SRCDIR) -type f -name '*.cpp')

# Find all test source files
TEST_SRC = $(wildcard $(TESTDIR)/*.cpp)

# Generate list of object files, mirroring directory structure
SRC_OBJ = $(patsubst $(SRCDIR)/%.cpp, $(OUTPUT)/%.o, $(SRC))

# Generate list of test object files
TEST_OBJ = $(patsubst $(TESTDIR)/%.cpp, $(OUTPUT)/test_%.o, $(TEST_SRC))

# Include directories under src for headers
# INCLUDE_DIRS = $(shell find $(SRCDIR) -type d)
# CXXFLAGS += $(addprefix -I,$(INCLUDE_DIRS))

# Generate list of test executables
TESTS = $(patsubst $(TESTDIR)/%.cpp, %, $(TEST_SRC))


## Executable/Object File Rules

all: $(TESTS) $(LIBBPF_OBJ) $(BPFTOOL) $(LIBBLAZESYM_OBJ)

# Rule to build each test executable
$(TESTS): %: $(OUTPUT)/test_%.o $(SRC_OBJ)
	$(CXX) -o $@ $^ $(LDFLAGS)

# Compile source files into object files
$(OUTPUT)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

# Compile test source files into object files
$(OUTPUT)/test_%.o: $(TESTDIR)/%.cpp
	@mkdir -p $(OUTPUT)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

# Specific dependencies between eBPFProbe object and BPF code
# TODO just make list of probes
$(OUTPUT)/probe/eBPFProbes.o: $(LIBBPF_OBJ) $(OUTPUT)/probe/CudaTracerProbe.bpf.o $(SRCDIR)/probe/CudaTracerProbe.skel.h

# Build BPF code (bpf.c -> bpf.o)
$(OUTPUT)/%.bpf.o: $(SRCDIR)/%.bpf.c $(LIBBPF_OBJ) $(wildcard %.h) $(VMLINUX) | $(OUTPUT) $(SUBMOD_OUTPUT) $(BPFTOOL)
	$(call msg,BPF,$@)
	@mkdir -p $(dir $@)
	$(Q)$(CLANG) -g -O2 -target bpf -D__TARGET_ARCH_$(ARCH)		      \
		     $(INCLUDES) $(CLANG_BPF_SYS_INCLUDES)		      \
		     -c $(filter %.c,$^) -o $(patsubst %.bpf.o,%.tmp.bpf.o,$@)
	$(Q)$(BPFTOOL) gen object $@ $(patsubst %.bpf.o,%.tmp.bpf.o,$@)

# Generate BPF skeletons
$(SRCDIR)/%.skel.h: $(OUTPUT)/%.bpf.o | $(OUTPUT) $(SUBMOD_OUTPUT) $(BPFTOOL)
	$(call msg,GEN-SKEL,$@)
	$(Q)$(BPFTOOL) gen skeleton $< > $@

# Build user-space code
$(patsubst %,$(OUTPUT)/%.o,$(APP)): %.o: %.skel.h

# From cudemem Makefile:
# # Convert cpp files to object files
# APP_OBJ_FILES := $(patsubst %.cpp,$(OUTPUT)/%.o,$(APP_CPP_FILES))

# $(OUTPUT)/%.o: %.cpp $(wildcard %.h) | $(OUTPUT)
# 	$(call msg,CXX,$@)
# 	$(Q)$(CXX) $(CXXFLAGS) $(INCLUDES) -c $(filter %.cpp,$^) -o $@



## Build submodule dependencies: libbpf, bpftool, libblazesym

# Directories

$(OUTPUT) $(SUBMOD_OUTPUT) $(SUBMOD_OUTPUT)/libbpf $(BPFTOOL_OUTPUT):
	$(call msg,MKDIR,$@)
	$(Q)mkdir -p $@

# $(SUBMOD_OUTPUT) $(SUBMOD_OUTPUT)/libbpf $(BPFTOOL_OUTPUT):
# 	$(call msg,MKDIR,$@)
# 	$(Q)mkdir -p $@

CARGO ?= $(shell which cargo)

# Get Clang's default includes on this system. We'll explicitly add these dirs
# to the includes list when compiling with `-target bpf` because otherwise some
# architecture-specific dirs will be "missing" on some architectures/distros -
# headers such as asm/types.h, asm/byteorder.h, asm/socket.h, asm/sockios.h,
# sys/cdefs.h etc. might be missing.
#
# Use '-idirafter': Don't interfere with include mechanics except where the
# build would have failed anyways.
CLANG_BPF_SYS_INCLUDES ?= $(shell $(CLANG) -v -E - </dev/null 2>&1 \
	| sed -n '/<...> search starts here:/,/End of search list./{ s| \(/.*\)|-idirafter \1|p }')

ifeq ($(V),1)
	Q =
	msg =
else
	Q = @
	msg = @printf '  %-8s %s%s\n'					\
		      "$(1)"						\
		      "$(patsubst $(abspath $(OUTPUT))/%,%,$(2))"	\
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


# Build libbpf
$(LIBBPF_OBJ): $(wildcard $(LIBBPF_SRC)/*.[ch] $(LIBBPF_SRC)/Makefile) | $(SUBMOD_OUTPUT)/libbpf
	$(call msg,LIB,$@)
	$(Q)$(MAKE) -C $(LIBBPF_SRC) BUILD_STATIC_ONLY=1		      \
		    OBJDIR=$(dir $@)/libbpf DESTDIR=$(dir $@)		      \
		    INCLUDEDIR= LIBDIR= UAPIDIR=			      \
		    install

# Build bpftool
$(BPFTOOL): | $(BPFTOOL_OUTPUT)
	$(call msg,BPFTOOL,$@)
	$(Q)$(MAKE) ARCH= CROSS_COMPILE= OUTPUT=$(BPFTOOL_OUTPUT)/ -C $(BPFTOOL_SRC) bootstrap

# Build libblazesym
$(LIBBLAZESYM_SRC)/target/release/libblazesym_c.a::
	$(Q)cd $(LIBBLAZESYM_SRC) && $(CARGO) build --package=blazesym-c --release

$(LIBBLAZESYM_OBJ): $(LIBBLAZESYM_SRC)/target/release/libblazesym_c.a | $(SUBMOD_OUTPUT)
	$(call msg,LIB, $@)
	$(Q)sudo cp $(LIBBLAZESYM_SRC)/target/release/libblazesym_c.a $@

# Clean files generated from our code
clean:
	rm -rf $(OUTPUT)

# Clean all files generated from our code and submodules
cleanall:
	rm -rf $(OUTPUT) $(SUBMOD_OUTPUT)
	find $(SRCDIR) -name '*skel.h' -delete
# TODO add cleanall vs clean

#all: $(OUTPUT)/probe/bpf/CudaMemcpyProbe.bpf.o
# Rules
.PHONY: clean all

# # Build all test executables
# all: $(OUTPUT)/probe/EventProbe.o
# # all: $(OUTPUT)/probe/bpf/CudaMemcpyProbe.bpf.o 