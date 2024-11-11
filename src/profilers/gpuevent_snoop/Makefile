# Define the compiler and the flags
CXX ?= g++
CFLAGS = -O2 -target bpf -c -g
USERSPACE_CFLAGS = -O2
USERSPACE_LINKER_FLAGS = -lbpf -lncurses

# Userspace programs
USERSPACE_SRC = GpuEventSnoop.cpp
USERSPACE_BIN = $(USERSPACE_SRC:.cpp=.out)
# Define the BPF program source and the output object file
BPF_SRC = gpuevent_snoop.bpf.c
BPF_OBJ = $(BPF_SRC:.c=.o)
BPF_HDR = $(BPF_SRC:bpf.c=skel.h)

# Get Clang's default includes on this system. We'll explicitly add these dirs
# to the includes list when compiling with `-target bpf` because otherwise some
# architecture-specific dirs will be "missing" on some architectures/distros -
# headers such as asm/types.h, asm/byteorder.h, asm/socket.h, asm/sockios.h,
# sys/cdefs.h etc. might be missing.
#
# Use '-idirafter': Don't interfere with include mechanics except where the
# build would have failed anyways.
CLANG_BPF_SYS_INCLUDES ?= $(shell clang -v -E - </dev/null 2>&1 \
	| sed -n '/<...> search starts here:/,/End of search list./{ s| \(/.*\)|-idirafter \1|p }')


all: $(BPF_OBJ) $(USERSPACE_BIN)

$(BPF_OBJ): $(BPF_SRC) vmlinux.h 
	clang $(CFLAGS) $(CLANG_BPF_SYS_INCLUDES) $(BPF_SRC) -o $(BPF_OBJ)

$(USERSPACE_BIN): $(USERSPACE_SRC) $(BPF_HDR)
	$(CXX) $(USERSPACE_CFLAGS) $(USERSPACE_SRC) -o $(USERSPACE_BIN) $(USERSPACE_LINKER_FLAGS)

$(BPF_HDR): $(BPF_OBJ)
	bpftool gen skeleton $(BPF_OBJ) > $(BPF_HDR)

vmlinux.h:
	bpftool btf dump file /sys/kernel/btf/vmlinux format c > vmlinux.h

clean:
	rm -f $(BPF_OBJ) $(USERSPACE_BIN) $(BPF_HDR)

cleanall: clean
	rm -f vmlinux.h

.PHONY: all clean
