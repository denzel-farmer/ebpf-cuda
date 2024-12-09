# 1. Repository Structure 
The repository top-level contains submodule dependencies (blazesym, bpftool, libbpf, and vmlinux)
a `docs` folder, and `CudaTracer`, the main folder containing our source code.

Within `CudaTracer`, there is a `CudaTracer/src` folder with the CPP code for the TracerAgent (in the `CudaTracer/src/agent` folder),
the eBPF probes (`CudaTracer/src/probe`), some common utilities (`CudaTracer/src/common`), and code for the host allocator (`CudaTracer/src/host-allocator`). 

There is also a `CudaTracer/test` folder, with a few selected programs useful for demonstration (including `TestTracerAgent.cpp`, the main
executable for tracing an arbitrary PID).

During builds, built objects from the submodule dependencies are placed in `CudaTracer/submodules_obj`, and built objects from our code
are placed in `CudaTracer/obj`.

# 2. Installation
The [eBPF-CUDA repository](https://github.com/denzelfarmer/ebpf-cuda) contains both
the Smart-Pinning Allocator and the TracingAgent framework, as well as submodule dependencies.
This section describes how to install each.

## 2.1 Compatibility
The current version of each tools has been tested only with the following software/hardware versions:
| Component       | Version                |
|-----------------|------------------------|
| Linux Flavor    | Debian 11              |
| Kernel          | 5.10.226-1 (2024-10-3) |
| NVIDIA Driver   | 550.90.07              |
| CUDA            | 12.4                   |
| GPU             | NVIDIA L4              |

## 2.2 Installing Dependencies 
## 2.2.1 CUDA Dependencies
This project requires a CUDA GPU, and requires the **open-source NVIDIA kernel module driver**. Linux installation instructions can be found [on nvidia.com](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html).

In addition, the CUDA toolkit is required, with installation instructions [also available from NVIDIA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#).

## 2.2.2 APT Dependencies 
In addition to the CUDA toolkit, NVIDIA drivers, and dependencies already built into the Debian 11 disk image, the following 
should be installed from APT:

- `boost` - A C++ library with many useful, optimized features
- `libelf` - A C++ library for managing compiled objects 
- `clang` - The compiler used to compile probes 
- `fmt` - A C++ library for safe c-style formatting

These can all be installed with the following command:
```bash
sudo apt-get update && sudo apt-get install -y libboost-all-dev libelf-dev  libfmt-dev clang
```

To use `blazesym`, we will also need to install `cargo`, the rust package manager. The 
easiest way to do this is to [install rust via rustup](https://doc.rust-lang.org/cargo/getting-started/installation.html):
```bash
curl https://sh.rustup.rs -sSf | sh
```
and configure `env` with 
```bash
 . ~/.cargo/env
 ```

### 2.2.2 Downloading Submodules 

The repository contains a number of  submodule dependencies:
- `libbpf` provides a C++ library for managing eBPF probes
- `bpftool` is a tool built on `libbpf` that provides further commands for interacting with probes
- `blazesym` helps inspect symbols of ELF binaries, for the purpose of installing uprobes 
- `vmlinux.h` provides additional required header files for various architectures

To clone all of the linked submodules, use `git submodule`:

```bash
# Initialize and update all submodules recursively
git submodule update --init --recursive
```

# 3. Compilation and Makefile

With the above dependencies satisfied, all default test binaries can be compiled using 
the Makefile in `CudaTracer`:
```bash
cd CudaTracer
make all
```

The `all` target will first compile all the submodule dependencies, then the selected
test examples from the `CudaTracer/test` folder and any source files from `CudaTracer/src`
required. 

The `clean` target will remove the `CudaTracer/obj` folder and skeleton header,
i.e. all the objects built from code in our project.

The `cleanall` target, on the other hand, runs the clean target but also
removes built submodule objects. 

**NOTE:** The Makefile is a bit buggy when it comes to detecting dependencies, so you
may be best off running `make clean` (not `cleanall`) and `make all` again if it appears broken. 

# 4. Demonstration Executables
The targets built by `make all` are the selected demo executables that show the tracing
framework and allocator working properly. 

**NOTOE:** All executables should be run with root permission (i.e. via sudo)

## 4.1 TestTracerAgent and simple_cuda_memcpy
The `TestTracerAgent` executable wraps the `TracerAgent` framework in an executable, 
and takes in as an argument a single PID of a target program to trace. It then 
attaches probes to that function, and dumps runlogs to `logs.txt` and a history
to `history_dump.json`:

```bash
sudo ./TestTracerAgent <pid>
```

A good demonstration target for this is the `simple_cuda_memcpy` example, which does an
extremely simple host allocation, transfer to device, and free. It also prints out 
its PID, and awaits user input before running. 

```bash
sudo ./simple_cuda_memcpy
```

## 4.2 cuda_benchmark_application

The `cuda_benchmark_application` executable is the primary application we used to do
basic benchmarking for our custom pinning allocator. It takes in four parameters,
`single_iters`, `multi_iters`, `multi_transfers`, and `pre-pinned size`. 

The application performs a test based on these parameters four times:
1. Using `malloc` and `free`, to demonstrate fully unpinned page allocation 
2. Using `cudaMallocHost` and `cudaMallocFree` to demonstrate fully pinned page allocation
3. Using our custom pinning allocator in `profile` mode, to trace the test with the `TracerAgent`
framework and build a history (which it saves)
4. Using our custom pinning allocator in `use` mode, applying the saved history to optimize
transfers and minimize pinning

The test itself is based on the parameters given. First, `pre-pinned size` bytes are allocated 
as pinned memory and written to. This allows benchmarking in scenarios where pinned memory is restricted. 

Then, `single_iters` allocations of hardcoded sizes are made, followed by `multi_iters` allocations. The
`single_iters` allocations are transfered from host to device once, while the `multi_iters` are transfered
back and forth `multi_transfers` number of times. 

For each test, the runtime performance and pinned page count is recorded. This allows benchmarking the 
performance of the allocator in various scenarios, with different degrees of hotspots and coldspots, while
also comparing to the best and worst case scenarios. 


# 5. TraceAgent Framework 

The TraceAgent framework is simple, and can be either used via the `TestTraceAgent` executable described 
in section 4.1 or as a class by including the `TraceAgent.h` header. 

The core components of the `TracerAgent` API are outlined below:

- The constructor takes a PID to trace, or the current process PID if none is provided.
```C++
 TracerAgent()
 TracerAgent(Pid_t pid) 
 ```
 
 - The `StartAgentAsync` and `StopAgent` allow starting and stoping tracing (while mantaining memory history)
 without destroying the `TracerAgent` object. `StartAgentAsync` attaches relevant probes and launches an event
 processing thread, while `StopAgent` detaches probes and stops the processing thread.
 ```C++
bool StartAgentAsync();
void StopAgent();
```

- The `GetMemHistory` accessor allows getting a reference to the current memory history object, which is valid
for the lifetime of the `TracerAgent`. The retrieved object is safe under concurrent accesses and event recording
(although for this the `HandleEvent` method is more appropriate), if inserting custom events is required. This accessor
is mostly for flexibility, using the TracerAgent for something not hotspot-related.
 ```C++
MemHistory& GetMemHistory();
```

- The hotspot and coldspot accessor methods allow quickly accessing high/low transfer count allocations. The first two take the
nth highest/lowest transfer count allocations, and return their information. The second two return information about 
all allocations above/below a provided threshold.
 ```C++
vector<Allocation> GetHotspots(size_t num) const;
vector<Allocation> GetColdspots(size_t num) const;

vector<Allocation> GetHotspotsThreshold(unsigned long min_transfers) const;
vector<Allocation> GetColdspotsThreshold(unsigned long max_transfers) const;
```

- The `DumpHistory` method allows dumping the current memory history to a file for persistent storage
or examination. It takes in a `DumpFormat` indicating whether to dump as human-readable JSON or compressed
binary, and a `verbose` flag indicating whether or not to included per-event information for each allocation,
or just overall staticstics like `transfer_count`.
```C++
enum class DumpFormat {
    JSON,
    BINARY
};

void DumpHistory(const char *filename, DumpFormat format);
void DumpHistory(const char *filename, DumpFormat format, bool verbose);

```

- The `HandleEvent` method is mainly used by the processing thread to submit events, but can also be used 
to submit custom events (for example, when the custom allocator must indicate that it is doing allocation)
```C++
void HandleEvent(AllocationEvent event);
```

# 6. Smart-Pinning Allocator

The custom allocator is meant to be a drop-in replacement for calls like `cudaHostAlloc` and `malloc`. Using
it requires including the `CustomAllocatorManager.h` header. This header delcares a global variable, which 
can be used to allocate and deallocate memory. Alternatively, there are two equivalently-named `extern` functions
that wrap access to the global variable.

```C++
void* allocate_memory(size_t size);
void deallocate_memory(void* ptr, size_t size);
```

The other core allocator function is `initialze`, which allows reseting the global allocator. This both
resets internal call-number counts, stops the built-in `TracerAgent` if it is tracing, and enters either
'profile' or 'use' mode. If 'profile' mode is selected, the allocator clears TracerAgent history and restarts
it, collecting a memory history until the program ends or `initialize` is called again. In this mode, memory
defaults to being allocated as pinned memory. 

The 'use' mode loads in the memory history currently collected by the tracer, and decides on which 
allocations in future runs to pin. 
```C++
void initialize(const std::string& mode);
```