#pragma once

#include <set>
#include <thread>
#include <atomic>
#include <unordered_map>
#include "AllocationHistory.h"
#include "SyncUtils.h"
#include "SymUtils.h"
#include "MemHistory.h"
#include <bpf/libbpf.h>

#include "CudaEvents.h"
#include "Logger.h"

#include "CudaTracerProbe.skel.h"

// 16 MB ring buffers
constexpr unsigned long probe_ringbuf_size = 1 << 24;

using namespace std;

enum class ProbeType { KPROBE, UPROBE };

enum class ProbeTarget { 
	CUDA_HOST_ALLOC,
	MALLOC,
	CUDA_MEMCPY,
	PIN_PAGES,
	CUDA_FREE,
	CUDA_HOST_FREE,
	FREE
	};

class ProbeManager;

struct ProgramInfo {
	struct bpf_program *prog;
	struct ring_buffer *ringbuf;
	std::vector<struct bpf_link *> links;
	ProbeTarget target_func;
	pid_t target_pid;
	ProbeManager *manager;
};

inline ProbeType ConvertProbeTargetToProbeType(ProbeTarget target)
{
	switch (target) {
	case ProbeTarget::PIN_PAGES:
		return ProbeType::KPROBE;
	default:
		return ProbeType::UPROBE;
	}
}

inline struct bpf_program *GetProgramFromSkeleton(struct CudaTracerProbe_bpf *skel,
						  ProbeTarget target)
{
	switch (target) {
	case ProbeTarget::CUDA_MEMCPY:
		return skel->progs.handle_cudaMemcpy;
	case ProbeTarget::CUDA_HOST_ALLOC:
		return skel->progs.handle_cudaHostAlloc;
	case ProbeTarget::CUDA_HOST_FREE:
		return skel->progs.handle_cudaHostFree;
	case ProbeTarget::PIN_PAGES:
		return skel->progs.os_lock_user_pages;
	default:
		throw std::invalid_argument("Unknown ProbeTarget");
	}
}

inline struct bpf_map *GetRingBufferFromSkeleton(struct CudaTracerProbe_bpf *skel,
						 ProbeTarget target)
{
	switch (target) {
	case ProbeTarget::CUDA_MEMCPY:
		return skel->maps.rb_cuda_memcpy;
	case ProbeTarget::CUDA_HOST_ALLOC:
		return skel->maps.rb_cuda_host_alloc;
	case ProbeTarget::CUDA_HOST_FREE:
		return skel->maps.rb_cuda_host_free;
	case ProbeTarget::PIN_PAGES:
		return skel->maps.rb_lock_user_pages;
	default:
		throw std::invalid_argument("Unknown ProbeTarget");
	}
}

inline const char *GetSymbolNameFromProbeTarget(ProbeTarget target)
{
	switch (target) {
	case ProbeTarget::CUDA_HOST_ALLOC:
		return "cudaHostAlloc";
	case ProbeTarget::CUDA_MEMCPY:
		return "cudaMemcpy";
	case ProbeTarget::PIN_PAGES:
		return "os_lock_user_pages";
	case ProbeTarget::CUDA_HOST_FREE:
		return "cudaFreeHost";
	case ProbeTarget::CUDA_FREE:
		return "cudaFree";
	default:
		throw std::invalid_argument("Unknown ProbeTarget");
	}
}

class ProbeManager {
    public:
	ProbeManager(ThreadSafeQueue<AllocationEvent> &queue);

	~ProbeManager()
	{
	    Shutdown();
	}

	// Public API

	bool AttachAllProbes();

	bool AttachProbe(ProbeTarget target_func, pid_t target_pid);
	bool AttachKprobe(ProbeTarget target_func, pid_t target_pid);
	bool AttachUprobe(ProbeTarget target_func, pid_t target_pid);

	bool DetachProbe(ProbeTarget target_func);

    void Shutdown();

    // Process an event into an AllocationEvent and enqueue it to the event queue
    // TODO find a way to make this static and private? Called by HandleEvent
	void ProcessEvent(const void *data, size_t size, const ProgramInfo *info);

    private:

	optional<AllocationEvent> ParseMemcpyEvent(const CudaMemcpyEvent *event, const ProgramInfo *info);
	AllocationEvent ParseAllocEvent(const GenericAllocEvent *event, const ProgramInfo *info);
	AllocationEvent ParseFreeEvent(const GenericFreeEvent *event, const ProgramInfo *info);
	AllocationEvent ParsePinPagesEvent(const CudaPinPagesEvent *event, const ProgramInfo *info);

	size_t GetUpdateCallNo(unsigned long return_address);
	void Cleanup();
    void DestroyInfo(ProgramInfo *info);

	void StartPolling();

	void StopPolling();

	bool AnyProgramAttached();

	void PollThreadFunc();


    private:
	CudaTracerProbe_bpf *m_skel;
	int m_epoll_fd;
	atomic<bool> m_poll_stop;
	thread m_poll_thread;
	unordered_map<unsigned long, size_t> m_call_no_map;
	unordered_map<ProbeTarget, ProgramInfo *> m_programs;
	ThreadSafeQueue<AllocationEvent> &m_event_queue;
};

// Static callback for ring buffer events
// We get a pointer to this manager as ctx, and data from the ring buffer.
static int HandleEvent(void *ctx, void *data, size_t size)
{
	// // Global log
	globalLogger.log_debug("Handling event");

	ProgramInfo *info = static_cast<ProgramInfo *>(ctx);
	// Now we know which program this event came from by info->target_func.
	ProbeManager *manager = info->manager;
	manager->ProcessEvent(data, size, info);
	return 0;
}
