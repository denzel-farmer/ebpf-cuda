#pragma once

#include "AllocationHistory.h"
#include "SyncUtils.h"
#include "SymUtils.h"
#include "MemHistory.h"
#include <bpf/libbpf.h>

#include "CudaEvents.h"

#include "CudaTracerProbe.skel.h"

// 16 MB ring buffers
constexpr unsigned long probe_ringbuf_size = 1 << 24;

using namespace std;

enum class ProbeType { KPROBE, UPROBE };

enum class ProbeTarget { DEVICE_TRANSFER, CUDA_MEMCPY };

class ProbeManager;

struct ProgramInfo {
	struct bpf_program *prog;
	struct ring_buffer *ringbuf;
	std::vector<struct bpf_link *> link;
	ProbeTarget target_func;
	pid_t target_pid;
	ProbeManager *manager;
};

inline ProbeType ConvertProbeTargetToProbeType(ProbeTarget target)
{
	switch (target) {
	case ProbeTarget::DEVICE_TRANSFER:
		return ProbeType::KPROBE;
	case ProbeTarget::CUDA_MEMCPY:
		return ProbeType::UPROBE;
	default:
		throw std::invalid_argument("Unknown ProbeTarget");
	}
}

inline struct bpf_program *GetProgramFromSkeleton(struct CudaTracerProbe_bpf *skel,
						  ProbeTarget target)
{
	switch (target) {
	case ProbeTarget::CUDA_MEMCPY:
		return skel->progs.handle_cudaMemcpy;
	default:
		throw std::invalid_argument("Unknown ProbeTarget");
	}
}

inline const char *GetSymbolNameFromProbeTarget(ProbeTarget target)
{
	switch (target) {
	case ProbeTarget::DEVICE_TRANSFER:
		return "device_transfer";
	case ProbeTarget::CUDA_MEMCPY:
		return "cudaMemcpy";
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
        // Terminate the queue
        m_event_queue.terminate();
	}

	// Public API

	bool AttachAllProbes();

	bool AttachProbe(ProbeTarget target_func, pid_t target_pid);

	bool DetachProbe(ProbeTarget target_func);

    void Shutdown();

    private:
	void Cleanup();

	void StartPolling();

	void StopPolling();

	bool AnyProgramAttached();

	void PollThreadFunc();

	// Static callback for ring buffer events
	// We get a pointer to this manager as ctx, and data from the ring buffer.
	static int HandleEvent(void *ctx, void *data, size_t size);

	// Process an event into an AllocationEvent and enqueue it to the event queue
	void ProcessEvent(const void *data, size_t size, const ProgramInfo &info);

    private:
	CudaTracerProbe_bpf *m_skel;
	int m_epoll_fd;
	atomic<bool> m_poll_stop;
	thread m_poll_thread;
	unordered_map<ProbeTarget, ProgramInfo *> m_programs;
	ThreadSafeQueue<AllocationEvent> &m_event_queue;
};