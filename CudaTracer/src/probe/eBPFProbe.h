#pragma once

#include "EventProbe.h"
#include "AllocationHistory.h"
#include "SyncUtils.h"
#include "MemHistory.h"

struct CudaMemcpyEvent {
    void *source;
    void *destination;
    size_t size;
    cudaMemcpyKind direction;
    CudaProcessInfo processInfo;
};

// eBPF EventProbe implementation
class eBPFProbe : public EventProbe {
    public:
	eBPFProbe(ThreadSafeQueue<AllocationEvent> &queue)
		: EventProbe(queue)
	{
	}

    private:
	optional<AllocationEvent> PollEvent();

	bool LoadBPFProgram();
};