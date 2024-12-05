#pragma once

#include "EventProbe.h"
#include "AllocationHistory.h"
#include "SyncUtils.h"
#include "MemHistory.h"

// 16 MB ring buffers
constexpr unsigned long probe_ringbuf_size = 1 << 24;


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