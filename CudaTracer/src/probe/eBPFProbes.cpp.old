#include "AllocationHistory.h"
#include "SyncUtils.h"
#include "MemHistory.h"
#include "Logger.h"
#include "EventProbe.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <optional>
#include <set>
#include <shared_mutex>
#include <bpf/libbpf.h>
#include <thread>
#include <iostream>

#include "eBPFProbes.h"
#include "Guard.h"
#include "SymUtils.h"
#include "Logger.h"

#include "CudaTraceProbe.skel.h"

// Constructor that opens the eBPF program and loads it
eBPFProbe::eBPFProbe(ThreadSafeQueue<AllocationEvent> &queue, pid_t target_pid)
	: EventProbe(queue), target_pid(target_pid) {
	
	globalLogger.log_info("eBPFProbe created, opening and loading program");
	program = CudaMemcpyProbe_bpf__open_and_load();
	if (!program) {
		globalLogger.log_error("Failed to open and load eBPF program");
	}
	globalLogger.log_info("Program loaded");
}

// Destructor that cleans up the eBPF program
eBPFProbe::~eBPFProbe() {
	
}

optional<AllocationEvent> eBPFProbe::PollEvent() {
        return AllocationEvent(0, 0, 0, EventType::DEVICE_TRANSFER);
}

bool eBPFProbe::LoadBPFProgram() {
    struct CudaMemcpyProbe_bpf *skel = nullptr;
	int err = 0;


	SymUtils symUtils(target_pid);
	
	vector<bpf_link *> links;
	struct ring_buffer *rb = nullptr;

	// Open and load the BPF program
	skel = CudaMemcpyProbe_bpf__open_and_load();
	if (!skel) {
		std::cerr << "Failed to open and load BPF skeleton" << std::endl;
		return 1;
	}

	auto guard = Guard([&] {
		CudaMemcpyProbe_bpf__destroy(skel);
		for (auto link : links) {
			bpf_link__destroy(link);
		}
		if (rb) {
			ring_buffer__free(rb);
		}
	});

    return true;
}