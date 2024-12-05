// cudamem.cpp
#include <argp.h>
#include <bpf/libbpf.h>
#include <fmt/core.h>
#include <cstdio>
#include <set>
#include <chrono>
#include <vector>
#include <signal.h>
#include <iostream>

#include "SymUtils.h"
#include "Guard.h"
#include "MemoryHeatmap.h"

#include "cudamem.skel.h"
#include "cudamem.h"

using namespace std;

static volatile sig_atomic_t exiting = 0;

static const int64_t RINGBUF_MAX_ENTRIES = 64 * 1024 * 1024;

using namespace std;

static const string kCudaMemcpyName = "cudaMemcpy";

void handle_sigint(int /* sig */)
{
	exiting = 1;
}

struct EventHandlerCtx {
	SymUtils *symUtils;
	MemoryHeatmap *HostHeatmap;
	MemoryHeatmap *DeviceHeatmap;
};

// Called each time event polled from ring buffer
static int handle_event(void *ctx, void *data, size_t /* data_sz */)
{
	auto *transferEvent = static_cast<CudaTransferEvent *>(data);
	auto *context = static_cast<EventHandlerCtx *>(ctx);

	unsigned long src_addr = reinterpret_cast<unsigned long>(transferEvent->source);
	unsigned long dst_addr = reinterpret_cast<unsigned long>(transferEvent->destination);

	switch (transferEvent->direction) {
	case cudaMemcpyKind::cudaMemcpyHostToHost:
		context->HostHeatmap->RecordAccess(src_addr, transferEvent->size);
		context->HostHeatmap->RecordAccess(dst_addr, transferEvent->size);
		break;
	case cudaMemcpyKind::cudaMemcpyHostToDevice:
		context->HostHeatmap->RecordAccess(src_addr, transferEvent->size);
		context->DeviceHeatmap->RecordAccess(dst_addr, transferEvent->size);
		break;
	case cudaMemcpyKind::cudaMemcpyDeviceToHost:
		context->DeviceHeatmap->RecordAccess(src_addr, transferEvent->size);
		context->HostHeatmap->RecordAccess(dst_addr, transferEvent->size);
		break;
	case cudaMemcpyKind::cudaMemcpyDeviceToDevice:
		context->DeviceHeatmap->RecordAccess(src_addr, transferEvent->size);
		context->DeviceHeatmap->RecordAccess(dst_addr, transferEvent->size);
		break;
	case cudaMemcpyKind::cudaMemcpyDefault:
		// TODO handle unified memory
		break;
	}

	cout << "cudaMemcpy called: dst=" << transferEvent->destination
	     << ", src=" << transferEvent->source
	     << ", count=" << transferEvent->size
	     << ", kind=" << transferEvent->direction
	     << endl;
	return 0;
}

int main(int argc, char **argv)
{
	struct cudamem_bpf *skel = nullptr;
	int err = 0;
	pid_t target_pid;

	if (argc != 2) {
		std::cerr << "Usage: " << argv[0] << " <PID>" << std::endl;
		return 1;
	}

	target_pid = std::atoi(argv[1]);
	if (target_pid <= 0) {
		std::cerr << "Invalid PID: " << argv[1] << std::endl;
		return 1;
	}

	SymUtils symUtils(target_pid);
	
	// Set up memory heatmaps
	MemoryHeatmap HostHeatmap;
	MemoryHeatmap DeviceHeatmap;

	EventHandlerCtx eventHandlerCtx = {&symUtils, &HostHeatmap, &DeviceHeatmap};

	vector<bpf_link *> links;
	struct ring_buffer *rb = nullptr;

	// Open and load the BPF program
	skel = cudamem_bpf__open_and_load();
	if (!skel) {
		std::cerr << "Failed to open and load BPF skeleton" << std::endl;
		return 1;
	}

	auto guard = Guard([&] {
		cudamem_bpf__destroy(skel);
		for (auto link : links) {
			bpf_link__destroy(link);
		}
		if (rb) {
			ring_buffer__free(rb);
		}
	});

	auto offsets = symUtils.findSymbolOffsets(kCudaMemcpyName);
	if (offsets.empty()) {
		fmt::print(stderr, "Failed to find symbol {}\n", kCudaMemcpyName);
		return -1;
	}

	for (auto &offset : offsets) {
		auto link = bpf_program__attach_uprobe(skel->progs.handle_cudaMemcpy,
						       false /* retprobe */, target_pid,
						       offset.first.c_str(), offset.second);
		if (link) {
			links.emplace_back(link);
		}
	}

	// Set up ring buffer
	rb = ring_buffer__new(bpf_map__fd(skel->maps.ringbuf), handle_event, &eventHandlerCtx, nullptr);
	if (!rb) {
		std::cerr << "Failed to create ring buffer" << std::endl;
		return 1;
	}

	signal(SIGINT, handle_sigint);

	// Poll ring buffer
	while (!exiting) {
		err = ring_buffer__poll(rb, 100 /* timeout, ms */);
		if (err == -EINTR) {
			break;
		} else if (err < 0) {
			std::cerr << "Error polling ring buffer: " << err << std::endl;
			break;
		}
	}

	if (err == 0 || err == -EINTR){
		cout << "Host Memory Heatmap:" << endl;
		HostHeatmap.PrintHeatmap(32);
		cout << "Device Memory Heatmap:" << endl;
		DeviceHeatmap.PrintHeatmap(32);
	}
	return -err;
}
