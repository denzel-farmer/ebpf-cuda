
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <unistd.h>
#include <sys/epoll.h>
#include <atomic>
#include <thread>
#include <iostream>
#include <cassert>
#include <unordered_map>
#include <string>
#include <vector>
#include <stdexcept>
#include <fmt/core.h>

#include "Logger.h"
#include "SymUtils.h"
#include "ProbeManager.h"
#include "CudaEvents.h"

#include "CudaTracerProbe.skel.h"

using namespace std;

ProbeManager::ProbeManager(ThreadSafeQueue<AllocationEvent> &queue)
	: m_skel(nullptr)
	, m_epoll_fd(-1)
	, m_poll_stop(false)
	, m_event_queue(queue)
{
	m_skel = CudaTracerProbe_bpf__open_and_load();
	if (!m_skel) {
		globalLogger.log_error("Failed to open and load BPF skeleton");
		throw runtime_error("CudaTracerProbe__open_and_load failed");
	}

	// Create epoll for all ringbuffers
	m_epoll_fd = epoll_create1(EPOLL_CLOEXEC);
	if (m_epoll_fd < 0) {
		globalLogger.log_error("Failed to create epoll");
		CudaTracerProbe_bpf__destroy(m_skel);
		throw runtime_error("epoll_create1 failed");
	}
}

bool ProbeManager::AttachAllProbes()
{
	globalLogger.log_info("Attaching all probes (not implemented)");

	return false;
}

bool ProbeManager::AttachProbe(ProbeTarget target_func, pid_t target_pid)
{
	// TODO better lifetime management
	ProgramInfo *info = new ProgramInfo();
	info->target_func = target_func;
	info->target_pid = target_pid;
	info->manager = this; // So handle_event can access this object
	const char *target_sym_name = GetSymbolNameFromProbeTarget(target_func);

	// Log pid + target_func
	globalLogger.log_info("Attaching probe for pid: " + to_string(target_pid) +
			      " target: " + string(target_sym_name));

	globalLogger.log_info("Getting program\n");
	info->prog = GetProgramFromSkeleton(m_skel, target_func);
	if (!info->prog) {
		globalLogger.log_error("Failed to get program for target: " +
				       to_string(static_cast<int>(target_func)));
		DestroyInfo(info);
		return false;
	}

	// Log the program being attached
	globalLogger.log_info("Finding symbols");

	SymUtils symUtils(info->target_pid);
	auto offsets = symUtils.findSymbolOffsets(target_sym_name);
	if (offsets.empty()) {
		globalLogger.log_error("Failed to find symbol: " + string(target_sym_name));
		DestroyInfo(info);
		return false;
	}

	for (auto &offset : offsets) {
		// Log offset + target_sym_name
		// cerr << "Attaching probe at symbol: " << offset.first.c_str() << endl;
		// fmt::print(stderr, "Attaching probe at 0x{:x}\n", (uintptr_t) offset.second);
		// fmt::print(stderr, "skel->progs.handle_cudaMemcpy: 0x{:x}\n", (uintptr_t) m_skel->progs.handle_cudaMemcpy);
		// fmt::print(stderr, "Attaching to PID: {}\n", target_pid);
		// fmt::print(stderr, "skel->progs.handle_cudaMemcpy: 0x{:x}\n", (uintptr_t) info->prog);
		// fmt::print(stderr, "Attaching to PID: {}\n", info->target_pid);
		bpf_link *link = bpf_program__attach_uprobe(info->prog, false /* retprobe */,
										info->target_pid, offset.first.c_str(),
										offset.second);
		cerr << "After attaching probe" << endl;
		if (link) {
			info->links.emplace_back(link);
		} else {
			globalLogger.log_error("Failed to attach uprobe at offset: " + to_string(offset.second) + " errno: " + to_string(errno));
			DestroyInfo(info);
			return false;
		}
	}

	globalLogger.log_info("Creating ring buffer\n");
	info->ringbuf =
		ring_buffer__new(bpf_map__fd(m_skel->maps.ringbuf), &HandleEvent, info, nullptr);
	if (!info->ringbuf) {
		globalLogger.log_error("Failed to create ring buffer for target: " +
				       string(target_sym_name));
		for (auto link : info->links) {
			bpf_link__destroy(link);
		}
		DestroyInfo(info);
		return false;
	}

	// Add this ring buffer's fd to epoll
	globalLogger.log_info("Adding ring buffer fd to epoll\n");
	int rb_fd = ring_buffer__epoll_fd(info->ringbuf);
	struct epoll_event ev = {};
	ev.events = EPOLLIN;
	ev.data.ptr = info->ringbuf; // identify which ringbuf triggered event
	if (epoll_ctl(m_epoll_fd, EPOLL_CTL_ADD, rb_fd, &ev) < 0) {
		globalLogger.log_error("Failed to add ringbuf fd to epoll for " + string(target_sym_name));
		DestroyInfo(info);
		return false;
	}

	globalLogger.log_info("Starting polling\n");
	// If this is the first program attached, start polling
	if (!m_poll_thread.joinable()) {
		StartPolling();
	}

	m_programs[target_func] = info;

	return true;
}

bool ProbeManager::DetachProbe(ProbeTarget target_func)
{
	// Log
	globalLogger.log_info("Detaching probe for target: " +
			      string(GetSymbolNameFromProbeTarget(target_func)));

	auto it = m_programs.find(target_func);
	if (it == m_programs.end()) {
		globalLogger.log_error("Program not found: " +
				       string(GetSymbolNameFromProbeTarget(target_func)));
		return false;
	}

	ProgramInfo *info = it->second;
	if (info->links.empty()) {
		globalLogger.log_error("Program not attached: " +
				       string(GetSymbolNameFromProbeTarget(target_func)));
		return false;
	}

	// Remove from epoll
	// TODDO do this in Cleanup()? Or DestroyInfo()?
	int rb_fd = ring_buffer__epoll_fd(info->ringbuf);
	epoll_ctl(m_epoll_fd, EPOLL_CTL_DEL, rb_fd, nullptr);

	DestroyInfo(info);
	
	// If no more programs attached, stop polling
	if (!AnyProgramAttached()) {
		StopPolling();
	}

	m_programs.erase(it);

	return true;
}

void ProbeManager::Shutdown()
{
	StopPolling();
	Cleanup();
}

void ProbeManager::Cleanup()
{
	for (auto &pair : m_programs) {
		ProgramInfo *info = pair.second;
		DestroyInfo(info);
	}
	m_programs.clear();

	if (m_epoll_fd >= 0) {
		close(m_epoll_fd);
		m_epoll_fd = -1;
	}

	if (m_skel) {
		CudaTracerProbe_bpf__destroy(m_skel);
		m_skel = nullptr;
	}

	assert(m_programs.empty());
	// Log
	globalLogger.log_info("Cleanup complete");
}

void ProbeManager::DestroyInfo(ProgramInfo *info)
{
	// For link in info.links, destroy and remove
	for (auto link : info->links) {
		bpf_link__destroy(link);
	}

	info->links.clear();

	// Free ring buffer
	if (info->ringbuf)
		ring_buffer__free(info->ringbuf);
	info->ringbuf = nullptr;

	delete info;
}

void ProbeManager::StartPolling()
{
	// Logging
	globalLogger.log_info("Starting polling");

	m_poll_stop.store(false);
	m_poll_thread = thread([this]() { PollThreadFunc(); });
}

void ProbeManager::StopPolling()
{
	// Logging
	globalLogger.log_info("Stopping polling");

	if (m_poll_thread.joinable()) {
		m_poll_stop.store(true);
		// epoll_wait will time out or get interrupted if needed
		// If needed, we could write to a pipe or use eventfd to wake epoll.
		m_poll_thread.join();
	}
}

bool ProbeManager::AnyProgramAttached()
{
	for (auto &pair : m_programs) {
		if (!pair.second->links.empty())
			return true;
	}
	return false;
}


void ProbeManager::PollThreadFunc()
{
	const int MAX_EVENTS = 10; // TODO make expression
	struct epoll_event events[MAX_EVENTS];

	while (!m_poll_stop.load()) {
		int nfds = epoll_wait(m_epoll_fd, events, MAX_EVENTS, 1000);
		if (nfds < 0) {
			if (errno == EINTR)
				continue;
			globalLogger.log_error("epoll_wait error");
			break;
		}

		if (nfds == 0) {
			// timeout - check stop condition
			continue;
		}

		for (int i = 0; i < nfds; i++) {
			if (m_poll_stop.load())
				break;

			// events[i].data.ptr points to the ring_buffer instance
			struct ring_buffer *rb = (struct ring_buffer *)events[i].data.ptr;
			// Consume data from this ring buffer
			ring_buffer__consume(rb);
		}
	}
}


// Process an event into an AllocationEvent and enqueue it to the event queue
void ProbeManager::ProcessEvent(const void *data, size_t size, const ProgramInfo *info)
{
	// TODO different kinds of events
	CudaMemcpyEvent *evt = (CudaMemcpyEvent *)data;
	unsigned long start;	

	switch (evt->direction) {
		case cudaMemcpyHostToDevice:
			start = evt->source;
			break;
		case cudaMemcpyDeviceToHost:
			start = evt->destination;
			break;
		default:
			globalLogger.log_error("Unknown CudaMemcpyKind: " + to_string(evt->destination));
			return;
	}

	globalLogger.log_info("Event return address: " + to_string(evt->return_address));

	AllocationEvent event(start, evt->timestamp, evt->size, evt->return_address, EventType::DEVICE_TRANSFER);

	// Global log including event details, using AllocationEvent's to_string method
	globalLogger.log_info("[ProbeManager->ProcessEvent] Event: " + event.ToString());
	m_event_queue.enqueue(event);
}
