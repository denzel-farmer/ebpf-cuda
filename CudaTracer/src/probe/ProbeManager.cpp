
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

bool ProbeManager::AttachAllPrograms()
{
	bool success = true;
	for (auto &pair : m_programs) {
		if (!AttachProgram(pair.first)) {
			success = false;
		}
	}
	return success;
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
			      " target: " + target_sym_name);

	info->prog = GetProgramFromSkeleton(m_skel, target_func);
	if (!info->prog) {
		globalLogger.log_error("Failed to get program for target: " +
				       to_string(static_cast<int>(target_func)));
		return false;
	}

	SymUtils symUtils(info->target_pid);
	auto offsets = symUtils.findSymbolOffsets(target_sym_name);
	if (offsets.empty()) {
		globalLogger.log_error("Failed to find symbol: " + target_sym_name);
		return false;
	}

	for (auto &offset : offsets) {
		// Log offset + target_sym_name
		globalLogger.log_info("Attaching probe at offset: " + to_string(offset.second) +
				      " target: " + target_sym_name);
		bpf_link *link ProgramInfo *info = new ProgramInfo();
		= bpf_program__attach_uprobe(info.prog, false /* retprobe */, info.target_pid,
					     offset.first.c_str(), offset.second);
		if (link) {
			info.links.emplace_back(link);
		}
	}

	info->ringbuf =
		ring_buffer__new(bpf_map__fd(m_skel->maps.ringbuf), &HandleEvent, info, nullptr);
	if (!info->ringbuf) {
		globalLogger.log_error("Failed to create ring buffer for target: " +
				       target_sym_name);
		for (auto link : info.links) {
			bpf_link__destroy(link);
		}
		return false;
	}

	// Add this ring buffer's fd to epoll
	int rb_fd = ring_buffer__epoll_fd(info->ringbuf);
	struct epoll_event ev = {};
	ev.events = EPOLLIN;
	ev.data.ptr = info->ringbuf; // identify which ringbuf triggered event
	if (epoll_ctl(m_epoll_fd, EPOLL_CTL_ADD, rb_fd, &ev) < 0) {
		cerr << "Failed to add ringbuf fd to epoll for " << sec_name << "\n";
		ring_buffer__free(info->ringbuf);
		info->ringbuf = nullptr;
		bpf_link__destroy(info->link);
		info->link = nullptr;
		return false;
	}

	// If this is the first program attached, start polling
	if (!m_poll_thread.joinable()) {
		StartPolling();
	}

	m_programs[target_func] = info;

	return true;
}

bool ProbeManager::DetachProgram(ProbeTarget target_func)
{
	// Log
	globalLogger.log_info("Detaching probe for target: " +
			      GetSymbolNameFromProbeTarget(target_func));

	auto it = m_programs.find(target_func);
	if (it == m_programs.end()) {
		globalLogger.log_error("Program not found: " +
				       GetSymbolNameFromProbeTarget(target_func));
		return false;
	}

	ProgramInfo &info = it->second;
	if (!info.link) {
		globalLogger.log_error("Program not attached: " +
				       GetSymbolNameFromProbeTarget(target_func));
		return false;
	}

	// Remove from epoll
	int rb_fd = ring_buffer__epoll_fd(info.ringbuf);
	epoll_ctl(m_epoll_fd, EPOLL_CTL_DEL, rb_fd, nullptr);

	ring_buffer__free(info.ringbuf);
	info.ringbuf = nullptr;

	bpf_link__destroy(info.link);
	info.link = nullptr;

	// If no more programs attached, stop polling
	if (!AnyProgramAttached()) {
		StopPolling();
	}

	m_programs.erase(it);

	return true;
}

void ProbeManager::Shutdown() {
	StopPolling();
	Cleanup();
}

void ProbeManager::Cleanup()
{
	for (auto &pair : m_programs) {
		ProgramInfo &info = pair.second;
		if (info.ringbuf) {
			ring_buffer__free(info.ringbuf);
		}
		if (info.link) {
			bpf_link__destroy(info.link);
		}
		delete &info;
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
		if (pair.second.link)
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

// Static callback for ring buffer events
// We get a pointer to this manager as ctx, and data from the ring buffer.
static int ProbeManager::HandleEvent(void *ctx, void *data, size_t size)
{
	// Global log
	globalLogger.log_info("Handling event");

	ProgramInfo *info = static_cast<ProgramInfo *>(ctx);
	// Now we know which program this event came from by info->target_func.
	ProbeManager *manager = info->manager;
	manager->ProcessEvent(data, size, info);
	return 0;
}

// Process an event into an AllocationEvent and enqueue it to the event queue
void ProbeManager::ProcessEvent(const void *data, size_t size, const ProgramInfo &info)
{
	// TODO different kinds of events
	CudaMemcpyEvent *evt = (CudaMemcpyEvent *)data;

	AllocationEvent event(evt->source, evt->timestamp, evt->size, EventType::DEVICE_TRANSFER);

	// Global log including event details, using AllocationEvent's to_string method
	globalLogger.log_info("Event: " + event.ToString());
	event_queue.enqueue(event);
}
