#include <sys/types.h>
#include <vector>

#include "TracerAgent.h"
#include "AllocationHistory.h"
#include "SyncUtils.h"
#include "MemHistory.h"
#include "Logger.h"

#include "ProbeManager.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <optional>
#include <set>
#include <shared_mutex>
#include <thread>
#include <iostream>

// TracerAgent class
// TODO add a lock to avoid the race between start and stop
bool TracerAgent::StartAgentAsync()
{
	if (m_running.exchange(true)) {
		globalLogger.log_error("Agent already running");
		return false;
	}

	bool success = m_probe_manager->AttachProbe(ProbeTarget::CUDA_MEMCPY, m_target_pid);
	if (!success) {
		globalLogger.log_error("Failed to attach probe");
	}
	success = m_probe_manager->AttachProbe(ProbeTarget::PIN_PAGES, m_target_pid);
	if (!success) {
		globalLogger.log_error("Failed to attach probe");
	}
	success = m_probe_manager->AttachProbe(ProbeTarget::CUDA_HOST_ALLOC, m_target_pid);
	if (!success) {
		globalLogger.log_error("Failed to attach probe");
	}
	success = m_probe_manager->AttachProbe(ProbeTarget::CUDA_HOST_FREE, m_target_pid);
	if (!success) {
		globalLogger.log_error("Failed to attach probe");
	}

	m_proccessing_thread = thread(&TracerAgent::ProcessEvents, this);
	return true;
}


void TracerAgent::StopAgent() {
	if (m_running.exchange(false)) {
		m_probe_manager->DetachProbe(ProbeTarget::CUDA_MEMCPY);
	}
}

void TracerAgent::CleanupAgent() {
	m_probe_manager->Shutdown();
	
	// Signal thread to stop
	m_event_queue.terminate();
	
}

const MemHistory& TracerAgent::GetMemHistory() {
	return m_history;
}


// Transfer hotspot/coldspot specific interface
vector<Allocation> TracerAgent::GetHotspots(size_t num) const {
	return m_history.GetHotspots(num);
}
vector<Allocation> TracerAgent::GetColdspots(size_t num) const {
	return m_history.GetColdspots(num);
}

vector<Allocation> TracerAgent::GetHotspotsThreshold(unsigned long min_transfers) const {
	return m_history.GetHotspotsThreshold(min_transfers);
}
vector<Allocation> TracerAgent::GetColdspotsThreshold(unsigned long max_transfers) const {
	return m_history.GetColdspotsThreshold(max_transfers);
}




void TracerAgent::DumpHistory(const char *filename, DumpFormat format, bool verbose) {

	m_history.SaveDatabase(filename, format, verbose);
}

// Non-identifier event 
void TracerAgent::HandleEvent(AllocationEvent event) {
	// Process the event, locking as a writer
	// construct log string with start address (hex) size (hex) and event type
	std::stringstream log_stream;
	log_stream << "[TracerAgent->HandleEvent]: Event with start address = 0x" << std::hex << event.allocation_info.start 
			   << ", size = 0x" << event.allocation_info.size 
			   << ", type = " << EventTypeToString(event.event_info.type);
	std::string log_string = log_stream.str();
	
	globalLogger.log_info(log_string.c_str());

	m_history.RecordEvent(event);
}

void TracerAgent::ProcessEvents()
{
	while (true) {
		globalLogger.log_debug("Event processing thread waiting for event");
		optional<AllocationEvent> event = m_event_queue.dequeue_wait();
		globalLogger.log_debug("Event maybe dequeued");
		if (event.has_value()) {
			globalLogger.log_debug("Event dequeued");
			m_history.RecordEvent(event.value());
		} else {
			// Exit if queue is done and empty
			break;
		}
	}

	globalLogger.log_info("Event processing thread exiting");
	StopAgent();
}
