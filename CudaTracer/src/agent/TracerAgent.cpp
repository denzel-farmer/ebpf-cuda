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
void TracerAgent::StartAgentAsync()
{
	if (m_running.exchange(true)) {
		globalLogger.log_error("Agent already running");
		return;
	}

	bool success = probe_manager->AttachProbe(ProbeTarget::CUDA_MEMCPY, m_target_pid);
	if (!success) {
		globalLogger.log_error("Failed to attach probe");
		return;
	}
	// // Configure and attach probes
	// for (auto &probe : probes) {
	// 	probe->Configure();
	// 	// Launch each probe in a new thread
	// 	probe->probe_thread = thread(&EventProbe::LaunchProbe, probe.get());
	// }
	// // Start the event processing loop in a separate thread
	m_proccessing_thread = thread(&TracerAgent::ProcessEvents, this);
}


void TracerAgent::StopAgent() {
	if (m_running.exchange(false)) {
		probe_manager->DetachProbe(ProbeTarget::CUDA_MEMCPY);
	}

	// probe_manager->Shutdown();
	
	// // Signal thread to stop
	// m_event_queue.terminate();
	
	// // if (m_proccessing_thread.joinable()) {
	// // 	m_proccessing_thread.join();
	// }
}

void TracerAgent::CleanupAgent() {
	probe_manager->Shutdown();
	
	// Signal thread to stop
	m_event_queue.terminate();
	
	// if (m_proccessing_thread.joinable()) {
	// 	m_proccessing_thread.join();
}

// // Shutdown the agent, likely due to the target process exiting
// void TracerAgent::ShutdownAgent()
// {
// 	// Signal probes to stop
// 	for (auto &probe : probes) {
// 		probe->Terminate();
// 	}
// 	// Wait for probe threads to finish
// 	for (auto &probe : probes) {
// 		if (probe->probe_thread.joinable()) {
// 			probe->probe_thread.join();
// 		}
// 	}
// 	// Signal the processing thread to exit
	// event_queue.terminate();
	// if (event_processing_thread.joinable()) {
	// 	event_processing_thread.join();
// 	}

void TracerAgent::DumpHistory(const char *filename, bool verbose) {
	// Lock history as a writer, and dump the history to a file
	lock_guard<shared_mutex> lock(history_mutex);
	globalLogger.log_info("Dumping history to file");
	ofstream dump_file(filename);
	if (dump_file.is_open()) {
		mem_history.JSONSerialize(dump_file, verbose);
		dump_file.close();
		globalLogger.log_info("History successfully dumped to file");
	} else {
		globalLogger.log_error("Failed to open dump file");
	}
}

// Non-identifier event 
void TracerAgent::HandleEvent(AllocationEvent event) {
	// Process the event, locking as a writer
	// construct log string with start address (hex) size (hex) and event type
	std::stringstream log_stream;
	log_stream << "[TracerAgent->HandleEvent]: No identifier, start address = 0x" << std::hex << event.allocation_info.start 
			   << ", size = 0x" << event.allocation_info.size 
			   << ", type = " << EventTypeToString(event.event_info.type);
	std::string log_string = log_stream.str();
	
	globalLogger.log_info(log_string.c_str());

	lock_guard<shared_mutex> lock(history_mutex);
	mem_history.RecordEvent(event);
}

// Identifier event
void TracerAgent::HandleEvent(AllocationEvent event, AllocationIdentifier identifier) {
	//cerr << "Event call site: " << std::hex << identifier.call_site << ", call no: " << std::dec << identifier.call_no << endl;
	// Process the event, locking as a writer
	//globalLogger.log_info("Handling event");
	std::stringstream log_stream;
	log_stream << "[TracerAgent->HandleEvent]: Has identifier, start address = 0x" << std::hex << event.allocation_info.start 
			   << ", size = 0x" << event.allocation_info.size 
			   << ", type = " << EventTypeToString(event.event_info.type);
	std::string log_string = log_stream.str();

	globalLogger.log_info(log_string.c_str());
	// Log the identifier call site and call number
	std::stringstream log_stream2;
	log_stream2 << ", call site = 0x" << std::hex << identifier.call_site 
			   << ", call no = " << std::dec << identifier.call_no;
	std::string log_string2 = log_stream2.str();
	globalLogger.log_info(log_string2.c_str());

	lock_guard<shared_mutex> lock(history_mutex);
	mem_history.RecordEvent(event, identifier);
}


void TracerAgent::ProcessEvents()
{
	while (true) {
		globalLogger.log_debug("Event processing thread waiting for event");
		optional<AllocationEvent> event = m_event_queue.dequeue_wait();
		globalLogger.log_debug("Event maybe dequeued");
		if (event.has_value()) {
			// Process the event, locking as a writer
			lock_guard<shared_mutex> lock(history_mutex);
			globalLogger.log_debug("Event dequeued");
			mem_history.RecordEvent(event.value());
		} else {
			// Exit if queue is done and empty
			break;
		}
	}

	globalLogger.log_info("Event processing thread exiting");
	StopAgent();
}
