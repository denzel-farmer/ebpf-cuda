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

void TracerAgent::StartAgentAsync()
{

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
	// event_processing_thread = thread(&TracerAgent::ProcessEvents, this);
}


void TracerAgent::StopAgent() {
	probe_manager->Shutdown();
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
// 	event_queue.terminate();
// 	if (event_processing_thread.joinable()) {
// 		event_processing_thread.join();
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

void TracerAgent::ProcessEvents()
{
	while (true) {
		optional<AllocationEvent> event = m_event_queue.dequeue_wait();
		if (event.has_value()) {
			// Process the event, locking as a writer
			lock_guard<shared_mutex> lock(history_mutex);
			mem_history.RecordEvent(event.value());
		} else {
			// Exit if queue is done and empty
			break;
		}
	}

	globalLogger.log_info("Event processing thread exiting");
	StopAgent();
}
