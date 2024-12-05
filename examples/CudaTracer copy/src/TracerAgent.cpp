#include <sys/types.h>
#include <vector>

// #include "TracerAgent.h"
#include "AllocationHistory.h"
#include "SyncUtils.h"
#include "MemHistory.h"
#include "Logger.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <optional>
#include <set>
#include <shared_mutex>
#include <thread>
#include <iostream>

// // EventProbe class
// class EventProbe {
//     public:
//     // Constructor that includes queue reference
//     EventProbe(ThreadSafeQueue<AllocationEvent> &queue)
//         : event_queue(queue)
//     {

//         Logger::log_info("EventProbe created");
//     }

// 	void LaunchProbe()
// 	{
//         Logger::log_info("EventProbe launched");

//         AllocationEvent event = AllocationEvent(0, 0, 0, 0, EventType::ALLOC);
//         event_queue.enqueue(event);

// 		while (!CheckStop()) {
// 			auto event = PollEvent();
//             if (event.has_value()) {
//                 event_queue.enqueue(event.value());
//             }
    
// 			this_thread::sleep_for(chrono::milliseconds(100));
// 		}
// 	}

// 	void Terminate()
// 	{
// 		stop_flag.store(true, memory_order_release);
// 	}

//     private:
//     optional<AllocationEvent> PollEvent() {
//         return AllocationEvent(0, 0, 0, 0, EventType::DEVICE_TRANSFER);
//     }


//     inline bool CheckStop() {
//         return stop_flag.load(memory_order_acquire);
//     }

//     public:
// 	thread thread;

//     private:
// 	atomic<bool> stop_flag{ false };
//     ThreadSafeQueue<AllocationEvent> &event_queue;
// };

// TracerAgent class
class TracerAgent {
    public:
	TracerAgent(pid_t pid)
		: target_pid(pid)
	{
	}

	void StartAgentAsync()
	{
		// Configure and attach probes
		for (auto &probe : probes) {
			probe.Initialize();
			// Launch each probe in a new thread
			probe.thread = thread(&EventProbe::LaunchProbe, &probe,
						   ref(event_queue));
		}
		// Start the event processing loop in a separate thread
		event_processing_thread = thread(&TracerAgent::ProcessEvents, this);
	}

    // Shutdown the agent, likely due to the target process exiting
	void ShutdownAgent()
	{
		// Signal probes to stop
		for (auto &probe : probes) {
			probe.Terminate();
		}
		// Wait for probe threads to finish
		for (auto &probe : probes) {
			if (probe.thread.joinable()) {
				probe.thread.join();
			}
		}
		// Signal the processing thread to exit
		event_queue.terminate();
		if (event_processing_thread.joinable()) {
			event_processing_thread.join();
		}

        // Lock history as a writer, and dump the history to a file
        lock_guard<shared_mutex> lock(history_mutex);
        Logger::log_info("Dumping history to file");
        ofstream dump_file(dump_filename);
        if (dump_file.is_open()) {
            mem_history.JSONSerialize(dump_file, verbose);
            dump_file.close();
            Logger::log_info("History successfully dumped to file");
        } else {
            Logger::log_error("Failed to open dump file");
        }
	}

    private:
	void ProcessEvents()
	{
		while (true) {
			optional<AllocationEvent> event = event_queue.dequeue_wait();
			if (event.has_value()) {
				// Process the event, locking as a writer
                lock_guard<shared_mutex> lock(history_mutex);
				mem_history.RecordEvent(event.value());
			} else {
				// Exit if queue is done and empty
				break;
			}
		}

		Logger::log_info("Event processing thread exiting");
        ShutdownAgent();
	}

	pid_t target_pid;
    const char *dump_filename = "history.json";
    bool verbose = false;

    shared_mutex history_mutex;
	MemHistory mem_history;

	vector<EventProbe> probes;
	ThreadSafeQueue<AllocationEvent> event_queue;
	thread event_processing_thread;
};
