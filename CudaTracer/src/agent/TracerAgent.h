#pragma once
#include <sys/types.h>
#include <set>

#include "MemHistory.h"
#include "EventProbe.h"

// Class that manages launching probes and collecting their events

// TracerAgent class

class TracerAgent {
    public:
    TracerAgent(pid_t pid)
        : target_pid(pid)
    {
    }

    void StartAgentAsync();
    // Shutdown the agent, likely due to the target process exiting
    void ShutdownAgent();

    void ConstructProbes();

    private:
    void ProcessEvents();

    pid_t target_pid;
    const char *dump_filename = "history.json";
    bool verbose = false;

    std::shared_mutex history_mutex;
    MemHistory mem_history;

    std::set<std::unique_ptr<EventProbe>> probes;
    ThreadSafeQueue<AllocationEvent> event_queue;
    std::thread event_processing_thread;
};