#pragma once
#include <sys/types.h>
#include <set>
#include <thread>
#include <shared_mutex>


#include "MemHistory.h"
#include "ProbeManager.h"

using namespace std;

// Class that manages launching probes and collecting their events

// TracerAgent class

class TracerAgent {
    public:
    TracerAgent()
    {
        m_target_pid = getpid();
        probe_manager = make_unique<ProbeManager>(m_event_queue);
    }

    TracerAgent(pid_t pid)
    {
        m_target_pid = pid;
        probe_manager = make_unique<ProbeManager>(m_event_queue);
    }

    ~TracerAgent() {
        // // Ensure proper cleanup
        // if (event_processing_thread.joinable()) {
        //     event_processing_thread.join();
        // }
        StopAgent();
    }

    void StartAgentAsync();

    void HandleEvent(AllocationEvent event, AllocationIdentifier identifier);

    void DumpHistory(const char *filename) {
        DumpHistory(filename, false);
    }

    void DumpHistory(const char *filename, bool verbose);

    void StopAgent();

    // void ConstructProbes();

    private:
    void ProcessEvents();

    pid_t m_target_pid;
    const char *dump_filename = "history.json";
    bool verbose = false;

    shared_mutex history_mutex;
    MemHistory mem_history;

    unique_ptr<ProbeManager> probe_manager;
    ThreadSafeQueue<AllocationEvent> m_event_queue;

    thread m_proccessing_thread;
};;