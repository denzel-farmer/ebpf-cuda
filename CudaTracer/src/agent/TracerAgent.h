#pragma once
#include <sys/types.h>
#include <set>
#include <thread>
#include <shared_mutex>


#include "MemHistory.h"
#include "ProbeManager.h"
#include "Allocation.h"

using namespace std;


// Top-level class that manages launching probes and collecting their events
class TracerAgent {
    public:
    TracerAgent() : TracerAgent(getpid()) {}

    TracerAgent(pid_t pid)
    {
        m_running.store(false);
        m_target_pid = pid;
        m_probe_manager = make_unique<ProbeManager>(m_event_queue);
    }

    ~TracerAgent() {
        StopAgent();
        CleanupAgent();
    }

    // Start and stop the agent (low overhead), returns success
    bool StartAgentAsync(bool transfer_only = false);
    void StopAgent();

    // Provide events to the agent 
    // void HandleEvent(AllocationEvent event, AllocationIdentifier identifier);
    void HandleEvent(AllocationEvent event);

    // Generic interface to get concurrency-safe MemHistory reference
    const MemHistory& GetMemHistory();

    // Transfer hotspot/coldspot specific interface
    vector<Allocation> GetHotspots(size_t num) const;
    vector<Allocation> GetColdspots(size_t num) const;

    vector<Allocation> GetHotspotsThreshold(unsigned long min_transfers) const;
    vector<Allocation> GetColdspotsThreshold(unsigned long max_transfers) const;

    // Dump the MemHistory to a file 
    void DumpHistory(const char *filename, DumpFormat format) {
        DumpHistory(filename, format, false);
    }
    void DumpHistory(const char *filename, DumpFormat format, bool verbose);


    private:
    void CleanupAgent();
    // Thread main loop for processing events from queue
    void ProcessEvents();

    pid_t m_target_pid;

    MemHistory m_history;
    
    shared_mutex m_status_mutex;
    atomic<bool> m_running;

    unique_ptr<ProbeManager> m_probe_manager;
    ThreadSafeQueue<AllocationEvent> m_event_queue;

    thread m_proccessing_thread;
};;