#pragma once
#include <sys/types.h>
#include <set>

#include "MemHistory.h"
#include "EventProbe.h"

// Class that manages launching probes and collecting their events
class TracerAgent {



private:
void LaunchAgent() {
    // Configure and attach probes 
}

    pid_t target_pid;
    MemHistory mem_history;
    
    set<EventProbe> probes;

};