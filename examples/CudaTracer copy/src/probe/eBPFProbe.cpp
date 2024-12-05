#pragma once

#include "AllocationHistory.h"
#include "SyncUtils.h"
#include "MemHistory.h"
#include "Logger.h"
#include "EventProbe.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <optional>
#include <set>
#include <shared_mutex>
#include <thread>
#include <iostream>


class eBPFEventProbe : public EventProbe {
    public:
    eBPFEventProbe(ThreadSafeQueue<AllocationEvent> &queue) : EventProbe(queue) {}

    private:
    optional<AllocationEvent> PollEvent() {
        return AllocationEvent(0, 0, 0, 0, EventType::DEVICE_TRANSFER);
    }
};