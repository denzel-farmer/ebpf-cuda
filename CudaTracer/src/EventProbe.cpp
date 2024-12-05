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


void EventProbe::LaunchProbe()
{
    Logger::log_info("EventProbe launched (fake events)");

    AllocationEvent event = AllocationEvent(0, 0, 0, 0, EventType::ALLOC);
    event_queue.enqueue(event);

    while (!CheckStop()) {
        auto event = PollEvent();
        if (event.has_value()) {
            event_queue.enqueue(event.value());
        }

        this_thread::sleep_for(chrono::milliseconds(100));
    }
}

void EventProbe::Terminate()
{
    stop_flag.store(true, memory_order_release);
}

optional<AllocationEvent> EventProbe::PollEvent() {
    return AllocationEvent(0, 0, 0, 0, EventType::DEVICE_TRANSFER);
}
