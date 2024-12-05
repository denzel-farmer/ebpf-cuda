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



optional<AllocationEvent> eBPFProbe::PollEvent() {
        return AllocationEvent(0, 0, 0, 0, EventType::DEVICE_TRANSFER);
}