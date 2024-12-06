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
    globalLogger.log_info("EventProbe launched (fake events)");

    AllocationEvent event = AllocationEvent(0, 0, 0, EventType::ALLOC);
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
    return AllocationEvent(0, 0, 0, EventType::DEVICE_TRANSFER);
}

void EventProbe::Configure() {
    globalLogger.log_info("EventProbe configure stub");
}



// // EventProbe class
// class EventProbe {
//     public:
//     // Constructor that includes queue reference
//     EventProbe(ThreadSafeQueue<AllocationEvent> &queue)
//         : event_queue(queue)
//     {

//         globalLogger.log_info("EventProbe created");
//     }

// 	void LaunchProbe()
// 	{
//         globalLogger.log_info("EventProbe launched");

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
