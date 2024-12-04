#include <set>
#include <map>
#include <iostream>
#include <unordered_map>

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/mem_fun.hpp>

using namespace std;

#include "MemHistory.h"

// // Tracks the history of a single allocation
// AllocationHistory::AllocationHistory(AllocationInfo alloc_info, EventInfo initial_event)
// {
//     this->alloc_info = alloc_info;
//     transfer_count = 0;
//     state = AllocationState::UNKOWN;
//     SubmitEvent(initial_event);
// }

// unsigned long AllocationHistory::GetTransferCount() const {
//     return transfer_count;
// }

// unsigned long AllocationHistory::GetStartAddress() const {
//     return alloc_info.start;
// }

// AllocationState AllocationHistory::GetState() const {
//     return state;
// }

// const EventInfo& AllocationHistory::GetLatestEvent() const {
//     return *events.rbegin();
// }

// void AllocationHistory::SubmitEvent(EventInfo event) {

//     // Only update state if event is the latest
//     if (IsLatestEvent(event)) {
//         state = CalculateNextState(event.type);
//     }

//     if (event.type == EventType::DEVICE_TRANSFER) {
//         transfer_count++;
//     }
    
//     events.insert(event);
//     cout << "Event submitted: " << event.ToString() << endl;

// }

// AllocationState AllocationHistory::CalculateNextState(EventType new_type) {
//     switch (new_type) {
//         case EventType::ALLOC:
//             assert(state != AllocationState::ALLOCATED && "Memory already allocated");
//             return AllocationState::ALLOCATED;
//             break;
//         case EventType::FREE:
//             return AllocationState::FREED;
//             break;
//         default:
//             return state;
//     }
// }

// bool AllocationHistory::IsLatestEvent(const EventInfo& event) const {
//     return events.empty() || (event > GetLatestEvent());
// }

// Tracks the history of memory events
MemHistory::MemHistory() {
}

// Record a new memory event
void MemHistory::RecordEvent(AllocationEvent event) {

    // TODO do edge case testing here (ex no overlapping duplicate allocs, no double frees)

    // TODO do merging/splitting here

    UpdateHistories(event.allocation_info, event.event_info);
}

// Retrieve references to allocation history for n hotspots
vector<const AllocationHistory*> MemHistory::GetHotspots(int num) const {
    vector<const AllocationHistory*> hotspots;
    auto &index_by_transfer_count = histories.get<by_transfer_count>();

    auto it = index_by_transfer_count.begin();
    for (int i = 0; i < num && it != index_by_transfer_count.end(); i++, it++) {
        hotspots.push_back(&(*it));
    }

    return hotspots;
}

// Retrieve references to allocation history for all coldspots with fewer than max_transfers transfers
vector<const AllocationHistory*> MemHistory::GetColdspots(unsigned long max_transfers) const {
    vector<const AllocationHistory*> coldspots;
    auto &index_by_transfer_count = histories.get<by_transfer_count>();

    auto it = index_by_transfer_count.rbegin();
    while (it != index_by_transfer_count.rend() && it->GetTransferCount() < max_transfers) {
        coldspots.push_back(&(*it));
        it++;
    }

    return coldspots;
}

vector <const AllocationHistory*> MemHistory::GetAllocationHistories() const {
    vector<const AllocationHistory*> all_histories;
    for (const auto &history : histories) {
        all_histories.push_back(&history);
    }

    return all_histories;
}

void MemHistory::UpdateHistories(AllocationInfo alloc_info, EventInfo event_info) {
    // If no allocation in container, create a new one 
    auto &index_by_start = histories.get<by_start_address>();
    auto it = index_by_start.find(alloc_info.start);

    if (it == index_by_start.end()) {
        // Allocation does not exist, create a new one
        AllocationHistory new_alloc(alloc_info, event_info);
        histories.insert(move(new_alloc));
    } else {
        // Allocation exists, update safely 
        index_by_start.modify(it, [&](AllocationHistory &alloc) {
            alloc.SubmitEvent(event_info);
        });
    }
}
