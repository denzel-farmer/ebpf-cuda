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

// // Uniquely describes an allocation for its entire lifetime
// export struct AllocationInfo {
// 	unsigned long start;
// 	unsigned long size;
// 	// Currently unused
// 	unsigned long identifier;
// };

// export enum class EventType {
// 	ALLOC, // Allocation event, creates a new allocation
// 	HOST_TRANSFER, // Transfer event, moves data to another allocation on the host (e.g pinning)
// 	DEVICE_TRANSFER, // Transfer event, moves data to the device
// 	FREE // Free event, deallocates an allocation
// };

// export struct EventInfo {
//     unsigned long timestamp;
//     EventType type;
// };

// export struct AllocationEvent {
// 	AllocationInfo allocation_info;
//     EventInfo event_info;

// 	bool operator<(const AllocationEvent &other) const
// 	{
// 		return event_info.timestamp < other.event_info.timestamp; // Order by timestamp
// 	}
// };

// // Should eventually have more states
// export enum class AllocationState { ALLOCATED, FREED, UNKOWN };

// Tracks the history of a single allocation
// Current implementation is naive multiset of events (hotspot/coldspot optimization happens in the outer class)
AllocationHistory::AllocationHistory(AllocationInfo alloc_info, EventInfo initial_event)
{
    this->alloc_info = alloc_info;
    transfer_count = 0;
    state = AllocationState::UNKOWN;
    SubmitEvent(initial_event);
}

unsigned long AllocationHistory::GetTransferCount() const {
    return transfer_count;
}

unsigned long AllocationHistory::GetStartAddress() const {
    return alloc_info.start;
}

AllocationState AllocationHistory::GetState() const {
    return state;
}

const EventInfo& AllocationHistory::GetLatestEvent() const {
    return *events.rbegin();
}

void AllocationHistory::SubmitEvent(EventInfo event) {
    events.insert(event);

    // Only update state if event is the latest
    if (event.timestamp > GetLatestEvent().timestamp) {
        state = CalculateNextState(event.type);
    }

    if (event.type == EventType::DEVICE_TRANSFER) {
        transfer_count++;
    }
}

AllocationState AllocationHistory::CalculateNextState(EventType new_type) {
    switch (new_type) {
        case EventType::ALLOC:
            return AllocationState::ALLOCATED;
            break;
        case EventType::FREE:
            return AllocationState::FREED;
            break;
        default:
            return state;
    }
}

// Optimized container for tracking allocation history
using namespace boost::multi_index;

struct by_start_address {};
struct by_transfer_count {};

typedef multi_index_container<
    AllocationHistory,
    indexed_by<
        // Primary key is start address, used for fast lookup
        ordered_unique<
            tag<by_start_address>,
            const_mem_fun<AllocationHistory, unsigned long, &AllocationHistory::GetStartAddress>
        >,

        // Secondary key is transfer count, used for hotspot/coldspot optimization
        ordered_non_unique<
            tag<by_transfer_count>,
            const_mem_fun<AllocationHistory, unsigned long, &AllocationHistory::GetTransferCount>,
            std::greater<unsigned long>
        >
    >
> AllocationHistoryContainer;

// Tracks the history of memory events
export class MemHistory {

    public:
    MemHistory() {
    }

    // Record a new memory event
    void RecordEvent(AllocationEvent event) {
        UpdateHistories(event.allocation_info, event.event_info);
    }

    // TODO one of thse functions is wrong 

    // Retrieve references to allocation history for n hotspots
    vector<const AllocationHistory*> GetHotspots(int num) const {
        vector<const AllocationHistory*> hotspots;
        auto &index_by_transfer_count = histories.get<by_transfer_count>();

        auto it = index_by_transfer_count.begin();
        for (int i = 0; i < num && it != index_by_transfer_count.end(); i++, it++) {
            hotspots.push_back(&(*it));
        }

        return hotspots;
    }

    // Retrieve references to allocation history for all coldspots with fewer than max_transfers transfers
    vector<const AllocationHistory*> GetColdspots(int max_transfers) const {
        vector<const AllocationHistory*> coldspots;
        auto &index_by_transfer_count = histories.get<by_transfer_count>();

        auto it = index_by_transfer_count.begin();
        while (it != index_by_transfer_count.end() && it->GetTransferCount() < max_transfers) {
            coldspots.push_back(&(*it));
            it++;
        }

        return coldspots;
    }

    private:
    void UpdateHistories(AllocationInfo alloc_info, EventInfo event_info) {
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

    private:
    AllocationHistoryContainer histories;
};
