#include <set>
#include <map>
#include <iostream>
#include <unordered_map>

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/mem_fun.hpp>

using namespace std;

export module MemHistory;

// Uniquely describes an allocation for its entire lifetime
export struct AllocationInfo {
	unsigned long start;
	unsigned long size;
	// Currently unused
	unsigned long identifier;
};

export enum class EventType {
	ALLOC, // Allocation event, creates a new allocation
	HOST_TRANSFER, // Transfer event, moves data to another allocation on the host (e.g pinning)
	DEVICE_TRANSFER, // Transfer event, moves data to the device
	FREE // Free event, deallocates an allocation
};

export struct EventInfo {
    unsigned long timestamp;
    EventType type;
};

export struct AllocationEvent {
	AllocationInfo allocation_info;
    EventInfo event_info;

	bool operator<(const AllocationEvent &other) const
	{
		return event_info.timestamp < other.event_info.timestamp; // Order by timestamp
	}
};

// Should eventually have more states
export enum class AllocationState { ALLOCATED, FREED, UNKOWN };

// Tracks the history of a single allocation
// Current implementation is naive multiset of events (hotspot/coldspot optimization happens in the outer class)
class AllocationHistory {
    public:
	AllocationHistory(AllocationInfo alloc_info, EventInfo initial_event)
    {
        this->alloc_info = alloc_info;

        transfer_count = 0;
        state = AllocationState::UNKOWN;

        SubmitEvent(initial_event);
    }

    unsigned long GetTransferCount() const {
        return transfer_count;
    }

    unsigned long GetStartAddress() const {
        return alloc_info.start;
    }

	AllocationState GetState() const
	{
		return state;
	}

    const EventInfo& GetLatestEvent() const {
        return *events.rbegin();
    }


    void SubmitEvent(EventInfo event) {
        events.insert(event);

        // Only update state if event is the latest
        if (event.timestamp > GetLatestEvent().timestamp) {
            state = CalculateNextState(event.type);
        }

        if (event.type == EventType::DEVICE_TRANSFER) {
            transfer_count++;
        }
    }

    private:
    // Update own state based on latest event 
    AllocationState CalculateNextState(EventType new_type) {
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

    public:
    AllocationInfo alloc_info;

    AllocationState state;
	unsigned long transfer_count;

    // Events in chronological order
	multiset<EventInfo> events;
};

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
