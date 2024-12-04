#include <set>
#include <map>
#include <iostream>
#include <unordered_map>

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
// Current implementation is naive multiset of events
class AllocationHistory {
    public:
	AllocationHistory(AllocationInfo alloc_info, EventInfo initial_event)
    {
        this->alloc_info = alloc_info;

        transfer_count = 0;
        state = AllocationState::UNKOWN;

        SubmitEvent(initial_event);
    }

    // Accessors
	unsigned long GetTransferCount() const
	{
		return transfer_count;
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

    private:
    AllocationInfo alloc_info;

    AllocationState state;
	unsigned long transfer_count;

    // Events in chronological order
	multiset<EventInfo> events;
};

// Tracks the history of memory events
export class MemHistory {

    public:
    MemHistory() {
        allocations = map<unsigned long, AllocationHistory>();
    }

    // Record a new memory event
    void RecordEvent(AllocationEvent event) {
        auto alloc_info = event.allocation_info;
        auto event_info = event.event_info;

        if (allocations.find(alloc_info.start) == allocations.end()) {
            allocations[alloc_info.start] = AllocationHistory(alloc_info, event_info);
        }

        allocations[alloc_info.start].SubmitEvent(event_info);
    }

    private:
    map<unsigned long, AllocationHistory> allocations;
};
