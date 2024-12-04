#pragma once

#include <set>
#include <string>

using namespace std;

// Uniquely describes an allocation for its entire lifetime
struct AllocationInfo {
    unsigned long start;
    unsigned long size;
    // Currently unused
    unsigned long identifier;

    AllocationInfo() : start(0), size(0), identifier(0) {}
    AllocationInfo(unsigned long s, unsigned long sz, unsigned long id) : start(s), size(sz), identifier(id) {}

    bool operator<(const AllocationInfo &other) const;
    string ToString() const;
};


enum class EventType {
	ALLOC, // Allocation event, creates a new allocation
	HOST_TRANSFER, // Transfer event, moves data to another allocation on the host (e.g pinning)
	DEVICE_TRANSFER, // Transfer event, moves data to the device
	FREE // Free event, deallocates an allocation
};

constexpr const char* EventTypeToString(EventType type) {
    switch (type) {
        case EventType::ALLOC:
            return "ALLOC";
        case EventType::HOST_TRANSFER:
            return "HOST_TRANSFER";
        case EventType::DEVICE_TRANSFER:
            return "DEVICE_TRANSFER";
        case EventType::FREE:
            return "FREE";
        default:
            return "UNKNOWN";
    }
}

struct EventInfo {
    unsigned long timestamp;
    EventType type;

    EventInfo(unsigned long ts, EventType et) : timestamp(ts), type(et) {}

    bool operator<(const EventInfo &other) const;
    bool operator>(const EventInfo &other) const;
    string ToString() const;
};

struct AllocationEvent {
	AllocationInfo allocation_info;
    EventInfo event_info;

    AllocationEvent(AllocationInfo alloc_info, EventInfo event_info) : allocation_info(alloc_info), event_info(event_info) {}
    AllocationEvent(unsigned long start, unsigned long size, unsigned long identifier, unsigned long timestamp, EventType type) : allocation_info(start, size, identifier), event_info(timestamp, type) {}

    bool operator<(const AllocationEvent &other) const;
    string ToString() const;
    
};

// Should eventually have more states
enum class AllocationState { ALLOCATED, FREED, UNKOWN };
constexpr const char* AllocationStateToString(AllocationState state) {
    switch (state) {
        case AllocationState::ALLOCATED:
            return "ALLOCATED";
        case AllocationState::FREED:
            return "FREED";
        case AllocationState::UNKOWN:
            return "UNKNOWN";
        default:
            return "UNKNOWN";
    }
}

// Tracks the history of a single allocation
// Current implementation is naive multiset of events (hotspot/coldspot optimization happens in the outer class)
class AllocationHistory {
public:
    AllocationHistory(AllocationInfo alloc_info, EventInfo initial_event);

    unsigned long GetTransferCount() const;

    unsigned long GetStartAddress() const;

    AllocationState GetState() const;

    const EventInfo& GetLatestEvent() const;

    void SubmitEvent(EventInfo event);

    string ToString() const {
        return ToString(false);
    }

    string ToString(bool verbose) const;

private:
    AllocationState CalculateNextState(EventType new_type);
    bool IsLatestEvent(const EventInfo& event) const;

private:
    AllocationInfo alloc_info;
    AllocationState state;
    unsigned long transfer_count;
    multiset<EventInfo> events;
};