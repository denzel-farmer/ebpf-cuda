#pragma once

#include <set>
#include <string>
#include <boost/property_tree/ptree.hpp>

using namespace std;

struct AllocationIdentifier {
    unsigned long call_site;
    unsigned long call_no;

    AllocationIdentifier() : call_site(0), call_no(0) {}
    AllocationIdentifier(unsigned long site, unsigned long no) : call_site(site), call_no(no) {}
};


// Uniquely describes an allocation for its entire lifetime
struct AllocationRange {
    unsigned long start;
    unsigned long size;
    // // Currently unused
    // unsigned long identifier;

    AllocationRange() : start(0), size(0) {}
    AllocationRange(unsigned long s, unsigned long sz) : start(s), size(sz) {}

    bool operator<(const AllocationRange &other) const;

    boost::property_tree::ptree PtreeSerialize() const;
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
    boost::property_tree::ptree PtreeSerialize() const;
    string ToString() const;
};

// Even not associated with identifier
struct AllocationEvent {
	AllocationRange allocation_info;
    EventInfo event_info;

    AllocationEvent(AllocationRange alloc_info, EventInfo event_info) : allocation_info(alloc_info), event_info(event_info) {}
    AllocationEvent(unsigned long start, unsigned long size, unsigned long timestamp, EventType type) : allocation_info(start, size), event_info(timestamp, type) {}

    bool operator<(const AllocationEvent &other) const;
    boost::property_tree::ptree PtreeSerialize() const;
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
    AllocationHistory(AllocationRange alloc_info, EventInfo initial_event, AllocationIdentifier alloc_tag);
    AllocationHistory(AllocationRange alloc_info, EventInfo initial_event);

    unsigned long GetTransferCount() const;
    unsigned long GetStartAddress() const;
    AllocationState GetState() const;
    optional<AllocationIdentifier> GetAllocTag() const;

    const EventInfo& GetLatestEvent() const;

    void SubmitEvent(EventInfo event);

    boost::property_tree::ptree PtreeSerialize(bool verbose) const;

    string ToString() const {
        return ToString(false);
    }

    string ToString(bool verbose) const;

private:
    AllocationState CalculateNextState(EventType new_type);
    bool IsLatestEvent(const EventInfo& event) const;

private:
    AllocationRange alloc_info;
    optional<AllocationIdentifier> alloc_tag;
    AllocationState state;
    unsigned long transfer_count;
    multiset<EventInfo> events;
};