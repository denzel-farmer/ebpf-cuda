#pragma once

#include <set>
#include <string>
#include <boost/property_tree/ptree.hpp>
#include <shared_mutex>

#include "Allocation.h"

using namespace std;

// struct AllocationIdentifier {
//     unsigned long call_site;
//     unsigned long call_no;

//     AllocationIdentifier() : call_site(0), call_no(0) {}
//     AllocationIdentifier(unsigned long site, unsigned long no) : call_site(site), call_no(no) {}

//     boost::property_tree::ptree PtreeSerialize() const;

//     bool operator==(const AllocationIdentifier &other) const {
//         return call_site == other.call_site && call_no == other.call_no;
//     }

//     bool operator!=(const AllocationIdentifier &other) const {
//         return !(*this == other);
//     }

// };


// // Uniquely describes an allocation for its entire lifetime
// struct AllocationRange {
//     unsigned long start;
//     unsigned long size;
//     // // Currently unused
//     // unsigned long identifier;

//     AllocationRange() : start(0), size(0) {}
//     AllocationRange(unsigned long s, unsigned long sz) : start(s), size(sz) {}

//     bool operator<(const AllocationRange &other) const;

//     boost::property_tree::ptree PtreeSerialize() const;
//     string ToString() const;
// };

// struct Allocation {
//     AllocationRange range;
//     AllocationIdentifier identifier;

//     Allocation(unsigned long start, unsigned long size, unsigned long call_site, unsigned long call_no) : range(start, size), identifier(call_site, call_no) {}
//     Allocation(AllocationRange alloc_range, AllocationIdentifier alloc_tag) : range(alloc_range), identifier(alloc_tag) {}

//     boost::property_tree::ptree PtreeSerialize() const;
//     string ToString() const;
// };


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
    CallTag call_info;
    EventType type;

    EventInfo(unsigned long ts, EventType et) : timestamp(ts), type(et) {}
    EventInfo(unsigned long ts, unsigned long call_site, unsigned long call_no, EventType et) : timestamp(ts), call_info(call_site, call_no), type(et) {}

    bool operator<(const EventInfo &other) const {
        return timestamp < other.timestamp; // Order by timestamp
    }

    bool operator>(const EventInfo &other) const {
        return timestamp > other.timestamp; // Order by timestamp
    }

    boost::property_tree::ptree PtreeSerialize() const {
        boost::property_tree::ptree root;
        root.put("timestamp", timestamp);
        root.put("type", EventTypeToString(type));

        return root;
    }

    string ToString() const {
        stringstream ss;
        ss << "Timestamp: " << timestamp << ", EventType: ";
        ss << "Call Info: " << call_info.ToString() << ", ";
        ss << EventTypeToString(type);

        return ss.str();
    }

};

// Even not associated with identifier
struct AllocationEvent {
	AllocationRange allocation_info;
    EventInfo event_info;

    AllocationEvent(AllocationRange alloc_info, EventInfo event_info) : allocation_info(alloc_info), event_info(event_info) {}
    AllocationEvent(unsigned long start, unsigned long size, unsigned long timestamp, EventType type) : allocation_info(start, size), event_info(timestamp, type) {}
    AllocationEvent(unsigned long start, unsigned long size, unsigned long timestamp, unsigned long call_site, unsigned long call_no, EventType type) : allocation_info(start, size), event_info(timestamp, call_site, call_no, type) {}

    bool operator<(const AllocationEvent &other) const {
        return event_info.timestamp < other.event_info.timestamp; // Order by timestamp
    }

    boost::property_tree::ptree PtreeSerialize() const {
        boost::property_tree::ptree root;

        root.add_child("AllocationRange", allocation_info.PtreeSerialize());
        root.add_child("EventInfo", event_info.PtreeSerialize());

        return root;
    }

    string ToString() const {
        stringstream ss;
        ss << "AllocationRange: " << allocation_info.ToString() << ", EventInfo: " << event_info.ToString();
        return ss.str();
    }


    
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
    AllocationHistory(Allocation alloc_info, EventInfo initial_event);

    unsigned long GetTransferCount() const;
    unsigned long GetStartAddress() const;
    AllocationState GetState() const;
    CallTag GetAllocTag() const;
    Allocation GetAllocationInfo() const;

    EventInfo GetLatestEventInfo();

    void SubmitEvent(EventInfo event);

    boost::property_tree::ptree PtreeSerialize(bool verbose) const;

    string ToString() const {
        return ToString(false);
    }

    string ToString(bool verbose) const;

private:
    AllocationState CalculateNextState(EventType new_type);
    const EventInfo& GetLatestEventUnsafe();
    bool IsLatestEvent(const EventInfo& event);

    void SubmitEventUnsafe(EventInfo event);


private:
    // Protects all private data
    mutable shared_mutex m_alloc_mutex;
    Allocation alloc_info;
    // AllocationRange alloc_info; 
    // AllocationIdentifier alloc_tag; 
    AllocationState state; 
    unsigned long transfer_count; 
    multiset<EventInfo> events; 
};