#pragma once

#include <set>
#include <map>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <sstream>


#include <boost/multi_index_container.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/mem_fun.hpp>

#include "AllocationHistory.h"

// using namespace std;

// // Uniquely describes an allocation for its entire lifetime
// struct AllocationInfo {
// 	unsigned long start;
// 	unsigned long size;
// 	// Currently unused
// 	unsigned long identifier;

//     AllocationInfo() : start(0), size(0), identifier(0) {}
//     AllocationInfo(unsigned long s, unsigned long sz, unsigned long id) : start(s), size(sz), identifier(id) {}

//     bool operator<(const AllocationInfo &other) const {
//         return start < other.start;
//     }

//     string ToString() const {
//         stringstream ss;
//         ss << "Start: " << start << ", Size: " << size << ", Identifier: " << identifier;
//         return ss.str();
//     }
// };

// enum class EventType {
// 	ALLOC, // Allocation event, creates a new allocation
// 	HOST_TRANSFER, // Transfer event, moves data to another allocation on the host (e.g pinning)
// 	DEVICE_TRANSFER, // Transfer event, moves data to the device
// 	FREE // Free event, deallocates an allocation
// };

// constexpr const char* EventTypeToString(EventType type) {
//     switch (type) {
//         case EventType::ALLOC:
//             return "ALLOC";
//         case EventType::HOST_TRANSFER:
//             return "HOST_TRANSFER";
//         case EventType::DEVICE_TRANSFER:
//             return "DEVICE_TRANSFER";
//         case EventType::FREE:
//             return "FREE";
//         default:
//             return "UNKNOWN";
//     }
// }

// struct EventInfo {
//     unsigned long timestamp;
//     EventType type;

//     EventInfo(unsigned long ts, EventType et) : timestamp(ts), type(et) {}

//     bool operator<(const EventInfo &other) const
//     {
//         return timestamp < other.timestamp; // Order by timestamp
//     }
//     bool operator>(const EventInfo &other) const
//     {
//         return timestamp > other.timestamp; // Order by timestamp
//     }

//     string ToString() const {
//         stringstream ss;
//         ss << "Timestamp: " << timestamp << ", EventType: ";
//         ss << EventTypeToString(type);

//         return ss.str();
//     }
// };

// struct AllocationEvent {
// 	AllocationInfo allocation_info;
//     EventInfo event_info;

//     AllocationEvent(AllocationInfo alloc_info, EventInfo event_info) : allocation_info(alloc_info), event_info(event_info) {}
//     AllocationEvent(unsigned long start, unsigned long size, unsigned long identifier, unsigned long timestamp, EventType type) : allocation_info(start, size, identifier), event_info(timestamp, type) {}


// 	bool operator<(const AllocationEvent &other) const
// 	{
// 		return event_info.timestamp < other.event_info.timestamp; // Order by timestamp
// 	}

//     string ToString() const {
//         stringstream ss;
//         ss << "AllocationInfo: " << allocation_info.ToString() << ", EventInfo: " << event_info.ToString();
//         return ss.str();
//     }
    
// };

// // Should eventually have more states
// enum class AllocationState { ALLOCATED, FREED, UNKOWN };
// constexpr const char* AllocationStateToString(AllocationState state) {
//     switch (state) {
//         case AllocationState::ALLOCATED:
//             return "ALLOCATED";
//         case AllocationState::FREED:
//             return "FREED";
//         case AllocationState::UNKOWN:
//             return "UNKNOWN";
//         default:
//             return "UNKNOWN";
//     }
// }

// // Tracks the history of a single allocation
// // Current implementation is naive multiset of events (hotspot/coldspot optimization happens in the outer class)
// class AllocationHistory {
// public:
//     AllocationHistory(AllocationInfo alloc_info, EventInfo initial_event);

//     unsigned long GetTransferCount() const;

//     unsigned long GetStartAddress() const;

//     AllocationState GetState() const;

//     const EventInfo& GetLatestEvent() const;

//     void SubmitEvent(EventInfo event);

//     string ToString() const {
//         return ToString(false);
//     }

//     string ToString(bool verbose) const {
//         stringstream ss;
//         ss << "AllocationInfo: " << alloc_info.ToString() << ", State: ";
//         ss << AllocationStateToString(state);
//         ss << ", TransferCount: " << transfer_count; 
//         if (verbose) {
//             ss << ", Events: [";
//             for (const auto& event : events) {
//                 ss << "(" << event.ToString() << "), ";
//             }
//             ss << "]";
//         }
     
//         return ss.str();
//     }

// private:
//     AllocationState CalculateNextState(EventType new_type);
//     bool IsLatestEvent(const EventInfo& event) const;

// private:
//     AllocationInfo alloc_info;
//     AllocationState state;
//     unsigned long transfer_count;
//     multiset<EventInfo> events;
// };

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

class MemHistory {
public:
    MemHistory();

    void RecordEvent(AllocationEvent event);
    vector<const AllocationHistory*> GetHotspots(int num) const;
    vector<const AllocationHistory*> GetColdspots(unsigned long max_transfers) const;

    vector<const AllocationHistory*> GetAllocationHistories() const;

    string ToString() const {
        return ToString(false);
    }
    string ToString(bool verbose) const {
        stringstream ss;
        for (const auto& history : histories) {
            ss << history.ToString(verbose) << "\n";
        }
        return ss.str();
    }

private:
    void UpdateHistories(AllocationInfo alloc_info, EventInfo event_info);

private:
    AllocationHistoryContainer histories;
};
