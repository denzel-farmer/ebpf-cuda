#pragma once

#include <set>
#include <map>
#include <iostream>
#include <unordered_map>

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/mem_fun.hpp>

using namespace std;

// Uniquely describes an allocation for its entire lifetime
struct AllocationInfo {
	unsigned long start;
	unsigned long size;
	// Currently unused
	unsigned long identifier;
};

enum class EventType {
	ALLOC, // Allocation event, creates a new allocation
	HOST_TRANSFER, // Transfer event, moves data to another allocation on the host (e.g pinning)
	DEVICE_TRANSFER, // Transfer event, moves data to the device
	FREE // Free event, deallocates an allocation
};

struct EventInfo {
    unsigned long timestamp;
    EventType type;
};

struct AllocationEvent {
	AllocationInfo allocation_info;
    EventInfo event_info;

	bool operator<(const AllocationEvent &other) const
	{
		return event_info.timestamp < other.event_info.timestamp; // Order by timestamp
	}
};

// Should eventually have more states
enum class AllocationState { ALLOCATED, FREED, UNKOWN };

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

private:
    AllocationState CalculateNextState(EventType new_type);

public:
    AllocationInfo alloc_info;
    AllocationState state;
    unsigned long transfer_count;
    multiset<EventInfo> events;
};

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
    vector<const AllocationHistory*> GetColdspots(int max_transfers) const;

private:
    void UpdateHistories(AllocationInfo alloc_info, EventInfo event_info);

private:
    AllocationHistoryContainer histories;
};
