#include <assert.h>
#include <iostream>
#include <sstream>

#include "AllocationHistory.h"

using namespace std;


// AllocationInfo Implementations

bool AllocationInfo::operator<(const AllocationInfo &other) const {
    return start < other.start;
}

string AllocationInfo::ToString() const {
    stringstream ss;
    ss << "Start: " << start << ", Size: " << size << ", Identifier: " << identifier;
    return ss.str();
}

// EventInfo Implementations

bool EventInfo::operator<(const EventInfo &other) const {
    return timestamp < other.timestamp; // Order by timestamp
}

bool EventInfo::operator>(const EventInfo &other) const {
    return timestamp > other.timestamp; // Order by timestamp
}

string EventInfo::ToString() const {
    stringstream ss;
    ss << "Timestamp: " << timestamp << ", EventType: ";
    ss << EventTypeToString(type);

    return ss.str();
}



// AllocationEvent Implementations

bool AllocationEvent::operator<(const AllocationEvent &other) const {
    return event_info.timestamp < other.event_info.timestamp; // Order by timestamp
}

string AllocationEvent::ToString() const {
    stringstream ss;
    ss << "AllocationInfo: " << allocation_info.ToString() << ", EventInfo: " << event_info.ToString();
    return ss.str();
}


// AllocationHistory Implementations: tracks the history of a single allocation


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

    // Only update state if event is the latest
    if (IsLatestEvent(event)) {
        state = CalculateNextState(event.type);
    }

    if (event.type == EventType::DEVICE_TRANSFER) {
        transfer_count++;
    }
    
    events.insert(event);
    cout << "Event submitted: " << event.ToString() << endl;

}

AllocationState AllocationHistory::CalculateNextState(EventType new_type) {
    switch (new_type) {
        case EventType::ALLOC:
            assert(state != AllocationState::ALLOCATED && "Memory already allocated");
            return AllocationState::ALLOCATED;
            break;
        case EventType::FREE:
            return AllocationState::FREED;
            break;
        default:
            return state;
    }
}

bool AllocationHistory::IsLatestEvent(const EventInfo& event) const {
    return events.empty() || (event > GetLatestEvent());
}


string AllocationHistory::ToString(bool verbose) const {
    stringstream ss;
    ss << "AllocationInfo: " << alloc_info.ToString() << ", State: ";
    ss << AllocationStateToString(state);
    ss << ", TransferCount: " << transfer_count; 
    if (verbose) {
        ss << ", Events: [";
        for (const auto& event : events) {
            ss << "(" << event.ToString() << "), ";
        }
        ss << "]";
    }
    
    return ss.str();
}