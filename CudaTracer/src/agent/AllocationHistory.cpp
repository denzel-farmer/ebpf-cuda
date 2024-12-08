#include <assert.h>
#include <iostream>
#include <sstream>

#include "AllocationHistory.h"

#include "Logger.h"

using namespace std;


// boost::property_tree::ptree AllocationIdentifier::PtreeSerialize() const {
//     boost::property_tree::ptree root;
//     root.put("call_site", call_site);
//     root.put("call_no", call_no);

//     return root;
// }

// AllocationRange Implementations

// bool AllocationRange::operator<(const AllocationRange &other) const {
//     return start < other.start;
// }
// boost::property_tree::ptree AllocationRange::PtreeSerialize() const {
//     boost::property_tree::ptree root;
//     root.put("start", start);
//     root.put("size", size);

//     return root;
// }

// string AllocationRange::ToString() const {
//     stringstream ss;
//     ss << "Start: 0x" << hex << start << ", Size: 0x" << hex << size;
//     return ss.str();
// }

// EventInfo Implementations

// bool EventInfo::operator<(const EventInfo &other) const {
//     return timestamp < other.timestamp; // Order by timestamp
// }

// bool EventInfo::operator>(const EventInfo &other) const {
//     return timestamp > other.timestamp; // Order by timestamp
// }

// boost::property_tree::ptree EventInfo::PtreeSerialize() const {
//     boost::property_tree::ptree root;
//     root.put("timestamp", timestamp);
//     root.put("type", EventTypeToString(type));

//     return root;
// }
// string EventInfo::ToString() const {
//     stringstream ss;
//     ss << "Timestamp: " << timestamp << ", EventType: ";
//     ss << "Call Site: 0x" << hex << call_site << ", ";
//     ss << EventTypeToString(type);

//     return ss.str();
// }



// AllocationEvent Implementations

// bool AllocationEvent::operator<(const AllocationEvent &other) const {
//     return event_info.timestamp < other.event_info.timestamp; // Order by timestamp
// }

// boost::property_tree::ptree AllocationEvent::PtreeSerialize() const {
//     boost::property_tree::ptree root;

//     root.add_child("AllocationRange", allocation_info.PtreeSerialize());
//     root.add_child("EventInfo", event_info.PtreeSerialize());

//     return root;
// }

// string AllocationEvent::ToString() const {
//     stringstream ss;
//     ss << "AllocationRange: " << allocation_info.ToString() << ", EventInfo: " << event_info.ToString();
//     return ss.str();
// }


// AllocationHistory Implementations: tracks the history of a single allocation

// AllocationHistory::AllocationHistory(AllocationRange alloc_info, EventInfo initial_event, AllocationIdentifier alloc_tag)
// {
//     unique_lock<shared_mutex> lock(m_alloc_mutex);
//     this->alloc_info = alloc_info;
//     this->alloc_tag = alloc_tag;
//     transfer_count = 0;

//     SubmitEventUnsafe(initial_event);
// }



// AllocationHistory::AllocationHistory(AllocationRange alloc_info, EventInfo initial_event)
// {
//     this->alloc_info = alloc_info;
//     transfer_count = 0;
//     state = AllocationState::UNKOWN;
//     SubmitEvent(initial_event);
// }


AllocationHistory::AllocationHistory(Allocation alloc_info, EventInfo initial_event) {
    unique_lock<shared_mutex> lock(m_alloc_mutex);
    this->alloc_info = alloc_info;
    transfer_count = 0;
    state = AllocationState::UNKOWN;

    SubmitEventUnsafe(initial_event);
}

Allocation AllocationHistory::GetAllocationInfo() const {
    shared_lock<shared_mutex> lock(m_alloc_mutex);
    return alloc_info;
}

unsigned long AllocationHistory::GetTransferCount() const {
    shared_lock<shared_mutex> lock(m_alloc_mutex);
    return transfer_count;
}

unsigned long AllocationHistory::GetStartAddress() const {
    shared_lock<shared_mutex> lock(m_alloc_mutex);
    return alloc_info.range.start;
}

AllocationState AllocationHistory::GetState() const {
    shared_lock<shared_mutex> lock(m_alloc_mutex);
    return state;
}

CallTag AllocationHistory::GetAllocTag() const {
    shared_lock<shared_mutex> lock(m_alloc_mutex);
    return alloc_info.alloc_tag;
}

EventInfo AllocationHistory::GetLatestEventInfo() {
    shared_lock<shared_mutex> lock(m_alloc_mutex);
    return GetLatestEventUnsafe();
}

// Asumes at least reader lock held, for duration of EventInfo usage
const EventInfo& AllocationHistory::GetLatestEventUnsafe() {
    return *events.rbegin();
}

void AllocationHistory::SubmitEvent(EventInfo event) {
    unique_lock<shared_mutex> lock(m_alloc_mutex);
    SubmitEventUnsafe(event);
}

// Assumes write lock held
void AllocationHistory::SubmitEventUnsafe(EventInfo event) {
    // Only update state if event is the latest
    if (IsLatestEvent(event)) {
        state = CalculateNextState(event.type);
    }

    if (event.type == EventType::DEVICE_TRANSFER) {
        transfer_count++;
    }
    
    events.insert(event);
    globalLogger.log_debug("Event submitted: " + event.ToString());
}


// Assumes at least reader lock is held
AllocationState AllocationHistory::CalculateNextState(EventType new_type) {
    switch (new_type) {
        case EventType::ALLOC:
            // assert(state != AllocationState::ALLOCATED && "Memory already allocated");
            if (state == AllocationState::ALLOCATED) {
                globalLogger.log_warning("[AllocationHistory->CalculateNextState] Allocation event on already allocated memory");
            }

            return AllocationState::ALLOCATED;
            break;
        case EventType::FREE:
            if (state == AllocationState::FREED) {
                globalLogger.log_warning("[AllocationHistory->CalculateNextState] Free event on already freed memory");
            }

            return AllocationState::FREED;
            break;
        default:
            return state;
    }
}

// Assumes at least reader lock is held
bool AllocationHistory::IsLatestEvent(const EventInfo& event) {
    return events.empty() || (event > GetLatestEventUnsafe());
}


// JSON Serialization
boost::property_tree::ptree AllocationHistory::PtreeSerialize(bool verbose) const {
    boost::property_tree::ptree root;
    shared_lock<shared_mutex> lock(m_alloc_mutex);

    // Make a node for AllocationRange
    root.add_child("AllocationInfo", alloc_info.PtreeSerialize());   
    // root.add_child("AllocTag", alloc_tag.PtreeSerialize()); 
    root.put("current_state", AllocationStateToString(state));
    root.put("transfer_count", transfer_count);

    if (verbose) {
        // Add a single list of events to root
        boost::property_tree::ptree events_node;
        for (const auto& event : events) {
            boost::property_tree::ptree event_node;
            event_node.add_child("Event", event.PtreeSerialize());
            events_node.push_back(make_pair("", event_node));
        }
        root.add_child("Events", events_node);
    }

    return root;
}

string AllocationHistory::ToString(bool verbose) const {
    shared_lock<shared_mutex> lock(m_alloc_mutex);
    stringstream ss;
    ss << "AllocationRange: " << alloc_info.ToString() << ", State: ";
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