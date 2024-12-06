#include <assert.h>
#include <iostream>
#include <sstream>

#include "AllocationHistory.h"
#include "Logger.h"

using namespace std;


boost::property_tree::ptree AllocationIdentifier::PtreeSerialize() const {
    boost::property_tree::ptree root;
    root.put("call_site", call_site);
    root.put("call_no", call_no);

    return root;
}

// AllocationRange Implementations

bool AllocationRange::operator<(const AllocationRange &other) const {
    return start < other.start;
}
boost::property_tree::ptree AllocationRange::PtreeSerialize() const {
    boost::property_tree::ptree root;
    root.put("start", start);
    root.put("size", size);

    return root;
}

string AllocationRange::ToString() const {
    stringstream ss;
    ss << "Start: 0x" << hex << start << ", Size: 0x" << hex << size;
    return ss.str();
}

// EventInfo Implementations

bool EventInfo::operator<(const EventInfo &other) const {
    return timestamp < other.timestamp; // Order by timestamp
}

bool EventInfo::operator>(const EventInfo &other) const {
    return timestamp > other.timestamp; // Order by timestamp
}

boost::property_tree::ptree EventInfo::PtreeSerialize() const {
    boost::property_tree::ptree root;
    root.put("timestamp", timestamp);
    root.put("type", EventTypeToString(type));

    return root;
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

boost::property_tree::ptree AllocationEvent::PtreeSerialize() const {
    boost::property_tree::ptree root;

    root.add_child("AllocationRange", allocation_info.PtreeSerialize());
    root.add_child("EventInfo", event_info.PtreeSerialize());

    return root;
}

string AllocationEvent::ToString() const {
    stringstream ss;
    ss << "AllocationRange: " << allocation_info.ToString() << ", EventInfo: " << event_info.ToString();
    return ss.str();
}


// AllocationHistory Implementations: tracks the history of a single allocation

AllocationHistory::AllocationHistory(AllocationRange alloc_info, EventInfo initial_event, AllocationIdentifier alloc_tag)
{
    this->alloc_info = alloc_info;
    this->alloc_tag = alloc_tag;
    transfer_count = 0;

    SubmitEvent(initial_event);
}



// AllocationHistory::AllocationHistory(AllocationRange alloc_info, EventInfo initial_event)
// {
//     this->alloc_info = alloc_info;
//     transfer_count = 0;
//     state = AllocationState::UNKOWN;
//     SubmitEvent(initial_event);
// }

unsigned long AllocationHistory::GetTransferCount() const {
    return transfer_count;
}

unsigned long AllocationHistory::GetStartAddress() const {
    return alloc_info.start;
}

AllocationState AllocationHistory::GetState() const {
    return state;
}

AllocationIdentifier AllocationHistory::GetAllocTag() const {
    return alloc_tag;
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
   // cout << "Event submitted: " << event.ToString() << endl;

}

AllocationState AllocationHistory::CalculateNextState(EventType new_type) {
    switch (new_type) {
        case EventType::ALLOC:
            // assert(state != AllocationState::ALLOCATED && "Memory already allocated");
            if (state == AllocationState::ALLOCATED) {
                globalLogger.log_error("[AllocationHistory->CalculateNextState] Memory already allocated");
            }

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


// JSON Serialization
boost::property_tree::ptree AllocationHistory::PtreeSerialize(bool verbose) const {
    boost::property_tree::ptree root;

    // Make a node for AllocationRange
    root.add_child("AllocationRange", alloc_info.PtreeSerialize());   
    root.add_child("AllocTag", alloc_tag.PtreeSerialize()); 
    root.put("final_state", AllocationStateToString(state));
    root.put("transfer_count", transfer_count);

    if (verbose) {
        // Add a single list of events to root
        boost::property_tree::ptree events_node;
        for (const auto& event : events) {
            boost::property_tree::ptree event_node;
            event_node.put("Event", event.ToString());
            events_node.push_back(make_pair("", event_node));
        }
    }

    return root;
}

string AllocationHistory::ToString(bool verbose) const {
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