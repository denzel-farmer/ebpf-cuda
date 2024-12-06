#include <set>
#include <map>
#include <iostream>
#include <unordered_map>

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/mem_fun.hpp>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

using namespace std;
using namespace boost::multi_index;

#include "MemHistory.h"
#include "Logger.h"

// Tracks the history of memory events
MemHistory::MemHistory() {
}

// Record an event given an identifier. If event is an alloc, creates a new history 
// entry. Currently, does not check that active allocation matches the identifier.
void MemHistory::RecordEvent(AllocationEvent event, AllocationIdentifier identifier) {
    // TODO do edge case testing here (ex no overlapping duplicate allocs, no double frees)
	//cerr << "MemHistory Event call site: " << std::hex << identifier.call_site << ", call no: " << std::dec << identifier.call_no << endl;
    // TODO do merging/splitting here
    // TODO race condition for out of order events overlapping with create/free events
    globalLogger.log_debug("Recording event");
    // If type is alloc, create a new history entry
    if (event.event_info.type == EventType::ALLOC) {
        UpdateNewAlloc(event.allocation_info, event.event_info, identifier);
    } else {

        // For debugging, check if the active allocation matches the identifier
        auto active_alloc_opt = FindActiveAlloc(event.allocation_info.start);
        if (active_alloc_opt.has_value()) {
            auto active_alloc = active_alloc_opt.value();
            if (active_alloc->GetAllocTag() != identifier) {
                globalLogger.log_error("[MemHistory->RecordEvent] Active allocation does not match identifier");
            }
        } else {
            globalLogger.log_error("[MemHistory->RecordEvent] Error: Active allocation not found, skipping event");
            return;
        }
        
        UpdateExistingAlloc(event.allocation_info, event.event_info);
    }

}

// Record a new memory event, without an identifier
void MemHistory::RecordEvent(AllocationEvent event) {

    // TODO do edge case testing here (ex no overlapping duplicate allocs, no double frees)

    // TODO do merging/splitting here

    // Check if there is an active allocation with the same start address
    auto active_alloc_opt = FindActiveAlloc(event.allocation_info.start);
    if (!active_alloc_opt.has_value()) {
        globalLogger.log_error("[MemHistory->RecordEvent] No active allocation found, skipping event");
        
    }

    UpdateExistingAlloc(event.allocation_info, event.event_info);
}

// Retrieve references to allocation history for n hotspots
vector<const AllocationHistory*> MemHistory::GetHotspots(int num) const {
    vector<const AllocationHistory*> hotspots;
    auto &index_by_transfer_count = histories.get<by_transfer_count>();

    auto it = index_by_transfer_count.begin();
    for (int i = 0; i < num && it != index_by_transfer_count.end(); i++, it++) {
        hotspots.push_back(&(*it));
    }

    return hotspots;
}

// Retrieve references to allocation history for all coldspots with fewer than max_transfers transfers
vector<const AllocationHistory*> MemHistory::GetColdspots(unsigned long max_transfers) const {
    vector<const AllocationHistory*> coldspots;
    auto &index_by_transfer_count = histories.get<by_transfer_count>();

    auto it = index_by_transfer_count.rbegin();
    while (it != index_by_transfer_count.rend() && it->GetTransferCount() < max_transfers) {
        coldspots.push_back(&(*it));
        it++;
    }

    return coldspots;
}

vector <const AllocationHistory*> MemHistory::GetAllocationHistories() const {
    vector<const AllocationHistory*> all_histories;
    for (const auto &history : histories) {
        all_histories.push_back(&history);
    }

    return all_histories;
}



// Function to retrieve iterators of AllocationHistories with a given start address
pair<StartAddressIndexIterator, StartAddressIndexIterator> 
MemHistory::FindStartAddressAllocRange(unsigned long startAddress) const
{
    // Get the index by_start_address
    const auto& startAddressIndex = histories.get<by_start_address>();

    // Use equal_range to find the range of entries with the given start address
    return startAddressIndex.equal_range(startAddress);
}

// Function to find the active allocation with a given start address (should be only one)
optional<StartAddressIndexIterator> MemHistory::FindActiveAlloc(unsigned long startAddress) const {
    auto range = FindStartAddressAllocRange(startAddress);
    for (auto it = range.first; it != range.second; it++) {
        if (it->GetState() == AllocationState::ALLOCATED) {
            return optional<StartAddressIndexIterator>(it);
        }
    }
    return {};
}



void MemHistory::UpdateExistingAlloc(AllocationRange alloc_info, EventInfo event_info) {
    // Find the active allocation with the given start address 
    auto active_alloc_opt = FindActiveAlloc(alloc_info.start);
    // If it doesn't exist log an error and do nothing
    if (!active_alloc_opt.has_value()) {
        globalLogger.log_error("Error: UpdateExistingAlloc called on non-existent allocation");
        return;
    }
    auto active_alloc = active_alloc_opt.value();

    auto &index_by_start = histories.get<by_start_address>();
    if (active_alloc == index_by_start.end()) {
        // No active allocation, log error 
        globalLogger.log_error("UpdateExistingAlloc called on non-existent allocation");
    } else {
        // Allocation exists, update safely 
        index_by_start.modify(active_alloc, [&](AllocationHistory &alloc) {
            alloc.SubmitEvent(event_info);
        });
    }
  
}


void MemHistory::UpdateNewAlloc(AllocationRange alloc_info, EventInfo event_info, AllocationIdentifier identifier) {
    
    // Should not be any other active allocs with the same start address. Check and log error for debugging
    auto active_alloc_opt = FindActiveAlloc(alloc_info.start);
    if (active_alloc_opt.has_value()) {
        globalLogger.log_error("Error: UpdateNewAlloc called on existing allocation");
    }
    // Create a new alloc history and insert in history
    AllocationHistory new_alloc(alloc_info, event_info, identifier);
    histories.insert(move(new_alloc));
    
    // If no allocation in container, create a new one 
    //auto &index_by_start = histories.get<by_start_address>();
    // auto it = index_by_start.find(alloc_info.start);

    // if (it == index_by_start.end()) {
    //     // Allocation does not exist, create a new one
    //     AllocationHistory new_alloc(alloc_info, event_info, identifier);
    //     histories.insert(move(new_alloc));
    // } else {
    //     // Allocation exists, update safely 
    //     index_by_start.modify(it, [&](AllocationHistory &alloc) {
    //         alloc.SubmitEvent(event_info);
    //     });
    // }
}

 boost::property_tree::ptree MemHistory::PtreeSerialize(bool verbose) const {
    boost::property_tree::ptree root;

    boost::property_tree::ptree allocationsNode;
    for (const auto& history : histories) {
        allocationsNode.push_back(std::make_pair("", history.PtreeSerialize(verbose)));
    }
    root.add_child("Allocations", allocationsNode);
    return root;
 }

 void MemHistory::JSONSerialize(ostream& out, bool verbose) const {
    boost::property_tree::ptree root = PtreeSerialize(verbose);
    boost::property_tree::write_json(out, root);
 }


string MemHistory::ToString(bool verbose) const {
    stringstream ss;
    for (const auto& history : histories) {
        ss << history.ToString(verbose) << "\n";
    }
    return ss.str();
}

