#include <set>
#include <map>
#include <iostream>
#include <unordered_map>
#include <shared_mutex>

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/mem_fun.hpp>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <zlib.h>

using namespace std;
using namespace boost::multi_index;
#define BOOST_BIND_GLOBAL_PLACEHOLDERS

#include "MemHistory.h"
#include "Logger.h"

// Tracks the history of memory events
MemHistory::MemHistory() {
}

// Record an event given an identifier. If event is an alloc, creates a new history 
// entry. Currently, does not check that active allocation matches the identifier.
void MemHistory::RecordEvent(AllocationEvent event) {
    // TODO do edge case testing here (ex no overlapping duplicate allocs, no double frees)
	//cerr << "MemHistory Event call site: " << std::hex << identifier.call_site << ", call no: " << std::dec << identifier.call_no << endl;
    // TODO do merging/splitting here
    // TODO race condition for out of order events overlapping with create/free events
    globalLogger.log_debug("Recording event");
   
    if (event.event_info.type == EventType::ALLOC) {

        // If type is alloc, we can create an Allocation object with call tag as tag found in event
        Allocation new_alloc(event.allocation_info, event.event_info.call_info);

        unique_lock<shared_mutex> lock(m_history_mutex);
        UpdateNewAlloc(new_alloc, event.event_info);
    } else {


        unique_lock<shared_mutex> lock(m_history_mutex);
        UpdateExistingAlloc(event.allocation_info, event.event_info);
    }

}

// // Record a new memory event, without an identifier
// void MemHistory::RecordEvent(AllocationEvent event) {

//     // TODO do edge case testing here (ex no overlapping duplicate allocs, no double frees)

//     // TODO do merging/splitting here

//     // Check if there is an active allocation with the same start address
//     unique_lock<shared_mutex> lock(m_history_mutex);
//     auto active_alloc_opt = FindActiveAlloc(event.allocation_info.start);
//     if (!active_alloc_opt.has_value()) {
//         globalLogger.log_error("[MemHistory->RecordEvent] No active allocation found, skipping event");
        
//     }

//     UpdateExistingAlloc(event.allocation_info, event.event_info);
// }

vector<Allocation> MemHistory::GetHotspots(size_t num) const {
    vector<Allocation> hotspots;

    shared_lock<shared_mutex> lock(m_history_mutex);
    auto &index_by_transfer_count = m_histories.get<by_transfer_count>();

    auto it = index_by_transfer_count.begin();
    for (size_t i = 0; i < num && it != index_by_transfer_count.end(); i++, it++) {
        hotspots.push_back((*it)->GetAllocationInfo());
    }

    return hotspots;
}

vector<Allocation> MemHistory::GetColdspots(size_t num) const {
    vector<Allocation> coldspots;

    shared_lock<shared_mutex> lock(m_history_mutex);
    auto &index_by_transfer_count = m_histories.get<by_transfer_count>();

    auto it = index_by_transfer_count.rbegin();
    for (size_t i = 0; i < num && it != index_by_transfer_count.rend(); i++, it++) {
        coldspots.push_back((*it)->GetAllocationInfo());
    }

    return coldspots;
}

vector<Allocation> MemHistory::GetHotspotsThreshold(unsigned long min_transfers) const {
    vector<Allocation> hotspots;

    shared_lock<shared_mutex> lock(m_history_mutex);
    auto &index_by_transfer_count = m_histories.get<by_transfer_count>();

    auto it = index_by_transfer_count.begin();
    while (it != index_by_transfer_count.end()) {
        if ((*it)->GetTransferCount() >= min_transfers) {
            hotspots.push_back((*it)->GetAllocationInfo());
        }
        it++;
    }

    return hotspots;
}

vector<Allocation> MemHistory::GetColdspotsThreshold(unsigned long max_transfers) const {
    vector<Allocation> coldspots;

    shared_lock<shared_mutex> lock(m_history_mutex);
    auto &index_by_transfer_count = m_histories.get<by_transfer_count>();

    auto it = index_by_transfer_count.rbegin();
    while (it != index_by_transfer_count.rend()) {
        if ((*it)->GetTransferCount() < max_transfers) {
            coldspots.push_back((*it)->GetAllocationInfo());
        }
        it++;
    }

    return coldspots;
}

vector<Allocation> MemHistory::GetAllocations() const {
    vector<Allocation> allocations;

    shared_lock<shared_mutex> lock(m_history_mutex);
    for (const auto &history : m_histories) {
        allocations.push_back((*history).GetAllocationInfo());
    }

    return allocations;
}

// // Retrieve references to allocation history for n hotspots
// vector<const AllocationHistory*> MemHistory::GetHotspots(int num) const {
//     vector<const AllocationHistory*> hotspots;

//     shared_lock<shared_mutex> lock(m_history_mutex);
//     auto &index_by_transfer_count = m_histories.get<by_transfer_count>();

//     auto it = index_by_transfer_count.begin();
//     for (int i = 0; i < num && it != index_by_transfer_count.end(); i++, it++) {
//         hotspots.push_back(&(*it));
//     }

//     return hotspots;
// }

// Retrieve references to allocation history for all coldspots with fewer than max_transfers transfers
// vector<const AllocationHistory*> MemHistory::GetColdspots(unsigned long max_transfers) const {
//     vector<const AllocationHistory*> coldspots;

//     shared_lock<shared_mutex> lock(m_history_mutex);
//     auto &index_by_transfer_count = m_histories.get<by_transfer_count>();

//     auto it = index_by_transfer_count.rbegin();
//     while (it != index_by_transfer_count.rend() && it->GetTransferCount() < max_transfers) {
//         coldspots.push_back(&(*it));
//         it++;
//     }

//     return coldspots;
// }

// vector <const AllocationHistory*> MemHistory::GetAllocationHistories() const {
//     vector<const AllocationHistory*> all_histories;

//     shared_lock<shared_mutex> lock(m_history_mutex);
//     for (const auto &history : m_histories) {
//         all_histories.push_back(&history);
//     }

//     return all_histories;
// }



// Function to retrieve iterators of AllocationHistories with a given start address
// Expects at least reader lock to be held
pair<StartAddressIndexIterator, StartAddressIndexIterator> 
MemHistory::FindStartAddressAllocRange(unsigned long startAddress) const
{
    // Get the index by_start_address
    const auto& startAddressIndex = m_histories.get<by_start_address>();

    // Use equal_range to find the range of entries with the given start address
    return startAddressIndex.equal_range(startAddress);
}

// Function to find the active allocation with a given start address (should be only one)
// Expects at least reader lock to be held
optional<StartAddressIndexIterator> MemHistory::FindActiveAlloc(AllocationRange alloc_info) const {
    auto matching_allocs = FindStartAddressAllocRange(alloc_info.start);

    for (auto it = matching_allocs.first; it != matching_allocs.second; it++) {
        if ((*it)->GetState() == AllocationState::ALLOCATED) {
            // Debugging check that the range matches (TODO handle spliting)
            if ((*it)->GetAllocationInfo().range != alloc_info) {
                globalLogger.log_warning("Warning: Active allocation range does not match event range");
            }
            return optional<StartAddressIndexIterator>(it);
        }
    }
    // No active allocation found
    return {};
}


// // Update functions assume writer lock is already acquired
// // Expects writer lock to be held
// void MemHistory::UpdateExistingAlloc(AllocationRange alloc_info, EventInfo event_info) {
//     // Find the active allocation with the given start address 
//     auto active_alloc_opt = FindActiveAlloc(alloc_info.start);
//     // If it doesn't exist log an error and do nothing
//     if (!active_alloc_opt.has_value()) {
//         globalLogger.log_error("Error: UpdateExistingAlloc called on non-existent allocation");
//         return;
//     }
//     auto active_alloc = active_alloc_opt.value();

//     auto &index_by_start = m_histories.get<by_start_address>();
//     if (active_alloc == index_by_start.end()) {
//         // No active allocation, log error 
//         globalLogger.log_error("UpdateExistingAlloc called on non-existent allocation");
//     } else {
//         // Allocation exists, update safely 
//         index_by_start.modify(active_alloc, [&](AllocationHistory &alloc) {
//             alloc.SubmitEvent(event_info);
//         });
//     }
  
// }

// Expect writer lock to be held
void MemHistory::UpdateNewAlloc(Allocation new_alloc, EventInfo event_info) {
    
    // Should not be any other active allocs with the same start address. Check and log error for debugging
    auto active_alloc_opt = FindActiveAlloc(new_alloc.range);
    if (active_alloc_opt.has_value()) {
        globalLogger.log_error("Error: UpdateNewAlloc called on existing allocation"); // TODO handle splitting
    }
    // Create a new alloc history and insert in history

    shared_ptr<AllocationHistory> new_shared_alloc = make_shared<AllocationHistory>(new_alloc, event_info);
    m_histories.insert(new_shared_alloc);
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


// Update the history with a new event (does not create allocation)
// Expects write lock to be held
void MemHistory::UpdateExistingAlloc(AllocationRange alloc_info, EventInfo event_info) {

    // Search for current allocation based on start address
    auto active_alloc_opt = FindActiveAlloc(alloc_info);

    // If the found allocation doesn't exist, then we haven't seen an alloc event for this address (TODO handle splitting)
    if (!active_alloc_opt.has_value()) {
        globalLogger.log_error("[MemHistory->RecordEvent] Active allocation not found, skipping event");
        return;
    }

    auto active_alloc = active_alloc_opt.value();

    // If the range of the active allocation doesn't match the range of the event, then alloc is split (TODO handle)
    if ((*active_alloc)->GetAllocationInfo().range != alloc_info) {
        globalLogger.log_error("[MemHistory->RecordEvent] Active allocation range does not match event range");
        return;
    }

    // Now, have found active allocation associated with the event, can update the history
    
    auto &index_by_start = m_histories.get<by_start_address>();
    if (active_alloc == index_by_start.end()) {
        // No active allocation, log error 
        globalLogger.log_error("UpdateExistingAlloc called on non-existent allocation");
    } else {
        // Allocation exists, update safely 
        index_by_start.modify(active_alloc, [&](shared_ptr<AllocationHistory> &alloc) {
            (*alloc).SubmitEvent(event_info);
        });
    }
}


 boost::property_tree::ptree MemHistory::PtreeSerialize(bool verbose) const {
    boost::property_tree::ptree root;
    boost::property_tree::ptree allocationsNode;

    shared_lock<shared_mutex> lock(m_history_mutex);
    for (const auto& history : m_histories) {
        allocationsNode.push_back(std::make_pair("", (*history).PtreeSerialize(verbose)));
    }
    root.add_child("Allocations", allocationsNode);

    return root;
 }

 void MemHistory::JSONSerialize(ostream& out, bool verbose) const {
    boost::property_tree::ptree root = PtreeSerialize(verbose);
    boost::property_tree::write_json(out, root);
 }

void MemHistory::BinarySerialize(std::ostream& out, bool verbose) const {
        std::ostringstream jsonStream;
        JSONSerialize(jsonStream, verbose);
        std::string jsonData = jsonStream.str();

        uLongf compressedSize = compressBound(jsonData.size());
        std::string compressedData(compressedSize, '\0');

        if (compress(reinterpret_cast<Bytef*>(&compressedData[0]), &compressedSize,
                     reinterpret_cast<const Bytef*>(jsonData.data()), jsonData.size()) != Z_OK) {
            throw std::runtime_error("Failed to compress data");
        }

        compressedData.resize(compressedSize);
        out.write(compressedData.data(), compressedData.size());
        if (!out) {
            throw std::runtime_error("Failed to write compressed data to stream");
        }
}

void MemHistory::SaveDatabase(const char* filename, DumpFormat format, bool verbose) const {
    ofstream out(filename, ios::binary);
    if (!out.is_open()) {
        globalLogger.log_error("Failed to open file for writing");
        return;
    }

    switch (format) {
        case DumpFormat::JSON:
            JSONSerialize(out, verbose);
            break;
        case DumpFormat::BINARY:
            BinarySerialize(out, verbose);
            break;
        default:
            globalLogger.log_error("Invalid dump format");
            break;
    }

    out.close();
}


string MemHistory::ToString(bool verbose) const {
    stringstream ss;
    
    shared_lock<shared_mutex> lock(m_history_mutex);
    for (const auto& history : m_histories) {
        ss << (*history).ToString(verbose) << "\n";
    }
    return ss.str();
}

