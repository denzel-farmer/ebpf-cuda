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

#include <boost/property_tree/ptree.hpp>

#include "AllocationHistory.h"

// Optimized container for tracking allocation history
using namespace boost::multi_index;

struct by_start_address {};
struct by_transfer_count {};

typedef multi_index_container<
    AllocationHistory,
    indexed_by<
        // Primary key is start address, used for fast lookup
        ordered_non_unique<
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

typedef AllocationHistoryContainer::index<by_start_address>::type::iterator StartAddressIndexIterator;
typedef AllocationHistoryContainer::index<by_transfer_count>::type::iterator TransferCountIndexIterator;

class MemHistory {
public:
    MemHistory();
    void RecordEvent(AllocationEvent event, AllocationIdentifier identifier);
    void RecordEvent(AllocationEvent event);
    vector<const AllocationHistory*> GetHotspots(int num) const;
    vector<const AllocationHistory*> GetColdspots(unsigned long max_transfers) const;

    vector<const AllocationHistory*> GetAllocationHistories() const;

    string ToString() const {
        return ToString(false);
    }
    string ToString(bool verbose) const;

    boost::property_tree::ptree PtreeSerialize(bool verbose) const;
    void JSONSerialize(ostream& out, bool verbose) const;

private:
    void UpdateNewAlloc(AllocationRange alloc_info, EventInfo event_info, AllocationIdentifier identifier);
    void UpdateExistingAlloc(AllocationRange alloc_info, EventInfo event_info);

   pair<StartAddressIndexIterator, StartAddressIndexIterator> FindStartAddressAllocRange(unsigned long startAddress) const;
   optional<StartAddressIndexIterator> FindActiveAlloc(unsigned long startAddress) const;


private:
    AllocationHistoryContainer histories;
};
