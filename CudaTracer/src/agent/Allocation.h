#pragma once

#include <boost/property_tree/ptree.hpp>
#include <string>
#include <iostream>
#include <sstream>

using namespace std;

// Containers describing allocations, with no concept of state / events 


// Unique identifier for an allocation, based on the call site and call number
struct CallTag {
    unsigned long call_site;
    unsigned long call_no;

    CallTag() : call_site(0), call_no(0) {}
    CallTag(unsigned long site, unsigned long no) : call_site(site), call_no(no) {}


    boost::property_tree::ptree PtreeSerialize() const {
        boost::property_tree::ptree root;
        root.put("call_site", call_site);
        root.put("call_no", call_no);

        return root;
    }

    bool operator==(const CallTag &other) const {
        return call_site == other.call_site && call_no == other.call_no;
    }

    bool operator!=(const CallTag &other) const {
        return !(*this == other);
    }

    string ToString() const {
        stringstream ss;
        ss << "CallSite: 0x" << hex << call_site << ", CallNo: " << dec << call_no;
        return ss.str();
    }

};


// Uniquely describes an allocation for its entire lifetime
struct AllocationRange {
    unsigned long start;
    unsigned long size;

    AllocationRange() : start(0), size(0) {}
    AllocationRange(unsigned long s, unsigned long sz) : start(s), size(sz) {}

    bool operator<(const AllocationRange &other) const {
        return start < other.start;
    }
    
    bool operator==(const AllocationRange &other) const {
        return start == other.start && size == other.size;
    }

    bool operator!=(const AllocationRange &other) const {
        return !(*this == other);
    }

    boost::property_tree::ptree PtreeSerialize() const {
        boost::property_tree::ptree root;
        root.put("start", start);
        root.put("size", size);

        return root;
    }

    string ToString() const {
        stringstream ss;
        ss << "Start: 0x" << hex << start << ", Size: 0x" << hex << size;
        return ss.str();
    }

};


struct Allocation {
    AllocationRange range;
    CallTag alloc_tag;

    Allocation(unsigned long start, unsigned long size, unsigned long call_site, unsigned long call_no) : range(start, size), alloc_tag(call_site, call_no) {}
    Allocation(AllocationRange alloc_range, CallTag alloc_tag) : range(alloc_range), alloc_tag(alloc_tag) {}
    Allocation() : range(0, 0), alloc_tag(0, 0) {}
    
    boost::property_tree::ptree PtreeSerialize() const {
        boost::property_tree::ptree root;

        root.add_child("AllocationRange", range.PtreeSerialize());
        root.add_child("AllocationTag", alloc_tag.PtreeSerialize());

        return root;
    }
    string ToString() const {
        stringstream ss;
        ss << "AllocationRange: " << range.ToString() << ", AllocationIdentifier: " << alloc_tag.ToString();
        return ss.str();
    }
};

