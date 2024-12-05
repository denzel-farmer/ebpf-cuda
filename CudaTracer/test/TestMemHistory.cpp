#include "MemHistory.h"
#include <fstream>


void AddSimpleHistory(MemHistory& memHistory, unsigned long start, unsigned long size){
    // Create AllocationEvent
    EventInfo eventInfo = EventInfo(0, EventType::ALLOC);
    AllocationRange allocInfo = AllocationRange(start, size);
    AllocationEvent allocEvent = AllocationEvent(allocInfo, eventInfo);

    // Record an event
    memHistory.RecordEvent(allocEvent);
    cout << memHistory.ToString(true) << endl;

    AllocationEvent hostTransfer = AllocationEvent(start, size, 9, EventType::HOST_TRANSFER);
    memHistory.RecordEvent(hostTransfer);
    cout << memHistory.ToString(true) << endl;

    AllocationEvent deviceTransfer = AllocationEvent(start, size, 18, EventType::DEVICE_TRANSFER);
    memHistory.RecordEvent(deviceTransfer);
    cout << memHistory.ToString(true) << endl;

    hostTransfer = AllocationEvent(start, size, 11, EventType::HOST_TRANSFER);
    memHistory.RecordEvent(hostTransfer);
    cout << memHistory.ToString(true) << endl;

}

void AddNumTransfersHistory(MemHistory& memHistory, unsigned long start, unsigned long size, unsigned long transfers) {

    AllocationEvent allocEvent = AllocationEvent(start, size, 0, EventType::ALLOC);
    memHistory.RecordEvent(allocEvent);

    for (unsigned long i = 0; i < transfers; i++) {
        AllocationEvent deviceTransfer = AllocationEvent(start, size, i, EventType::DEVICE_TRANSFER);
        memHistory.RecordEvent(deviceTransfer);
    }

    cout << memHistory.ToString() << endl;
}

int main() {

    MemHistory memHistory;

    AddSimpleHistory(memHistory, 0xDF, 100);

    AddSimpleHistory(memHistory, 0x100, 100);

    AddNumTransfersHistory(memHistory, 0x200, 100, 10);
    AddNumTransfersHistory(memHistory, 0x201, 100, 15);
    AddNumTransfersHistory(memHistory, 0x202, 100, 1);
    AddNumTransfersHistory(memHistory, 0x203, 100, 2);
    

    auto hotspots = memHistory.GetHotspots(2);
    cout << "Hotspots: " << endl;
    for (const auto& hotspot : hotspots) {
        cout << hotspot->ToString() << endl;
    }

    auto coldspots = memHistory.GetColdspots(5);
    cout << "Coldspots: " << endl;
    for (const auto& coldspot : coldspots) {
        cout << coldspot->ToString() << endl;
    }

    // Write to alloc-log.json
    ofstream out("alloc-log.json");
    memHistory.JSONSerialize(out, true);
    out.close();



    return 0;
}