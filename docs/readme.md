# Overview
- Tool that uses eBPF to transparently measure CUDA memory transfers, 
for the purpose of identifying cantidates for pinning 
- Attaches to a target program (or the current program), and creates a heatmap of allocations and
device memory transfers
- Exposes two interfaces:
    - A live interface, that can indicate that an allocation should be replaced with a pinned one or removed from pinned memory
    - A database file for the program run, that an allocator can parse and use to decide whether to pin initially


## Tracing/Tracked information
- tracks a heatmap of memory over time, i.e. tracks lifecycle of each allocation (page?)
- Pin-alloc'd (like CudaHostAlloc) are alloc'd/pinned -> transfered -> freed 
    - Step 1: VMA created, DMA-able kernel pages allocated, vma remapped to kernel pages
    - Step 2: Transfered to device ??
    - Step 3: Freed  
- Non-pin-alloc'd (like mmap) are alloc'd -> transfered to pinned -> tranfered to device -> freed
    - Step 1: VMA created
    - Step 2: On write, pages allocated 
    - Step 3: On pin, pinnable pages alloc'd, copied to pinnable pages, remapped?
    - Step 4: Transfered to device 
    - Step 5: Free'd

## Hotspot/coldspot identification
- based on tracing, identifies pages? allocations? that are frequently transfered to device 
- hotspots are allocations that should probably be pinned now (one-shot) or should be pinned on the 
next run (multi-shot) 
- coldspots are spots that are hardly ever (or never) transfered to the device 

## Database: Multi-shot version
- Run a target program once, allocator uses default behavior (pin everything, or pin nothing)
- Tracer tracks allocation pattern, and identifies allocations that encounter frequent transfers
- On future runs, tracer advises allocator about which allocations to pin for transfer to the device
- Allocator should be able load in a tracing database 

### Problems
- How to uniquely identify allocations across multiple runs? Ordering might not be exactly the same
- might only work for our allocator, because we can have allocator add identification 

## Live interface
- Run target program, allocator initially maps everything pinned
- Tracer monitors allocations and transfers, and when transfers reach a threshold (maybe first time?)
then allocator should copy to pinned memory

# Evaluation
- Tool should have a 'tracing-only' mode that tracks allocations, and counts pinned pages used
- Can prove that we use way fewer pinned pages than pytorch
- Counting pins could be hard--do we consider total number of pins, or peak at any given time?
    - if doing peak, must also track unpins

# Limitations / Assumptions
- Limit to a small subset of functions to track, across categories: allocate, transfer, and free
- Track transfers to pinned memory, and transfers to device 
- Only care about to device, don't care about updates from the device

## Allocate functions
- Could only consider our custom allocation functions? But still want evaluation
- Could write uprobes for allocation, since it is always explicit
- Only care about allocations done with CudaHostAlloc, mmap (switch with malloc?), or our allocation function

## Transfer functions
- Only care about transfers made as a result of CudaMemcpy

## Free functions
- Only care about CudaHostFree


# Design
- In-kernel eBPF programs populate ringbuffers
- Tracer agent polls controls set of EventProbe threads, which stream AllocationEvents (via polling ringbuffers)
- Tracer agent maintains MemHistory by submitting each event in-order
- MemHistory takes a stream of events and stores them in memory, and exposes interface / dumps to disk
- HistoryParser can read database and present visualizations? 

## Event Probes
- EventProbes on various userspace and kernelspace functions
- Each EventProbe runs an independent thread, that polls and drains a kernel ringbuffer and populates a userspace queue
- Have installation and destruction
- Can be implemented either as direct callbacks or an eBPF function
    - Can eventprobes be agentless/installed as callback functions? 
- Delivers a stream of timestamped AllocationEvents 

## TracerAgent
- TracerAgent runs as a single thread, and waits until an event is pushed by EventProbes
- Delivers pushed AllocationEvent to the MemHistory it maintains 
- Tears down all EventProbes on cleanup
- Exposes 'live' interface to MemHistory

## AllocationEvent
- Timestamped event indication some allocation state changed
- For now, the state can be: Allocation, DeviceTransfer, or Free
- Static AllocationInfo identifies an allocation by start and length (and maybe id num?)
- Events contain structs AllocationInfo (start + len), and EventInfo (timestamp, new state)

## MemHistory
- Database of AllocationEvents 
- Simplest implementation can just keep a set of AllocationHistories, where each history 
is an AllocationInfo and an ordered vector EventInfo's
    - AllocationHistory should provide methods for NumTransfers 
- Hotspots returns list of 'n' active AllocationInfo structures with greatest transfer count
- Coldspots returns all AllocationInfo structures 
-  
- MemHistory can be serialized to a database file 
- MemHistory can provide history of each allocation 

### HistoryDatabase
- For now, can just be serialized JSON version of each Allocation + its events
- Effecient version would be compressed binary 

## Ideas about allocator
- Allocator could somehow register callback to move hotspots
- Version that is tightly integrated with training could checkpoint after every batch (or any
period known to have repeat allocations)

# Kernel Driver Notes

## Allocation via CudaHostAlloc
- Maybe RmP2PDmaMapPagesCoherent
    - os_alloc_mem
    - nv_dma_map_alloc

## Transfer to pinned memory 
- For now, only care about pins that happen right away (CudaHostAlloc)
- Eventually, should find underlying pin function (to catch RegisterHostAlloc?)
- RmP2PDmaMapPagesCoherent? 
- How does transfering work from unified memory?

## Transfered to device with CudaMemcpy


# Problems
- Can allocations merge? 
- How do we handle read only / copy on write