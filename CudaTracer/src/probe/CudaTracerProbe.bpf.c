#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "CudaEvents.h"
// #include "eBPFProbe.h"
// #include "agent/AllocationHistory.h"

// constexpr unsigned long max_rb_entries = probe_ringbuf_size / sizeof(CudaMemcpyEvent);

// struct {
//     __uint(type, BPF_MAP_TYPE_RINGBUF);
//     __uint(max_entries, max_rb_entries); 
// } ringbuf SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1 << 24); // 16 MB ring buffer
} ringbuf SEC(".maps");


void populate_proc_info(struct CudaProcessInfo *processInfo)
{
    processInfo->pid = bpf_get_current_pid_tgid() >> 32;
}


SEC("uprobe/cudaMemcpy")
int handle_cudaMemcpy(struct pt_regs *ctx)
{
    struct CudaMemcpyEvent *event;

    event = bpf_ringbuf_reserve(&ringbuf, sizeof(struct CudaMemcpyEvent), 0);
    if (!event) {
        return 0;
    }

    // Extract arguements from pt_regs struct (Archtecture specific?)
    event->destination = (unsigned long)PT_REGS_PARM1(ctx);
    event->source = (unsigned long)PT_REGS_PARM2(ctx);
    event->size = (size_t)PT_REGS_PARM3(ctx);
    event->direction = (enum cudaMemcpyKind)PT_REGS_PARM4(ctx);
    event->timestamp = bpf_ktime_get_ns();

        // Get the return address
    void *sp;
    void *ret_addr = NULL;

    // Read stack pointer from pt_regs
    sp = (void *)ctx->sp;

    // Read the return address from the stack pointer
    if (sp) {
        bpf_probe_read_user(&ret_addr, sizeof(ret_addr), sp);
    }

    event->return_address = (unsigned long)ret_addr;
    
    // Get the return address (x86 specific)
    // ctx->sp gives the stack pointer. Dereference it to get the return address.
    // Get the return address safely using bpf_core_read
    // void *sp;
    // void *ret_addr;
    // bpf_core_read(&sp, sizeof(sp), &ctx->sp); // Read stack pointer from ctx
    // bpf_probe_read_user(&ret_addr, sizeof(ret_addr), sp); // Read return address
    bpf_trace_printk("SP: 0x%lx, Return Address: 0x%lx\n", (unsigned long)sp, (unsigned long)ret_addr);

   // bpf_trace_printk("SP: 0x%lx, Return Address: 0x%lx\n", (unsigned long)sp, (unsigned long)ret_addr);
    // Populate process info
    populate_proc_info(&event->processInfo);

    bpf_ringbuf_submit(event, 0);

    return 0;
}

char __license[] SEC("license") = "GPL";