#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include "CudaEvents.h"

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1 << 24); // 16 MB ring buffer
} ringbuf SEC(".maps");


void populate_proc_info(struct CudaProcessInfo *processInfo)
{
    processInfo->pid = bpf_get_current_pid_tgid() >> 32;
}


SEC("kprobe/os_lock_user_pages")
int BPF_KPROBE(os_lock_user_pages, void *address, uint64_t page_count, void ** page_array, uint32_t flags)
{
    struct CudaPinPagesEvent *event;

    // TODO filter by PID here rather than in userspace (reduce event volume)

    event = bpf_ringbuf_reserve(&ringbuf, sizeof(struct CudaPinPagesEvent), 0);
    if (!event) {
        return 0;
    }

    event->address = (unsigned long)address;
    event->size = page_count*4096;
    event->timestamp = bpf_ktime_get_ns();

    // Get the return address
    void *sp;
    void *ret_addr = NULL;

    // Read stack pointer from pt_regs
    sp = (void *)PT_REGS_SP(ctx);

    // Read the return address from the stack pointer
    if (sp) {
        bpf_probe_read_user(&ret_addr, sizeof(ret_addr), sp);
    }

    event->return_address = (unsigned long)ret_addr;

    // Populate process info
    populate_proc_info(&event->processInfo);

    bpf_ringbuf_submit(event, 0);

    return 0;
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
    
    // Populate process info
    populate_proc_info(&event->processInfo);

    bpf_ringbuf_submit(event, 0);

    return 0;
}

SEC("uprobe/cudaFree")
int handle_cudaFree(struct pt_regs *ctx)
{
    struct GenericFreeEvent *event;

    event = bpf_ringbuf_reserve(&ringbuf, sizeof(struct GenericFreeEvent), 0);
    if (!event) {
        return 0;
    }

    // Extract arguements from pt_regs struct (Archtecture specific?)
    event->address = (unsigned long)PT_REGS_PARM1(ctx);
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

    // Populate process info
    populate_proc_info(&event->processInfo);

    bpf_ringbuf_submit(event, 0);

    return 0;
}

SEC("uprobe/cudaHostAlloc")
int handle_cudaHostAlloc(struct pt_regs *ctx)
{
    struct GenericAllocEvent *event;

    event = bpf_ringbuf_reserve(&ringbuf, sizeof(struct GenericAllocEvent), 0);
    if (!event) {
        return 0;
    }

    // Extract arguements from pt_regs struct (Archtecture specific?)
    event->address = (unsigned long)PT_REGS_PARM1(ctx);
    event->size = (size_t)PT_REGS_PARM2(ctx);
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

    // Populate process info
    populate_proc_info(&event->processInfo);

    bpf_ringbuf_submit(event, 0);

    return 0;
}


char __license[] SEC("license") = "GPL";