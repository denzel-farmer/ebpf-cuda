#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include "cudamem.h"

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
    struct CudaTransferEvent *event;
    event = bpf_ringbuf_reserve(&ringbuf, sizeof(struct CudaTransferEvent), 0);
    if (!event) {
        return 0;
    }

    // Extract arguements from pt_regs struct (Archtecture specific?)
    event->destination = (void *)PT_REGS_PARM1(ctx);
    event->source = (const void *)PT_REGS_PARM2(ctx);
    event->size = (size_t)PT_REGS_PARM3(ctx);
    event->direction = (int)PT_REGS_PARM4(ctx);

    // Populate process info
    populate_proc_info(&event->processInfo);

    bpf_ringbuf_submit(event, 0);

    return 0;
}

char LICENSE[] SEC("license") = "GPL";
