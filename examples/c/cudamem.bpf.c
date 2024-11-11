#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include "cudamem.h"

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1 << 24); // 16 MB ring buffer
} ringbuf SEC(".maps");

SEC("uprobe/cudaMemcpy")
int handle_cudaMemcpy(struct pt_regs *ctx)
{
    struct data_t *data;
    void *dst = (void *)PT_REGS_PARM1(ctx);
    const void *src = (const void *)PT_REGS_PARM2(ctx);
    size_t count = (size_t)PT_REGS_PARM3(ctx);
    int kind = (int)PT_REGS_PARM4(ctx);

    data = bpf_ringbuf_reserve(&ringbuf, sizeof(struct data_t), 0);
    if (!data) {
        return 0;
    }
    data->dst = dst;
    data->src = src;
    data->count = count;
    data->kind = kind;

    bpf_ringbuf_submit(data, 0);

    return 0;
}

char LICENSE[] SEC("license") = "GPL";
