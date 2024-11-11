// cudamem_user.c

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <bpf/libbpf.h>
#include <errno.h>
#include <signal.h>
#include <fcntl.h>
#include <sys/resource.h>
#include <limits.h>
#include <string.h>
#include "cudamem.skel.h"
#include "cudamem.h"

static volatile sig_atomic_t exiting = 0;

void handle_sigint(int sig) {
    exiting = 1;
}

static int handle_event(void *ctx, void *data, size_t data_sz) {
    struct data_t *d = data;
    printf("cudaMemcpy called: dst=%p, src=%p, count=%zu, kind=%d\n",
           d->dst, d->src, d->count, d->kind);
    return 0;
}

int main(int argc, char **argv) {
    struct cudamem_bpf *skel;
    int err;
    pid_t target_pid;

    if (argc != 2) {
        fprintf(stderr, "Usage: %s <PID>\n", argv[0]);
        return 1;
    }

    target_pid = atoi(argv[1]);
    if (target_pid <= 0) {
        fprintf(stderr, "Invalid PID: %s\n", argv[1]);
        return 1;
    }

    // Open and load the BPF program
    skel = cudamem_bpf__open_and_load();
    if (!skel) {
        fprintf(stderr, "Failed to open and load BPF skeleton\n");
        return 1;
    }

    // Find the path to libcudart.so in the target process
    char maps_path[PATH_MAX];
    snprintf(maps_path, sizeof(maps_path), "/proc/%d/maps", target_pid);

    FILE *maps = fopen(maps_path, "r");
    if (!maps) {
        perror("Failed to open maps file");
        goto cleanup;
    }

    char line[4096];
    char libcudart_path[PATH_MAX] = {0};
    while (fgets(line, sizeof(line), maps)) {
        if (strstr(line, "libcudart.so")) {
            // Get the path at the end of the line
            char *path = strchr(line, '/');
            if (path) {
                strncpy(libcudart_path, path, sizeof(libcudart_path)-1);
                // Remove newline character
                libcudart_path[strcspn(libcudart_path, "\n")] = 0;
                break;
            }
        }
    }

    fclose(maps);

    if (!libcudart_path[0]) {
        fprintf(stderr, "Failed to find libcudart.so in target process\n");
        goto cleanup;
    }

    // Attach uprobe
    skel->links.handle_cudaMemcpy = bpf_program__attach_uprobe(
        skel->progs.handle_cudaMemcpy,
        false, // not an offset, use symbol name
        target_pid,
        libcudart_path,
        "cudaMemcpy"
    );
    if (!skel->links.handle_cudaMemcpy) {
        fprintf(stderr, "Failed to attach uprobe: %s\n", strerror(errno));
        goto cleanup;
    }

    printf("Attached uprobe to cudaMemcpy in process %d\n", target_pid);

    // Set up ring buffer
    struct ring_buffer *rb = NULL;
    rb = ring_buffer__new(bpf_map__fd(skel->maps.ringbuf), handle_event, NULL, NULL);
    if (!rb) {
        fprintf(stderr, "Failed to create ring buffer\n");
        goto cleanup;
    }

    signal(SIGINT, handle_sigint);

    // Poll ring buffer
    while (!exiting) {
        err = ring_buffer__poll(rb, 100 /* timeout, ms */);
        // Ctrl-C causes -EINTR
        if (err == -EINTR) {
            break;
        } else if (err < 0) {
            fprintf(stderr, "Error polling ring buffer: %d\n", err);
            break;
        }
    }

cleanup:
    ring_buffer__free(rb);
    cudamem_bpf__destroy(skel);
    return -err;
}
