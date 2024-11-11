// cudamem.h

#ifndef __CUDAMEM_H
#define __CUDAMEM_H

struct data_t {
    void *dst;
    const void *src;
    size_t count;
    int kind;
};

#endif /* __CUDAMEM_H */
