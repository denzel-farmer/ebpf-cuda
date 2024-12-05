#pragma once
#include <queue>
#include <mutex>
#include <optional>
#include <condition_variable>

using namespace std;

// Thread-safe queue implementation
template <typename T>
class ThreadSafeQueue {
public:
    void enqueue(T value);

    optional<T> dequeue_wait();

    void terminate();

private:
    queue<T> m_queue;
    mutable mutex m_mutex;
    condition_variable m_cond_var;
    bool m_done = false;
};
