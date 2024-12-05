// SyncUtils.h

#ifndef SYNCUTILS_H
#define SYNCUTILS_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>

template <typename T>
class ThreadSafeQueue {
public:
    void enqueue(T value);
    std::optional<T> dequeue_wait();
    void terminate();

private:
    std::queue<T> m_queue;
    std::mutex m_mutex;
    std::condition_variable m_cond_var;
    bool m_done = false;
};

// Definitions of template member functions

template <typename T>
void ThreadSafeQueue<T>::enqueue(T value) {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_queue.push(std::move(value));
    }
    m_cond_var.notify_one();
}

template <typename T>
std::optional<T> ThreadSafeQueue<T>::dequeue_wait() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_cond_var.wait(lock, [this]() { return !m_queue.empty() || m_done; });
    if (!m_queue.empty()) {
        T result = std::move(m_queue.front());
        m_queue.pop();
        return result;
    }
    return {};
}

template <typename T>
void ThreadSafeQueue<T>::terminate() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_done = true;
    }
    m_cond_var.notify_all();
}

#endif // SYNCUTILS_H
