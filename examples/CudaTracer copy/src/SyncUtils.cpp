#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>

#include "SyncUtils.h"

using namespace std;

template <typename T>
void ThreadSafeQueue<T>::enqueue(T value) {
        {
            lock_guard<mutex> lock(m_mutex);
            m_queue.push(move(value));
        }
        m_cond_var.notify_one();
    }

template <typename T>
std::optional<T> ThreadSafeQueue<T>::dequeue_wait() {
    unique_lock<mutex> lock(m_mutex);
    m_cond_var.wait(lock, [this]() { return !m_queue.empty() || m_done; });
    if (!m_queue.empty()) {
        T result = move(m_queue.front())
        m_queue.pop();
        return result;
    }

    return {};
}

template <typename T>
void ThreadSafeQueue<T>::terminate() {
    {
        lock_guard<mutex> lock(m_mutex);
        m_done = true;
    }
    m_cond_var.notify_all();
}