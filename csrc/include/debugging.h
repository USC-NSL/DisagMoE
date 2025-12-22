/* Copyright (c) 2025 University of Southern California
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR(S) DISCLAIM ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL AUTHORS BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#ifndef DEBUGGING_H
#define DEBUGGING_H

#include <stdint.h>
#include <cstdio>
#include <queue>
#include <utility>
#include <functional>
#include <string>
#include <thread>
#include <memory>
#include <mutex>
#include <pthread.h>


/**
 * Structure to save each future timeout and callback.
 */
struct TimeoutEntry {
    uint64_t firetick;
    // The lambda function to execute on timeout
    std::function<void()> onTimeout;

    // Comparator to keep the smallest firetime at the top
    struct GreaterThan {
        bool operator()(const TimeoutEntry& lhs, const TimeoutEntry& rhs) const {
            return lhs.firetick > rhs.firetick;
        }
    };
};

/**
 * Tools to help debugging hangs.
 * It provides stack dump on timeout, logging messages, etc.
 */
class HangDebugger {

public:
    static void registerTimeoutForStackDump(pthread_t pthreadId, int timeoutInSec, std::string threadName);
    static void startMonThread(int device_id);
    static void terminate();

    template <typename... Args>
    int logMsg(const char *fmt, Args&&... args) {
        if (!_logFile) return 0; // Guard against null file pointer
        return fprintf(_logFile, fmt, args...);
    }

    /**
     * Same as logMsg, but flushes buffer before return.
     */
    template <typename... Args>
    int logfMsg(const char *fmt, Args&&... args) {
        if (!_logFile) return 0; // Guard against null file pointer
        int ret = fprintf(_logFile, fmt, args...);
        fflush(_logFile);
        return ret;
    }

    inline void logFlush() {
        fflush(_logFile);
    }

    ////////////////////////////////////////////////
    // Parameters and configs
    ////////////////////////////////////////////////
    static constexpr const char* logFilePathDefault{"stackdump.log"};
    static const int printInterval{60};
    static const int hangTimeoutBase{90};
    static inline int calcDumpTimeout(int device_id, int offset) {
        return hangTimeoutBase + 6 * device_id + offset;
    }

private:
    static HangDebugger* getDefault();
    void init(const char* pathToLogfile);
    void registerTimeout(int timeoutInSec, std::function<void()> action);
    bool _initialized{false};
    bool _end_flag{false};
    std::unique_ptr<std::thread> _monitor_thread;
    FILE* _logFile;
    uint64_t _startTick{0};
    // uint64_t _lastPollTick{0};
    std::priority_queue<TimeoutEntry, std::vector<TimeoutEntry>, TimeoutEntry::GreaterThan> _timeoutEntries;
    std::mutex _queueMutex;
};

#endif  // DEBUGGING_H