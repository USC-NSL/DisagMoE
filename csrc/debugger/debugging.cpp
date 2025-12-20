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

#include <memory>
#include <chrono>
#include <cstring>
#include <cinttypes>
#include <csignal>
#include <execinfo.h>
#include <unistd.h>

#include "debugging.h"
#include "cycles.h"
#include "comm.h"

using Cycles = RAMCloud::Cycles;

// Used in signal handler. May be updated to the logging file.
int log_fd = 2;

std::unique_ptr<HangDebugger> _defaultDebugger;
std::mutex _defaultDebuggerLock;

// The Signal Handler to dump stack (Runs inside the hung thread)
// This handler will be triggered by the monitor thread when relevant timeout
// is fired.
void dump_stack_signal_handler(int signum) {
    // 1. Get the thread name
    char nameBuf[16]; // Linux limit is 16 bytes
    if (pthread_getname_np(pthread_self(), nameBuf, sizeof(nameBuf)) != 0) {
        const char* unknown = "Unknown";
        // Manual copy to avoid unsafe string functions
        int i = 0; while(unknown[i]) { nameBuf[i] = unknown[i]; i++; } 
        nameBuf[i] = '\0';
    }

    // 2. Safe Printing (Construct message manually)
    // Do NOT use fprintf or string operators here.
    const char* msg = "\n*** [SIGUSR1] Stack dump for thread: ";
    write(log_fd, msg, strlen(msg));
    write(log_fd, nameBuf, strlen(nameBuf));
    write(log_fd, " ***\n", 5);
    void *array[20];
    int size = backtrace(array, 20);
    backtrace_symbols_fd(array, size, log_fd);
    write(log_fd, "--- Stack Dump End ---\n", 23);
}

HangDebugger* HangDebugger::getDefault() {
    std::lock_guard lock(_defaultDebuggerLock);
    if (!_defaultDebugger) {
        _defaultDebugger = std::make_unique<HangDebugger>();
        _defaultDebugger->init(logFilePathDefault);
    }
    return _defaultDebugger.get();
}

void HangDebugger::init(const char* pathToLogfile) {
    std::lock_guard lock(_queueMutex);
    if (_initialized)
        return;
    _initialized = true;

    RAMCloud::Cycles::init();

    _logFile = fopen(pathToLogfile, "a");
    if (_logFile == nullptr) {
        // perror prints the actual system error (e.g., "Permission denied")
        fprintf(stderr, "Failed to open log file\n"); 
        return;
    }
    log_fd = fileno(_logFile);

    // 2. Register the signal handler
    struct sigaction sa;
    sa.sa_handler = dump_stack_signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGUSR1, &sa, NULL);
}

void HangDebugger::registerTimeoutForStackDump(pthread_t pthreadId, int timeoutInSec, std::string threadName) {
    HangDebugger* debugger = getDefault();
    debugger->registerTimeout(timeoutInSec, [pthreadId, debugger, threadName]() {
        int currentTime = int(Cycles::toSeconds(Cycles::rdtsc() - debugger->_startTick));
        debugger->logfMsg("%" PRIu64 " [%3d s] Stack trace for %s\n",
                Cycles::rdtsc(), currentTime, threadName.c_str());
        pthread_kill(pthreadId, SIGUSR1);
    });
}

void HangDebugger::startMonThread(int device_id) {
    HangDebugger* debugger = getDefault();
    if (debugger->_monitor_thread) {
        debugger->logMsg("[WARNING] The monitor thread has been started already.\n");
        return;
    }

    // This is a bit hacky way to avoid duplicate timeout registration.
    // Putting these here, instead of MuHelper::start().
    std::string mth_name = "MainThread@" + std::to_string(device_id);
    pthread_t mth_id = pthread_self();
    pthread_setname_np(mth_id, mth_name.c_str());
    int timeout4dump = HangDebugger::calcDumpTimeout(device_id, 4);
    registerTimeoutForStackDump(mth_id, timeout4dump, mth_name);

    // Now, start the monitoring thread which checks the registered timeouts.
    debugger->_monitor_thread = std::make_unique<std::thread>(
        [debugger, device_id]() {
            std::string th_name = "MonThread@" + std::to_string(device_id);
            pthread_setname_np(pthread_self(), th_name.c_str());
            
            debugger->logfMsg("***** Monitor thread has started.\n");
            debugger->_startTick = Cycles::rdtsc();
            while (!debugger->_end_flag) {
                uint64_t now = Cycles::rdtsc();
                {
                    // Check timeouts and fire callbacks.
                    std::lock_guard<std::mutex> lock(debugger->_queueMutex);
                    while (!debugger->_timeoutEntries.empty()) {
                        // Check the top (nearest) entry
                        if (now >= debugger->_timeoutEntries.top().firetick) {
                            // Execute the stored lambda
                            if (debugger->_timeoutEntries.top().onTimeout) {
                                debugger->_timeoutEntries.top().onTimeout();
                            }
                            debugger->_timeoutEntries.pop();
                        } else {
                            break; // Nearest hasn't expired, so no others have
                        }
                    }
                }
                // Sleep for 10 ms. Using sleep_for instead of spin wait in Cycles.
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            debugger->logfMsg("***** TESTTEST. Monitor thread is terminating.\n");
        }
    );
    
    // Register system state printing.
    // std::function<void()> printStatusAndRegisterNext =
    //     [debugger]() {
    //         NcclChannel::print_history(debugger->_logFile);
    //         // debugger->registerTimeout(printInterval, printStatusAndRegisterNext);
    //     };
    // debugger->registerTimeout(printInterval, printStatusAndRegisterNext);
}

void HangDebugger::terminate() {
    HangDebugger* debugger = getDefault();
    debugger->_end_flag = true;
    debugger->_monitor_thread->join();
}

/**
 * Registers a timeout with a custom action.
 * @param timeoutInSec Seconds from now when the action should fire.
 * @param action The lambda/function to execute.
 */
void HangDebugger::registerTimeout(int timeoutInSec, std::function<void()> action) {
    uint64_t firetick = Cycles::rdtsc() + Cycles::fromSeconds(timeoutInSec);

    std::lock_guard<std::mutex> lock(_queueMutex);
    _timeoutEntries.push({firetick, std::move(action)});
}