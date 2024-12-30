#pragma once

#include "logging.h"
#include "constants.h"

#include <shared_mutex>
#include <thread>
#include <memory>
#include <vector>
#include <mutex>
#include <stack>
#include <ctime>
#include <map>

class Recorder;

typedef std::shared_ptr<Recorder> recorder_t;

struct TraceContext {
    std::string msg;

    double t_start;  // in ms
    double t_dur;    // in ms
    int track_id;
};

static double walltime_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

class Recorder {
protected:
    std::map<std::thread::id, std::vector<TraceContext>> ctx;

    // stack: [(time_stamp_ms, message)]
    std::map<std::thread::id, std::stack<std::pair<double, std::string>>> stack;
    
    // define in `muhelper.cpp`
    static std::shared_mutex mtx;
    static recorder_t instance;

public:

    Recorder() {}

    void create_thread() {
        auto tid = std::this_thread::get_id();
        DMOE_LOG(DEBUG) << "Creating thread " << tid << LEND;
        ASSERT(ctx.find(tid) == ctx.end());
        ctx[tid] = std::vector<TraceContext>();
        stack[tid] = std::stack<std::pair<double, std::string>>();
    }

    void push_(const std::string &msg) {
        /*
            ! NOTE(hogura|20241226): we assume all threads are created initially, 
            ! which means the map are read-only during runtime, therefore no lock is required.
        */
        auto tid = std::this_thread::get_id();
        stack.at(tid).push(std::make_pair(walltime_ms(), msg));
    }

    void pop_() {
        double ts = walltime_ms();
        auto tid = std::this_thread::get_id();
        auto top = stack.at(tid).top();

        stack.at(tid).pop();
        ctx.at(tid).push_back(TraceContext{top.second, top.first, ts - top.first, (int) stack.at(tid).size()});
    }

    std::map<size_t, std::vector<TraceContext>> output_() {
        /*
            ! NOTE(hogura|20241226): this function should only be called at the end of the program.
        */
        std::map<size_t, std::vector<TraceContext>> res;
        for (auto &pr: ctx) {
            DMOE_LOG(DEBUG) << "Thread " << pr.first << " has " << pr.second.size() << " records" << LEND;
            auto tid = std::hash<std::thread::id>{}(pr.first);
            res[tid] = pr.second;
        }
        return res;
    }

    static void create() {
        std::unique_lock lock(mtx);

        instance->create_thread();
    }

    static void push(const std::string &msg) {
        std::shared_lock lock(mtx);

        instance->push_(msg);
    }

    static void pop() {
        std::shared_lock lock(mtx);

        instance->pop_();
    }

    static std::map<size_t, std::vector<TraceContext>> output() {
        std::unique_lock lock(mtx);

        return instance->output_();
    }
};

class ScopedRange {

public:
    ScopedRange(ScopedRange&&) = delete;
    ScopedRange(const ScopedRange&) = delete;
    ScopedRange& operator=(const ScopedRange&) = delete;

    ScopedRange(const std::string &msg) {
        Recorder::push(msg);
    }

    ~ScopedRange() {
        Recorder::pop();
    }
};