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

    bool enabled;
    
    // define in `muhelper.cpp`
    static std::shared_mutex mtx;
    static recorder_t instance;

public:

    Recorder(const char *enabled) {
        DMOE_LOG(DEBUG) << "Recorder enable status: " << enabled << LEND;
        this->enabled = enabled && strcmp(enabled, "1") == 0;
    }

    void create_thread() {
        if (!enabled)
            return;
        auto tid = std::this_thread::get_id();
        DMOE_LOG(DEBUG) << "Creating thread " << tid << LEND;
        if (ctx.find(tid) != ctx.end()) {
            DMOE_LOG(WARNING) << "Thread " << tid << " already exists" << LEND;
            return;
        }
        ctx[tid] = std::vector<TraceContext>();
        stack[tid] = std::stack<std::pair<double, std::string>>();
    }

    void push_(const std::string &msg) {
        /*
            ! NOTE(hogura|20241226): we assume all threads are created initially, 
            ! which means the map are read-only during runtime, therefore no lock is required.
        */
       if (!enabled)
           return;
        auto tid = std::this_thread::get_id();
        stack.at(tid).push(std::make_pair(walltime_ms(), msg));
    }

    void pop_() {
        if (!enabled)
            return;
        double ts = walltime_ms();
        auto tid = std::this_thread::get_id();
        auto top = stack.at(tid).top();

        stack.at(tid).pop();
        if ((ts - top.first) * 1000 > TRACER_GAP_US) // only time > gap is commited
            ctx.at(tid).push_back(TraceContext{top.second, top.first, ts - top.first, (int) stack.at(tid).size()});
    }

    std::map<size_t, std::vector<TraceContext>> output_() {
        /*
            ! NOTE(hogura|20241226): this function should only be called at the end of the program.
        */
       if (!enabled)
           return {};
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