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

struct ProfileContext {
    std::string msg;

    float t_start;  // in ms
    float t_dur;    // in ms
};

class Recorder {
protected:
    std::map<std::thread::id, std::vector<ProfileContext>> ctx;

    // stack: [(time_stamp_ms, message)]
    std::map<std::thread::id, std::stack<std::pair<float, std::string>>> stack;
    std::shared_mutex mtx;

public:
    Recorder(const Recorder&) = delete;
    Recorder& operator=(const Recorder&) = delete;

    void create_thread() {
        auto tid = std::this_thread::get_id();
        std::lock_guard<std::shared_mutex> lock(mtx);
        ctx[tid] = std::vector<ProfileContext>();
        stack[tid] = std::stack<std::pair<float, std::string>>();
    }

    void push(const std::string &msg) {
        /*
            ! NOTE(hogura|20241226): we assume all threads are created initially, 
            ! which means the map are read-only during runtime, therefore no lock is required.
        */
        auto tid = std::this_thread::get_id();
        std::shared_lock<std::shared_mutex> lock(mtx);
        stack[tid].push(std::make_pair(1000.0 * clock() / CLOCKS_PER_SEC, msg));
    }

    void pop() {
        float ts = 1000.0 * clock() / CLOCKS_PER_SEC;
        auto tid = std::this_thread::get_id();
        auto top = stack[tid].top();

        std::shared_lock<std::shared_mutex> lock(mtx);
        stack[tid].pop();
        ctx.at(tid).push_back(ProfileContext{top.second, top.first, ts - top.first});
    }

    std::map<std::thread::id, std::vector<ProfileContext>> output() {
        /*
            ! NOTE(hogura|20241226): this function should only be called at the end of the program.
        */
        std::unique_lock<std::shared_mutex> lock(mtx);
        return ctx;
    }

    static recorder_t instance() {
        static recorder_t recorder = std::make_shared<Recorder>();
        return recorder;
    }

    static void create() {
        auto recorder = instance();
        recorder->create_thread();
    }

    static void push(const std::string &msg) {
        instance()->push(msg);
    }

    static void pop() {
        instance()->pop();
    }

    static std::map<std::thread::id, std::vector<ProfileContext>> output() {
        return instance()->output();
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