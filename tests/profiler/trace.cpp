#include "profiler.hpp"
#include <thread>
#include <vector>
#include <cstdio>

int main() {
    int n = 10;
    std::vector<std::thread> threads;
    for (int i = 0; i < n; i ++) {
        threads.emplace_back(std::thread([i]() {
            printf("thread %d start\n", i);
            Recorder::create();

            printf("thread %d push\n", i);
            Recorder::push("thread " + std::to_string(i));
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            printf("thread %d pop\n", i);
            Recorder::pop();
        }));
    }

    for (auto &t: threads)
        t.join();

    auto output = Recorder::output();
    return 0;
}