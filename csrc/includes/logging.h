#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <thread>

const std::string C_RESET = "\033[0m";
const std::string C_RED = "\033[31m";
const std::string C_GREEN = "\033[32m";
const std::string C_YELLOW = "\033[33m";
const std::string C_BLUE = "\033[34m";
const std::string C_MAGENTA = "\033[35m";
const std::string C_CYAN = "\033[36m";


enum LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    CRITICAL
};

const static std::vector<std::string> COLOR_MAP = {
    C_BLUE + "[DEBUG]",
    C_GREEN + "[INFO]",
    C_YELLOW + "[WARNING]",
    C_RED + "[ERROR]",
    C_MAGENTA + "[CRITICAL]"
};

static void log(LogLevel level, const std::string& message) {
    switch (level) {
        case INFO:
            std::cerr << C_GREEN << "[INFO] " << message << C_RESET << std::endl;
            break;
        case WARNING:
            std::cerr << C_YELLOW << "[WARNING] " << message << C_RESET << std::endl;
            break;
        case ERROR:
            std::cerr << C_RED << "[ERROR] " << message << C_RESET << std::endl;
            break;
        case CRITICAL:
            std::cerr << C_MAGENTA << "[CRITICAL] " << message << C_RESET << std::endl;
            break;
        default:
            std::cerr << message << std::endl;
            break;
    }
}

#define LOG(LEVEL) std::cerr << COLOR_MAP[int(LEVEL)] << "" \
                             << "(" << std::this_thread::get_id() << ")" \
                             << " - " << __FILE__ ":" << __LINE__ \
                             << "@" << __FUNCTION__ \
                             << ">: " << C_RESET
#define LEND std::endl