#pragma once

#include <iostream>

const std::string C_RESET = "\033[0m";
const std::string C_RED = "\033[31m";
const std::string C_GREEN = "\033[32m";
const std::string C_YELLOW = "\033[33m";
const std::string C_BLUE = "\033[34m";
const std::string C_MAGENTA = "\033[35m";
const std::string C_CYAN = "\033[36m";


enum LogLevel {
    INFO,
    WARNING,
    ERROR,
    CRITICAL
};

void log(LogLevel level, const std::string& message) {
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