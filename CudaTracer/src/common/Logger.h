#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <mutex>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <execinfo.h> // For backtrace functions
#include <cxxabi.h>   // For demangling C++ symbols

using namespace std;

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};


class Logger {
private:
    LogLevel logLevel;   // Log level for this instance
    mutex logMutex;      // Mutex for thread-safe logging
    string logFileName;  // Log file name

    // Get current time as a formatted string
    inline string getCurrentTime() const {
        auto now = chrono::system_clock::now();
        auto now_time_t = chrono::system_clock::to_time_t(now);
        ostringstream oss;
        oss << put_time(localtime(&now_time_t), "%Y-%m-%d %H:%M:%S");
        return oss.str();
    }

    inline string logLevelToString(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO: return "INFO";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}
    // Get the current function name using backtrace
    inline string getFunctionName() const {
        void* callstack[2]; // Limit to the immediate caller
        int frames = backtrace(callstack, 2);
        if (frames < 2) {
            return "UNKNOWN";
        }

        char** symbols = backtrace_symbols(callstack, frames);
        if (!symbols) {
            return "UNKNOWN";
        }

        string funcName = symbols[1]; // Caller function is at index 1
        free(symbols);

        // Demangle the function name for readability
        size_t start = funcName.find('(');
        size_t end = funcName.find('+');
        if (start != string::npos && end != string::npos) {
            string mangled = funcName.substr(start + 1, end - start - 1);
            int status;
            char* demangled = abi::__cxa_demangle(mangled.c_str(), nullptr, nullptr, &status);
            if (status == 0 && demangled) {
                funcName = demangled;
            }
            free(demangled);
        }

        return funcName;
    }

public:
    Logger(LogLevel level, const string& fileName) : logLevel(level), logFileName(fileName) {}

    // Set log level
    inline void setLogLevel(LogLevel level) {
        logLevel = level;
    }

    inline void log_debug(const string& message) {
        log(LogLevel::DEBUG, message);
    }

    inline void log_info(const string& message) {
        log(LogLevel::INFO, message);
    }

    inline void log_warning(const string& message) {
        log(LogLevel::WARNING, message);
    }

    inline void log_error(const string& message) {
        log(LogLevel::ERROR, message);
    }

    // Log function
    inline void log(LogLevel level, const string& message) {
        if (level < logLevel) {
            return;
        }

        ostringstream logEntry;
        logEntry << "[" << getCurrentTime() << "] "
                 << "[" << logLevelToString(level) << "] "
                 << "[" << getFunctionName() << "] "
                 << "[Thread " << this_thread::get_id() << "] "
                 << message;

        {
            lock_guard<mutex> lock(logMutex);

            // Print to standard output
            cout << logEntry.str() << endl;

            // Append to log file
            ofstream logFile(logFileName, ios::app);
            if (logFile.is_open()) {
                logFile << logEntry.str() << endl;
            }
        }
    }
};

// Global instance of Logger
extern Logger globalLogger;