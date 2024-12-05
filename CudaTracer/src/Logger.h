#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <mutex>
#include <sstream>
#include <chrono>
#include <iomanip>

using namespace std;


enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

inline string logLevelToString(LogLevel level) {
        switch (level) {
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO: return "INFO";
            case LogLevel::WARNING: return "WARNING";
            case LogLevel::ERROR: return "ERROR";
            default: return "UNKNOWN";
        }
}



class Logger {
private:
    static LogLevel globalLogLevel;   // Global log level
    static mutex logMutex;       // Mutex for thread-safe logging
    static const string logFileName; // Log file name

    // Get current time as a formatted string
    static inline string getCurrentTime() {
        auto now = chrono::system_clock::now();
        auto now_time_t = chrono::system_clock::to_time_t(now);
        ostringstream oss;
        oss << put_time(localtime(&now_time_t), "%Y-%m-%d %H:%M:%S");
        return oss.str();
    }

public:
    // Set global log level
    static inline void setLogLevel(LogLevel level) {
        globalLogLevel = level;
    }

    static inline void log_debug(const string& message, const char* functionName = __func__) {
        log(LogLevel::DEBUG, message, functionName);
    }

    static inline void log_info(const string& message, const char* functionName = __func__) {
        log(LogLevel::INFO, message, functionName);
    }

    static inline void log_warning(const string& message, const char* functionName = __func__) {
        log(LogLevel::WARNING, message, functionName);
    }

    static inline void log_error(const string& message, const char* functionName = __func__) {
        log(LogLevel::ERROR, message, functionName);
    }

    // Log function
    static inline void log(LogLevel level, const string& message, const char* functionName = __func__) {
        if (level < globalLogLevel) {
            return;
        }

        ostringstream logEntry;
        logEntry << "[" << getCurrentTime() << "] "
                 << "[" << logLevelToString(level) << "] "
                 << "[" << functionName << "] "
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

// Static member initialization
LogLevel Logger::globalLogLevel = LogLevel::DEBUG;
mutex Logger::logMutex;
const string Logger::logFileName = "logfile.log";
