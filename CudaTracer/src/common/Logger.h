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
#include <cstring>    // For strerror function
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
        oss << put_time(localtime(&now_time_t), "%H:%M:%S");
        auto now_ms = chrono::time_point_cast<chrono::milliseconds>(now);
        auto value = now_ms.time_since_epoch();
        long duration = value.count() % 1000;
        oss << '.' << setfill('0') << setw(3) << duration;
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



    // std::string GetCallerName() {
    //     const int maxFrames = 3; // We need the second frame for the caller
    //     void* callStack[maxFrames];
    //     int frames = backtrace(callStack, maxFrames);

    //     if (frames < 2) {
    //         return "Unknown Caller";
    //     }

    //     // Get symbol names
    //     char** symbols = backtrace_symbols(callStack, frames);
    //     if (!symbols) {
    //         return "Unknown Caller";
    //     }

    //     std::string callerName;
    //     // Print the caller name for debugging purposes
    //     cout << "Caller: " << symbols[2] << endl;
    //     // Demangle the symbol name
    //     char* demangledName = nullptr;
    //     int status = 0;
    //     demangledName = abi::__cxa_demangle(symbols[2], nullptr, nullptr, &status);

    //     if (status == 0 && demangledName) {
    //         callerName = demangledName;
    //         free(demangledName); // Free the memory allocated by __cxa_demangle
    //     } else {
    //         callerName = symbols[2]; // Fallback to mangled name
    //     }

    //     free(symbols); // Free memory allocated by backtrace_symbols
    //     return callerName;
    // }


    // Get the current function name using backtrace
    inline string getFunctionName() const {
        void* callstack[3]; // Increase the size to get the next name up the stack
        int frames = backtrace(callstack, 3);
        if (frames < 3) {
            return "UNKNOWN";
        }

        char** symbols = backtrace_symbols(callstack, frames);
        if (!symbols) {
            return "UNKNOWN";
        }

        string funcName = symbols[2]; // Caller function is now at index 2
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
    Logger(LogLevel level, const string& fileName) : logLevel(level), logFileName(fileName) {
        log_info("-------------------- Logger Initialized --------------------");
    }

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
       // log(LogLevel::ERROR, message);
        int errnum = errno;
        if (errnum != 0) {
            ostringstream errMsg;
            errMsg << "ERROR: " << message << " | Error number: " << errnum << ", Error message: " << strerror(errnum);
            log(LogLevel::ERROR, errMsg.str());
            cerr << errMsg.str() << endl;
        } else {
            log(LogLevel::ERROR, message);
            cerr << "ERROR: " << message << endl;
        }
    }

    // Log function
    inline void log(LogLevel level, const string& message) {
        if (level < logLevel) {
            return;
        }

        // ostringstream logEntry;
        // logEntry << "[" << getCurrentTime() << "] "
        //          << "[" << getFunctionName() << "] "
        //          << "[Thread " << this_thread::get_id() << "] "
        //          << message;
        ostringstream logEntry;
        logEntry << "[" << getCurrentTime() << "] "
             << "[" << logLevelToString(level) << "] "
             << message;


        {
            lock_guard<mutex> lock(logMutex);
            // Print to standard error
            // cerr << logEntry.str() << endl;

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