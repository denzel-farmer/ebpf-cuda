#include <iostream>
#include <fstream>
#include <csignal>
#include <unistd.h>

#include "TracerAgent.h"

void signalHandler(int signum) {
    std::cout << "Interrupt signal (" << signum << ") received.\n";
    exit(signum);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <pid>" << std::endl;
        return 1;
    }

    int pid = std::stoi(argv[1]);


    // Register signal handler
    signal(SIGINT, signalHandler);

    TracerAgent agent(pid);

    std::cout << "TracerAgent created, press enter to start" << std::endl;

    std::cin.get();

    agent.StartAgentAsync();

    std::cout << "Agent started. Press Enter to dump history and exit..." << std::endl;
    std::cin.get();

    agent.dumpHistory("history_dump.json");
    std::cout << "History dumped to history_dump.json" << std::endl;

    return 0;
}