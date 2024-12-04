# Compiler and Flags
CXX = g++
CXXFLAGS = -std=c++20 -Wall -fmodules-ts -I./src

# Directories
SRC_DIR = src
TEST_DIR = tests
BUILD_DIR = build

# Source Files
SRC_MODULE_INTERFACE = $(SRC_DIR)/MemHistory.ixx
SRC_MODULE_IMPL = $(SRC_DIR)/MemHistory.cpp
TEST_FILE = $(TEST_DIR)/TestMemHistory.cpp

# Output Files
MODULE_IFACE_UNIT = $(BUILD_DIR)/MemHistory.o
TEST_EXE = $(BUILD_DIR)/TestMemHistory

# Default Target
all: $(TEST_EXE)

# Build Module Interface Unit
$(MODULE_IFACE_UNIT): $(SRC_MODULE_INTERFACE)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Build Test Executable
$(TEST_EXE): $(MODULE_IFACE_UNIT) $(SRC_MODULE_IMPL) $(TEST_FILE) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SRC_MODULE_IMPL) $(TEST_FILE) $(MODULE_IFACE_UNIT) -o $@

# Create Build Directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean Build Files
clean:
	rm -rf $(BUILD_DIR)
