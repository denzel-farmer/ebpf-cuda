# Variables
SRCDIR = src
OBJDIR = obj
TESTDIR = test

CXX = g++
CXXFLAGS = -Wall -Wextra -Werror -Wpedantic -O2 -std=c++20
LDFLAGS = -lpthread

# Find all source files under src
SRC = $(shell find $(SRCDIR) -type f -name '*.cpp')

# Find all test source files
TEST_SRC = $(wildcard $(TESTDIR)/*.cpp)

# Generate list of object files, mirroring directory structure
SRC_OBJ = $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SRC))

# Generate list of test object files
TEST_OBJ = $(patsubst $(TESTDIR)/%.cpp, $(OBJDIR)/test_%.o, $(TEST_SRC))

# Include directories under src for headers
INCLUDE_DIRS = $(shell find $(SRCDIR) -type d)
CXXFLAGS += $(addprefix -I,$(INCLUDE_DIRS))

# Generate list of test executables
TESTS = $(patsubst $(TESTDIR)/%.cpp, %, $(TEST_SRC))

# Rules
.PHONY: all clean

# Build all test executables
all: $(TESTS)

# Rule to build each test executable
$(TESTS): %: $(OBJDIR)/test_%.o $(SRC_OBJ)
	$(CXX) -o $@ $^ $(LDFLAGS)

# Compile source files into object files
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Compile test source files into object files
$(OBJDIR)/test_%.o: $(TESTDIR)/%.cpp
	@mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -rf $(OBJDIR) $(TESTS)
