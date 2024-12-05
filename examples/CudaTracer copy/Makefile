# Variables
SRCDIR = src
OBJDIR = obj
TESTDIR = test

CXX = g++
CXXFLAGS = -Wall -Wextra -O2 -std=c++20
LDFLAGS =

# Find all source files under src
SRC = $(shell find $(SRCDIR) -type f -name '*.cpp')

# Find all test source files
TEST_SRC = $(wildcard $(TESTDIR)/*.cpp)

# Generate list of object files, mirroring directory structure
SRC_OBJ = $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SRC))

# Generate list of test object files
TEST_OBJ = $(patsubst $(TESTDIR)/%.cpp, $(OBJDIR)/test_%.o, $(TEST_SRC))

# Generate list of test executables
TESTS = $(patsubst $(TESTDIR)/%.cpp, %, $(TEST_SRC))

# Include directories under src for headers
INCLUDE_DIRS = $(shell find $(SRCDIR) -type d)
CXXFLAGS += $(addprefix -I,$(INCLUDE_DIRS))

# Rules
.PHONY: all clean

all: $(TESTS)

# Rule to build each test executable
$(TESTS): %: $(SRC_OBJ) $(OBJDIR)/test_%.o
	$(CXX) $(LDFLAGS) -o $@ $^

# Compile source files into object files
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)/%
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Compile test source files into object files
$(OBJDIR)/test_%.o: $(TESTDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Ensure obj directories exist
$(OBJDIR)/%:
	mkdir -p $(dir $@)

$(OBJDIR):
	mkdir -p $(OBJDIR)

clean:
	rm -rf $(OBJDIR) $(TESTS)
