# Variables
SRCDIR = src
OBJDIR = obj
TESTDIR = test

CXX = g++
CXXFLAGS = -Wall -Wextra -O2 -std=c++20 -I$(SRCDIR)
LDFLAGS = 
TARGET_NAME = TestMemHistory
SRC = $(wildcard $(SRCDIR)/*.cpp)
TEST_SRC = $(wildcard $(TESTDIR)/*.cpp)
SRC_OBJ = $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SRC))
TEST_OBJ = $(patsubst $(TESTDIR)/%.cpp, $(OBJDIR)/test_%.o, $(TEST_SRC))
TARGET = $(TARGET_NAME)

# Hardcoded headers
HEADERS = $(SRCDIR)/MemHistory.h $(SRCDIR)/AllocationHistory.h

# Rules
.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC_OBJ) $(TEST_OBJ) | $(TESTDIR)
	$(CXX) $(LDFLAGS) -o $@ $^

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(HEADERS) | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(OBJDIR)/test_%.o: $(TESTDIR)/%.cpp $(HEADERS) | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(TESTDIR):
	mkdir -p $(TESTDIR)

clean:
	rm -rf $(OBJDIR) $(TARGET)
