# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -mavx2 -ftree-vectorize -fopt-info-vec
OMPFLAGS = -fopenmp
INCLUDES = -I.
LDFLAGS = 

# Source files and targets
SERIAL_SRCS = strassen_serial.cpp
PARALLEL_SRCS = matrix.cpp
MAIN_SRC = main.cpp

SERIAL_OBJS = $(SERIAL_SRCS:.cpp=.o)
PARALLEL_OBJS = $(PARALLEL_SRCS:.cpp=.o)
MAIN_OBJ = $(MAIN_SRC:.cpp=.o)

TARGET = strassen_compare

# Default target
all: $(TARGET)

# Linking
$(TARGET): $(SERIAL_OBJS) $(PARALLEL_OBJS) $(MAIN_OBJ)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $^ $(LDFLAGS)

# Compile main with OpenMP support
$(MAIN_OBJ): $(MAIN_SRC)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(INCLUDES) -c -o $@ $<

# Compile parallel implementation with OpenMP
$(PARALLEL_OBJS): %.o: %.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(INCLUDES) -c -o $@ $<

# Compile serial implementation
$(SERIAL_OBJS): %.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

# Clean up
clean:
	rm -f $(SERIAL_OBJS) $(PARALLEL_OBJS) $(MAIN_OBJ) $(TARGET)

# Dependencies
matrix.o: matrix.cpp matrix.h
strassen_serial.o: strassen_serial.cpp strassen_serial.h
main.o: main.cpp matrix.h strassen_serial.h

.PHONY: all clean