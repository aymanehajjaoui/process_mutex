# Default model (can be set from the command line)
MODEL ?= Z10

# Compiler Definitions
CC := gcc
CXX := g++

# Common compilation flags (shared between C and C++)
COMMON_FLAGS  = -Wall -Wextra -O3 -pedantic -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard -mtune=cortex-a9 -D$(MODEL)
COMMON_FLAGS += -I/opt/redpitaya/include
COMMON_FLAGS += -I$(CURDIR)/include
COMMON_FLAGS += -I$(CURDIR)/CMSIS -I$(CURDIR)/CMSIS/Core/Include
COMMON_FLAGS += -I$(CURDIR)/CMSIS/DSP/Include
COMMON_FLAGS += -I$(CURDIR)/CMSIS/NN/Include
COMMON_FLAGS += -I$(CURDIR)/CMSIS/NN/Source/ActivationFunctions
COMMON_FLAGS += -I$(CURDIR)/CMSIS/NN/Source/ConvolutionFunctions
COMMON_FLAGS += -I$(CURDIR)/CMSIS/NN/Source/FullyConnectedFunctions
COMMON_FLAGS += -I$(CURDIR)/model/include

# Specific flags for C and C++
CFLAGS  = -std=gnu11 $(COMMON_FLAGS)
CXXFLAGS = -std=c++20 $(COMMON_FLAGS)

# Linking flags
LDFLAGS = -L/opt/redpitaya/lib -flto -Wl,--gc-sections
LDLIBS  = -lrp -lrp-i2c -lm -lpthread -lrt -lrp-hw -lrp-hw-calib -lrp-hw-profiles -lstdc++

# Model-specific libraries
ifeq ($(MODEL),Z20_250_12)
    COMMON_FLAGS += -I/opt/redpitaya/include/api250-12
    LDLIBS += -lrp-hw-calib -lrp-hw-profiles -lrp-gpio -lrp-i2c
endif

# List of compiled programs
PRGS = can

# Step 1: Compile the Model files
MODEL_C_FILES := $(wildcard model/*.c)
MODEL_OBJS := $(MODEL_C_FILES:.c=.o)

# Step 2: Compile CMSIS NN files
CMSIS_C_FILES := $(wildcard CMSIS/NN/**/*.c)
CMSIS_CPP_FILES := $(wildcard CMSIS/NN/**/*.cpp)
CMSIS_OBJS := $(CMSIS_C_FILES:.c=.o) $(CMSIS_CPP_FILES:.cpp=.o)

# Step 3: Compile SRC and INCLUDE files
SRC_FILES := $(wildcard src/*.cpp) $(wildcard include/*.cpp)
OBJS := $(SRC_FILES:.cpp=.o)

# Targets
all: clean $(PRGS)

# Compile the model first
$(MODEL_OBJS): %.o: %.c
	$(CC) -c $< $(CFLAGS) -o $@

# Ensure CMSIS object files are compiled
$(CMSIS_OBJS): %.o: %.c
	$(CC) -c $< $(CFLAGS) -o $@

$(CMSIS_OBJS): %.o: %.cpp
	$(CXX) -c $< $(CXXFLAGS) -o $@

# Compile SRC and INCLUDE files
%.o: %.cpp
	$(CXX) -c $< $(CXXFLAGS) -o $@

# Link everything together
$(PRGS): $(MODEL_OBJS) $(CMSIS_OBJS) $(OBJS)
	$(CXX) $(MODEL_OBJS) $(CMSIS_OBJS) $(OBJS) $(LDFLAGS) $(LDLIBS) -o $@

# Clean rule to remove all object files and binaries
clean:
	find . -name "*.o" -delete
	$(RM) $(PRGS)
	@if [ -d DataOutput ]; then find DataOutput -type f -delete; fi
	@if [ -d ModelOutput ]; then find ModelOutput -type f -delete; fi

.PHONY: all clean
