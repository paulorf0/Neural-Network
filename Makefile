CXX = g++
CXXFLAGS = -std=c++23 -Wall -Wextra -O3 -Iinclude 

TARGET = $(BUILD_DIR)/main
BUILD_DIR = build

# Define onde o Make deve procurar por arquivos .cpp
vpath %.cpp . src

SRCS = main.cpp network.cpp
OBJS = $(addprefix $(BUILD_DIR)/, $(notdir $(SRCS:.cpp=.o)))

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $(BUILD_DIR)/$(notdir $@)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

run: all
	$(TARGET)

.PHONY: all clean run