CXX = g++
CXXFLAGS = -std=c++23 -Wall -Wextra -O3 -Iinclude 

TARGET = main
BUILD_DIR = build
SRCS = main.cpp network.cpp
OBJS = $(addprefix $(BUILD_DIR)/, $(SRCS:.cpp=.o))

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

run: all
	./$(TARGET)

.PHONY: all clean run