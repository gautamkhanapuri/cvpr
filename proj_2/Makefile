#!/usr/bin/make -f

CC := clang
CXX := clang++
CPPFLAGS := -I/opt/homebrew/include/opencv4 -I/opt/homebrew/include/onnxruntime
VERSION := 20
CXXFLAGS := -Wall -std=c++$(VERSION)
LDFLAGS := -L/opt/homebrew/lib/opencv4/3rdparty -L/opt/homebrew/lib
LDLIBS := -ltiff -lpng -ljpeg -llapack -lblas -lz -lwebp -framework AVFoundation -framework CoreMedia -framework CoreVideo -framework CoreServices -framework CoreGraphics -framework AppKit -framework OpenCL  -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lonnxruntime

TARGET := p1 p2
HDRS := $(wildcard *.h) $(wildcard *.hpp)
SRCS := $(wildcard *.cpp)
OBJS := $(SRCS:.cpp=.o)

all: $(TARGET)


debug: CXXFLAGS += -g -O0 -DDEBUG
debug: $(TARGET)


%.o: %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $<
#$(OBJS): $(HDRS) $(SRCS)
#	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(SRCS)

p1: p1.o csv_util.o mycv_utils.o utils.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)


p2: p2.o csv_util.o mycv_utils.o utils.o dist_utils.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)


clean:
	rm -f $(OBJS) $(TARGET)


.PHONY: clean all
