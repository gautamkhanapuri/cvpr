#!/usr/bin/make -f

CC := clang
CXX := clang++
CPPFLAGS := -I/opt/homebrew/include/opencv4 -I/opt/homebrew/include/onnxruntime
VERSION := 17
CXXFLAGS := -Wall -std=c++$(VERSION)
LDFLAGS := -L/opt/homebrew/lib/opencv4/3rdparty -L/opt/homebrew/lib
LDLIBS := -ltiff -lpng -ljpeg -llapack -lblas -lz -lwebp -framework AVFoundation -framework CoreMedia -framework CoreVideo -framework CoreServices -framework CoreGraphics -framework AppKit -framework OpenCL  -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lonnxruntime

TARGET := task1 task2  # timer ext
HDRS := $(wildcard *.h) $(wildcard *.hpp)
SRCS := $(wildcard *.cpp)
OBJS := $(SRCS:.cpp=.o)

all: $(TARGET)


debug: CXXFLAGS += -g -O0 -DDEBUG
debug: $(TARGET)


# $(TARGET): $(OBJS)
#	 $(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

	
# $(OBJS): $(HDRS) $(SRCS)
# 	 $(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(SRCS)


%.o: %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< 


image_filter: image_filter.o filter.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)


timer: filter.o timeBlur.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)


ext: extensions.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)


task2: vidDisplay.o filter.o faceDetect.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)


task1: imgDisplay.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)


clean: 
	rm -f $(OBJS) $(TARGET)


.PHONY: clean all
