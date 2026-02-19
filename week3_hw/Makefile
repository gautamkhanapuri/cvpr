#!/usr/bin/make -f

CC := clang
CXX := clang++
CPPFLAGS := -I/opt/homebrew/include/opencv4 -I/opt/homebrew/include/onnxruntime
VERSION := 17
CXXFLAGS := -Wall -std=c++$(VERSION)
LDFLAGS := -L/opt/homebrew/lib/opencv4/3rdparty -L/opt/homebrew/lib
# LDLIBS := -ltiff -lpng -ljpeg -llapack -lblas -lz -lwebp -framework AVFoundation -framework CoreMedia -framework CoreVideo -framework CoreServices -framework CoreGraphics -framework AppKit -framework OpenCL  -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect 
LDLIBS := -ltiff -lpng -ljpeg -llapack -lblas -lz -lwebp -framework AVFoundation -framework CoreMedia -framework CoreVideo -framework CoreServices -framework CoreGraphics -framework AppKit -framework OpenCL  -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lonnxruntime


TARGET := hist
HDRS := $(wildcard *.h) $(wildcard *.hpp)
# SRCS := $(wildcard *.cpp)
# OBJS := $(SRCS:.cpp=.o)

all: $(TARGET)


debug: CXXFLAGS += -g -O0 -DDEBUG
debug: $(TARGET)


%.o: %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< 


hist: makeHist.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)


clean: 
	rm -f $(OBJS) $(TARGET)


.PHONY: clean all
