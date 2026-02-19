#!/usr/bin/make -f

CC := clang
CXX := clang++
CPPFLAGS := -I/opt/homebrew/include/opencv4
VERSION := 11
CXXFLAGS := -Wall -std=c++$(VERSION)
LDFLAGS := -L/opt/homebrew/lib/opencv4/3rdparty -L/opt/homebrew/lib
LDLIBS := -ltiff -lpng -ljpeg -llapack -lblas -lz -ljasper -lwebp -lIlmImf -lgs -framework AVFoundation -framework CoreMedia -framework CoreVideo -framework CoreServices -framework CoreGraphics -framework AppKit -framework OpenCL  -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect

TARGET :=
HDRS := $(wildcard *.h)
SRCS := $(wildcard *.cpp)
OBJS := $(SRCS:.cpp=.o)

all: $(TARGET)


debug: CXXFLAGS += -g -O0 -DDEBUG
debug: $(TARGET)


$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

	
$(OBJS): $(HDRS) $(SRCS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $(SRCS)


clean: 
	rm -f $(OBJS) $(TARGET)


.PHONY: clean all
