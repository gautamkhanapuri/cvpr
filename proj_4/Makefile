#!/usr/bin/make -f

CC := clang
CXX := clang++
CPPFLAGS := -I/opt/homebrew/include/opencv4 -I/opt/homebrew/include/onnxruntime
VERSION := 20
CXXFLAGS := -Wall -std=c++$(VERSION)
LDFLAGS := -L/opt/homebrew/lib/opencv4/3rdparty -L/opt/homebrew/lib
LDLIBS := -ltiff -lpng -ljpeg -llapack -lblas -lz -lwebp -framework AVFoundation -framework CoreMedia -framework CoreVideo -framework CoreServices -framework CoreGraphics -framework AppKit -framework OpenCL  -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lopencv_calib3d -lopencv_features2d
# Add ' -lonnxruntime -lopencv_dnn' to LDLIBS for using models

TASK7 := task7_orb task7_ff
TARGET := caar
HDRS := $(wildcard *.h) $(wildcard *.hpp)
SRCS := $(wildcard *.cpp)
OBJS := $(SRCS:.cpp=.o)

all: $(TARGET) $(TASK7)
task7: $(TASK7)


debug: CXXFLAGS += -g -O0 -DDEBUG
debug: $(TARGET)


%.o: %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $<


caar: main.o caar.o calibrate.o ar.o utils.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)


task7_orb: task7_orb.o utils.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)


task7_hf: task7_harris_corners.o utils.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)


task7_ff: task7_fast_features.o utils.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)


clean:
	rm -f $(OBJS) $(TARGET) $(TASK7)


.PHONY: clean all
