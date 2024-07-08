HEADERS=des.h
SOURCES=main.c des.c
OBJECTS=main.o des.o
EXECUTABLE=BUILD/des
CC=gcc -std=gnu99
LIBS=-lrt -lpthread
OPTS = -O3 -march=native -mtune=native -ftree-vectorize -ftree-vectorizer-verbose=1 -fopt-info-vec-optimized
CFLAGS= -c $(OPTS)
CFLAGS_CL = -DOPENCL
LIBS_CL = -lOpenCL

all:
	-test -e "main.o" && make clean_objects
	make nocl
	make clean_objects
	make opencl
	make clean_objects
	make tests
	make clean_objects
	make tests_cl

build_dir:
	-test ! -e "BUILD" && mkdir BUILD



opencl: opencl_flags nocl build_dir

nocl: $(OBJECTS) build_dir
	$(CC) $(OBJECTS) $(LIBS) -o $(EXECUTABLE)

opencl_flags:
	$(eval CFLAGS := $(CFLAGS) $(CFLAGS_CL))
	$(eval LIBS := $(LIBS) $(LIBS_CL))
	$(eval EXECUTABLE := $(EXECUTABLE)_cl)

tests_flag:
	$(eval CFLAGS := $(CFLAGS) -D UNIT_TESTS)
	$(eval EXECUTABLE := $(EXECUTABLE)_tests)

debug:
	$(eval CFLAGS:= -c -g)

tests: tests_flag nocl build_dir

tests_cl: tests_flag opencl build_dir

main.o: main.c des.h
	$(CC) $(CFLAGS) $< -o $@

des.o: des.c des.h
	$(CC) $(CFLAGS) $< -o $@

clean:
	make clean_objects
	make clean_build

clean_objects:
	rm $(OBJECTS)

clean_build:
	rm -r BUILD
