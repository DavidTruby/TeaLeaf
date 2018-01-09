vpath %.c $(KERNELS_PATH)

include $(KERNELS_PATH)/make.deps

KERNEL_SRCS := $(wildcard $(KERNELS_PATH)/*.c)
KERNEL_OBJS := $(patsubst $(KERNELS_PATH)/%.c, obj/$(KERNELS)/%.o, $(KERNEL_SRCS))
KERNEL_OBJS += obj/$(KERNELS)/cg.o obj/$(KERNELS)/local_halos.o

TL_LINK = $(CPP)

build_kernels: $(KERNEL_OBJS)
	@echo "Built OpenMP 4.0 kernels"

TL_FLAGS += -DDIFFUSE_OVERLOAD
TL_LDFLAGS += $(LDFLAGS)

obj/$(KERNELS)/cg.o: $(KERNELS_PATH)/cg.cpp Makefile make.deps $(KERNELS_PATH)/make.deps
	$(CPP) $(CFLAGS) $(OPTIONS) -I$(RAJA_ROOT)/include -std=c++14 -fopenmp-implicit-declare-target -fopenmp-implicit-map-lambdas -c $< -o $@ --gcc-toolchain=$(HOME)/power/usr

obj/$(KERNELS)/local_halos.o: $(KERNELS_PATH)/local_halos.cpp Makefile make.deps $(KERNELS_PATH)/make.deps
	$(CPP) $(CFLAGS) $(OPTIONS) -I$(RAJA_ROOT)/include -std=c++14 -fopenmp-implicit-declare-target -fopenmp-implicit-map-lambdas -c $< -o $@ --gcc-toolchain=$(HOME)/power/usr
