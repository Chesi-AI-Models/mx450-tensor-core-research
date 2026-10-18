obj-m += mx450_probe.o
obj-m += mx450_p1.o
obj-m += mx450_p2.o
obj-m += mx450_p3.o
obj-m += mx450_p4.o
obj-m += mx450_p5.o
KDIR ?= /lib/modules/$(shell uname -r)/build
all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules
clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
