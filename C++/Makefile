CC = gcc
CFLAGS = -lm -fopenmp 
CFLAGS_DEP = -O3 -march=native

default: chen_attractor_c

chen_attractor_c: chen_attractor_c.o
	$(CC) chen_attractor_c.o -o chen_attractor_c $(CFLAGS)

chen_attractor_c.o: chen_attractor_c.c
	$(CC) -c chen_attractor_c.c $(CFLAGS) $(CFLAGS_DEP)
clean: 
	$(RM) count *.o *~
