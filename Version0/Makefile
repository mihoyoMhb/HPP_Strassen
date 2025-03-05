CC = gcc
CFLAGS = -O3 -march=native -ffast-math -fopenmp -Wall -fopt-info-vec -g
TARGET = matrix_program

OBJS = main.o matrix_common.o matrix_serial.o matrix_parallel.o strassen_parallel.o strassen_serial.o

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

%.o: %.c matrix.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)
