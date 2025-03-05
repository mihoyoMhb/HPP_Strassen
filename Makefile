CC = gcc
CFLAGS = -O2 -fopenmp -Wall
TARGET = matrix_program

OBJS = main.o matrix_common.o matrix_serial.o matrix_parallel.o

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

%.o: %.c matrix.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)
