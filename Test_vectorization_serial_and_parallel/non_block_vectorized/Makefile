CC = gcc -std=c99 -fopenmp
CFLAGS = -O3 -march=native -ffast-math -fopenmp -Wall -fopt-info-vec
TARGET = non_block_vectorized_mm

OBJS = main.o matrix_non_block_vectorized_serial.o

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

%.o: %.c matrix_non_block_vectorized_serial.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)
