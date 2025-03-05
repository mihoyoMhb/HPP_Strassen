CC = gcc
CFLAGS = -O3 -march=native -ffast-math -fopenmp -Wall -fopt-info-vec
TARGET = vectorized_mm

OBJS = main.o matrix_vectorized_serial.o

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

%.o: %.c matrix_vectorized_serial.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)
