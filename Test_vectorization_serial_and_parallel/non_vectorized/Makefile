CC = gcc -std=c99 -fopenmp
CFLAGS = -O2 -Wall 
TARGET = non_vectorized_mm

OBJS = main.o matrix_non_vectorized_serial.o

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

%.o: %.c matrix_non_vectorized_serial.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)
