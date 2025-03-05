CC = gcc -std=c99 -fopenmp
CFLAGS = -O0 -Wall 
TARGET = non_block_mm

OBJS = main.o matrix_non_block_serial.o

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

%.o: %.c matrix_non_block_serial.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)
