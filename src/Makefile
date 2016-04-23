CC=gcc
CFLAGS=-I. -Wall -Werror -msse4.2
DEPS=
OBJ= main.o
TARGET= build

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(TARGET): $(OBJ)
	gcc -o $@ $^ $(CFLAGS)

clean:
	rm -f $(OBJ) $(TARGET)
