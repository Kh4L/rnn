CC = clang++
BIN = rnn
OBJS = rnn.o main.o
RM = rm -rvf
CXXFLAGS += -std=c++14 -Wall -g

all: $(BIN)

$(BIN): $(OBJS)
	$(CC) $(CXXFLAGS) $(OBJS) -o $(BIN)
clean:
	$(RM) $(OBJS) $(BIN)

