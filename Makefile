CC=g++  
OPT=-O2
SANITIZER= -fsanitize=address  
CXXFLAGS= -g3 -ggdb -std=c++20 -fopenmp -Wall  -Wpedantic -fno-omit-frame-pointer
OBJ= main.o
BIN=mempool_test

default: main.o
clean: 
	rm  ${BIN}
allclean:
	rm ${BIN} &

main.o: main.cpp
	${CC} ${SANITIZER} ${CXXFLAGS} ${OPT}  -o ${BIN} main.cpp 


