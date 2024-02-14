CC=g++
OPT=
#SANITIZER= -fsanitize=address -fno-omit-frame-pointer 
CXXFLAGS= -g3 -ggdb -std=c++20 -fopenmp -Wno-c++98-compat -Wall  -Wpedantic -fno-omit-frame-pointer
GTEST= -L/home/kstppd/libs/googletest/build/lib  -I/home/kstppd/libs/googletest/googletest/include -lgtest -lgtest_main -lpthread
OBJ= main.o
BIN=mempool_test

default: main.o
clean: 
	rm  ${BIN}
allclean:
	rm ${BIN} &

main.o: main.cpp
	${CC} ${SANITIZER} ${CXXFLAGS} ${OPT}  -o ${BIN} main.cpp ${GTEST}



