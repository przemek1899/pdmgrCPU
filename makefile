bregman: main.o read.o
	g++ -fopenmp  main.o read.o -o bregman
main.o: main.cpp
	g++ -fopenmp -c main.cpp -O2 -larmadillo -llapack -lblas -o main.o
read.o: readFiles.cpp
	g++ -c readFiles.cpp -o read.o