dj: dj.o
	mpicc -fopenmp dj.o -o dj
	
dj.o:  dj.c
	mpicc -c -fopenmp -Wall dj.c 

clean:
	rm -f dj a.out dj.o 
	
