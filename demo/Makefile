CC = g++
CFLAGS = -g -Wall -std=c++11 -lstdc++fs
SRCS = main.cpp
PROG = main

OPENCV = $(shell pkg-config opencv --cflags --libs)
OPENBLAS = -I /opt/OpenBLAS/include/ -L/opt/OpenBLAS/lib -lopenblas
BOOST =  -lboost_filesystem -lboost_system
LIBS = $(OPENCV) $(OPENBLAS) $(BOOST) -ldlib
LFLAGS = -Wl,-rpath,/usr/local/lib/

$(PROG) : $(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS) $(LFLAGS)
