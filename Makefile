SRCS = src/GetHypernym.cpp src/Model.cpp src/Word.cpp src/Cluster.cpp src/Isa.cpp
OBJS = src/GetHypernym.o src/Model.o src/Word.o src/Cluster.o src/Isa.o

CPPFLAGS=  -Wall -O2 -g -std=c++11 -fopenmp

DEP = Makefile.depend

all: $(DEP) GetHypernym
GetHypernym: $(OBJS)
	g++ $(CPPFLAGS) -o GetHypernym $(OBJS)

clean:
	rm src/*.o GetHypernym

depend : $(DEP)

$(DEP): $(SRCS)
	g++ $(SRCS) -MM -MG -std=c++11 | sed -e 's/^\([^ ]\)/src\/\1/' > $(DEP)

-include Makefile.depend
