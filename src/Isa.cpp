// Isa.cpp

#include "Isa.h"

using namespace std;

coin::Isa::Isa(const Parameters& params, Word* hypo, Word* hype):
  params_(params), id_(hypo->id()), hypo_(hypo), hype_(hype){}

bool coin::more(const Estimate& lhs, const Estimate& rhs){
  return lhs.sim()>rhs.sim();
}


