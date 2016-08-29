// Isa.h
#ifndef ISA_H_
#define ISA_H_
#include "Word.h"
#include <cmath>

namespace coin{                                                                                                       class Isa{                                                                                                        
  protected:                                                                                                        
    const Parameters& params_;                                                                                       
    int id_; // hyponymのid                                                                                         
    Word* hypo_ = nullptr;                                                                                          
    Word* hype_ = nullptr;                                                                                          
    int indicator_ = -1;                                                                                            
  public:                                                                                                           
    Isa(const Parameters& params, Word* hypo, Word* hype);                                                          
    ~Isa(){}                                                                                                        
    void setHype(Word* hype){ hype_ = hype; }                                                                       
    void setIndicator(int c_id){ indicator_ = c_id; }                                                               
    int id() const{ return id_; }                                                                                   
    Word* hype() const{ return hype_; }                                                                             
    Word* hypo() const{ return hypo_; }                                                                             
    int indicator() const{ return indicator_; }                                                                     
    bool operator==(const Isa& rhs) const{                                                                          
      return this->id() == rhs.id();                                                                                
    }

  };
  class Estimate{
  private:
    Word* hype_;
    double sim_;
    int indicator_ = 0;
  public:
  Estimate(Word* hype): hype_(hype){}
    ~Estimate(){}
    Word* hype(){
      return hype_;
    }
    double sim() const{
      return sim_; 
    }
    int indicator() const{
      return indicator_;
    }
    void setSim(double sim){
      sim_ = sim;
    }
    void setIndicator(int id){
      indicator_ = id;
    }
  };
  bool more(const Estimate& lhs, const Estimate& rhs);
}
#endif
