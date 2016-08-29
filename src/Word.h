// Word.h
#ifndef WORD_H_
#define WORD_H_
#include "Global.h"
#include "Parameters.h"
#include <cmath>
#include <iostream>
#include <fstream>
namespace coin{
  class Word final{
  private:
    const Parameters& params_;
    int id_;
    Vector vector_;
    std::string w_str_;
    int hype_id_ = -1;
  public:
  Word(const Parameters& params,int id, Vector vector, std::string w_str):
    params_(params), id_(id), vector_(vector), w_str_(w_str){}
  Word(const Parameters& params, std::ifstream& is): params_(params){
      load(is);
    }
    ~Word(){}
    void setHypeId(int id){ hype_id_ = id; }
    int id() const{ return id_; }
    const Vector& vector() const{ return vector_; }
    Vector* pvec(){ return &vector_; }
    const std::string& w_str() const{ return w_str_; }
    int hype_id() const{ return hype_id_; }
    
    void save(std::ofstream& os);
    void load(std::ifstream& is);
    friend std::ostream& operator<<(std::ostream& stream, const Word& word);
    bool operator==(const Word& rhs){
      return this->id() == rhs.id();
    }
  };
  extern std::ostream& operator<<(std::ostream& stream, const Word& word);
  
} // namespace coin
#endif
