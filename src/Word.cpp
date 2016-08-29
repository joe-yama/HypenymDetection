// Word.cpp

#include "Word.h"
#include <cassert>

using namespace std;

void coin::Word::save(ofstream& os){
  double grad = 0, bias = 0;
  os.write((char *)&id_, sizeof(int));
  os.write((char *)&bias, sizeof(double));
  os.write((char *)&grad, sizeof(double));
  const int dim = params_.dim_;
  for(int i = 0; i < dim; ++i){
    os.write((char*)&(vector_[i]), sizeof(double));
  }
  size_t w_size = w_str_.size();
  const char* word_char = w_str_.c_str();
  os.write((char *)&w_size, sizeof(w_size));
  os.write(word_char, w_size);
  os.write((char *)&hype_id_, sizeof(int));
}

void coin::Word::load(ifstream& is){
  double grad = 0, bias = 0;
  is.read((char *)&id_, sizeof(int));
  is.read((char *)&bias, sizeof(double));
  is.read((char *)&grad, sizeof(double));
  const int dim = params_.dim_;
  vector_ = Vector(dim);
  for(int i = 0; i < dim; ++i){
    is.read((char *)&(vector_[i]), sizeof(double));
  }
  vector_.normalize();
  size_t w_size;
  is.read((char *)&w_size, sizeof(w_size));
  char w[w_size + 1];
  is.read(w, w_size);
  w[w_size] = '\0';
  w_str_ = w;
  is.read((char *)&hype_id_, sizeof(int));
}
