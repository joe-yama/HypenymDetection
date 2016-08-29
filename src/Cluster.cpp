// Cluster.cpp
#include "Cluster.h"

using namespace std;
using namespace coin;

void coin::Cluster::save(ofstream& os){
  const int dim = params_.dim_;
  for(int i = 0; i < dim; i++){
    for(int j = 0; j < dim; j++){
      os.write((char *)&matrix_(i,j), sizeof(double));
    }
  }
  os.write((char *)&bias_, sizeof(double));
  os.write((char *)&alpha_, sizeof(double));
  os.write((char *)&beta1_, sizeof(double));
  os.write((char *)&beta2_, sizeof(double));
  for(int i = 0; i < dim; i++){
    for(int j = 0; j < dim; j++){
      os.write((char *)&m_(i,j), sizeof(double));
    }
  }
  for(int i = 0; i < dim; i++){
    for(int j = 0; j < dim; j++){
      os.write((char *)&v_(i,j), sizeof(double));
    }
  }
  os.write((char *)&bm_, sizeof(double));
  os.write((char *)&bv_, sizeof(double));
  size_t list_size = list_.size();
  os.write((char *)&list_size, sizeof(list_size));
  for(unsigned int i = 0; i < list_size; i++){
    os.write((char *)&list_[i], sizeof(int));
  }
}

void coin::Cluster::load(ifstream& is){
  const int dim = params_.dim_;
  matrix_ = Matrix::Zero(dim,dim);
  m_ = Matrix::Zero(dim,dim);
  v_ = Matrix::Zero(dim,dim);
  for(int i = 0; i < dim; i++){
    for(int j = 0; j < dim; j++){
      is.read((char *)&matrix_(i,j), sizeof(double));
    }
  }
  is.read((char *)&bias_, sizeof(double));
  is.read((char *)&alpha_, sizeof(double));
  is.read((char *)&beta1_, sizeof(double));
  is.read((char *)&beta2_, sizeof(double));
  for(int i = 0; i < dim; i++){
    for(int j = 0; j < dim; j++){
      is.read((char *)&m_(i,j), sizeof(double));
    }
  }
  for(int i = 0; i < dim; i++){
    for(int j = 0; j < dim; j++){
      is.read((char *)&v_(i,j), sizeof(double));
    }
  }
  is.read((char *)&bm_, sizeof(double));
  is.read((char *)&bv_, sizeof(double));
  size_t list_size = 0;
  is.read((char *)&list_size, sizeof(list_size));
  for(unsigned int i = 0; i < list_size; i++){
    int id = 0;
    is.read((char *)&id, sizeof(int));
    list_.push_back(id);
  }
}

Vector coin::Cluster::project(Isa* isa){
  return isa->hypo()->vector() * matrix_;
}

double coin::Cluster::sim(Isa* isa){
  Vector xphi = project(isa);
  const Vector& y = isa->hype()->vector();
  double dot = xphi.dot(y);
  return  dot + bias_;
}

double coin::Cluster::sim(Word* hypo, Word* hype){
  const Vector& x = hypo->vector();
  const Vector& y = hype->vector();
  return (x * matrix_).dot(y) + bias_;
}

double coin::Cluster::distance(Isa* isa){
  Vector offset = isa->hype()->vector() - isa->hypo()->vector();
  return (mean_ - offset).squaredNorm();
}

void coin::Cluster::shuffle(){
  std::shuffle( list_.begin(), list_.end(), Random::engine() );
}

void coin::Cluster::ResetLearningRate(){
  nupdate_ = 0;
  beta1_ = 1;
  beta2_ = 1;
  alpha_ = params_.alpha_;
  const int dim = params_.dim_;
  m_ = Matrix::Zero(dim,dim);
  v_ = Matrix::Zero(dim,dim);
}

void coin::Cluster::AddGrad(const Matrix& phi_grad, const double& bias_grad){
  phi_grad_ += phi_grad;
  bias_grad_ += bias_grad;
  ngrad_++;
}

void coin::Cluster::ResetGrad(){
  const int dim = params_.dim_;
  phi_grad_ = Matrix::Zero(dim, dim);
  bias_grad_ = 0;
  ngrad_ = 0;
}

void coin::Cluster::update(const Matrix& phi_grad, const double& bias_grad){
  AddGrad(phi_grad, bias_grad);
  
  phi_grad_ = phi_grad_.array() / ngrad_;
  bias_grad_ = bias_grad_ / ngrad_;
  
  ++nupdate_;
  beta1_ = beta1_*params_.beta1_;
  beta2_ = beta2_*params_.beta2_;
    
  m_ = params_.beta1_*m_ + (1-params_.beta1_)*phi_grad_;
  bm_ = params_.beta1_*bm_ + (1-params_.beta1_)*bias_grad_;

  v_ = params_.beta2_*v_ + (1-params_.beta2_)*(phi_grad_.cwiseProduct(phi_grad_));
  bv_ = params_.beta2_*bv_ + (1-params_.beta2_)*bias_grad_*bias_grad_;

  alpha_ = params_.alpha_*sqrt(1-beta2_) / (1-beta1_);
  Matrix a_m = alpha_ * m_;
  Matrix sqrtv_e = v_.cwiseSqrt().array() + params_.epsilon_;
  matrix_ -= a_m.cwiseQuotient(sqrtv_e);
  bias_ -= alpha_ * bm_/(sqrt(bv_)+params_.epsilon_);
  ResetGrad();
}
