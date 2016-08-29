// Cluster.h
#ifndef CLUSTER_H_
#define CLUSTER_H_
#include "Isa.h"
#include <vector>
#include <algorithm>
#include <ctime>



namespace coin{
  class Isa;
  class Cluster{
  private:
    const Parameters& params_;
    int id_;
    Matrix matrix_;
    double bias_ = 0;
    
    // Adam
    int nupdate_ = 0;
    double alpha_;
    double beta1_;
    double beta2_;
    Matrix m_;
    Matrix v_;
    double bm_ = 0;
    double bv_ = 0;
    Matrix phi_grad_;
    double bias_grad_ = 0;
    int ngrad_ = 0;
    std::vector<int> list_;

    Vector mean_;
    
  public:
  Cluster(const Parameters& params, int id): params_(params), id_(id){
      const int dim = params.dim_;
      matrix_ = Matrix::Random(dim, dim);
      m_ = Matrix::Zero(dim,dim);
      v_ = Matrix::Zero(dim,dim);
      alpha_ = params_.alpha_;
      beta1_ = 1;
      beta2_ = 1;
      phi_grad_ = Matrix::Zero(dim, dim);
    }
  Cluster(const Parameters params, std::ifstream& is, int id):params_(params), id_(id){
      load(is);
    }
    ~Cluster(){}
    int id() const{ return id_; }
    const Matrix& matrix() const{ return matrix_; }
    const double& bias() const{ return bias_; }
    Vector project(Isa* isa);
    double sim(Isa* isa);
    double sim(Word* hypo, Word* hype);
    const std::vector<int>& list() const{
      return list_;
    }
    void KmeansInit(){
      mean_ = Vector::Random(params_.dim_);
      mean_.normalize();
    }
    void setMatrix(const Matrix& phi){
      matrix_ = phi;
    }
    void add(int id){
      list_.push_back(id);
    }
    void del(int id){
      list_.erase(remove(list_.begin(), list_.end(), id), list_.end());
    }
    bool empty() const{
      return list_.empty();
    }
    void save(std::ofstream& os);
    void load(std::ifstream& is);
    double distance(Isa* isa);
    void shuffle();
    void setMean(const Vector& mean){
      mean_ = mean;
    }

    const Vector& mean() const{
      return mean_;
    }
    
    // Adam
    int nupdate() const{
      return nupdate_;
    }
    void ResetLearningRate();
    void update(const Matrix& phi_grad, const double& bias_grad);
    void AddGrad(const Matrix& phi_grad, const double& bias_grad);
    void ResetGrad();

    double sigmoid(const double& d){
      return 1. / (1. + exp(-d));
    }
  };
  
} // namespace coin
#endif
