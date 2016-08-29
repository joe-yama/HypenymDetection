// Global.h

#ifndef GLOBAL_H_
#define GLOBAL_H_
#define EIGEN_STACK_ALLOCATION_LIMIT 99999999999 // 非常に大きい値
#include <eigen3/Eigen/Dense>
#include <chrono>
#include <iostream>
#include <cassert>
#include <limits>
#include <cstdlib>
#include <cmath>
#include <random>

const double MINIMAM_DOUBLE = - (std::numeric_limits<double>::max()); 

namespace coin{
  using Vector = Eigen::RowVectorXd;
  using Matrix = Eigen::MatrixXd;

  class Timer{
  private:
    std::chrono::time_point<std::chrono::system_clock> start_;
  public:
  Timer():start_(std::chrono::system_clock::now()){}
    int seconds() const{
      return (int)std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now()-start_).count();
    }
  };

  class Random{
  private:
  Random(): mt(1){}
  //Random(): mt(std::random_device()()){}
    std::mt19937 mt;
    Random(const Random& arg) = delete;
  public:
    static int random_id(int min, int max){
      static Random object;
      std::uniform_int_distribution<int> dist(min, max);
      return dist(object.mt);
    }
    static std::mt19937& engine(){
      static Random object;
      return object.mt;
    }
  };  

} // namespace coin
#endif
