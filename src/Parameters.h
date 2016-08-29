// Parameters.h
#ifndef PARAMETERS_H_
#define PARAMETERS_H_
#include "Global.h"
#include <string>
#include <fstream>
#include <cstring>

namespace coin{
  class Parameters final{
  private:
  public:
  Parameters(int dim):dim_(dim){}
    ~Parameters(){}
    int dim_ = 300;
    double threshold_ = MINIMAM_DOUBLE;
    int k_ = 1; // number of negative instances

    //initial params of Adam
    double alpha_ = 0.001; //original paper recommends 0.001
    double beta1_ = 0.9; //original paper recommends 0.9
    double beta2_ = 0.999; //original paper recommends 0.999
    double epsilon_ = 0.00000001; //original paper recommends 10^(-8)

    void save(std::ofstream& os){
      os.write((char *)&threshold_, sizeof(double));
      os.write((char *)&k_, sizeof(int));
      os.write((char *)&alpha_, sizeof(double));
      os.write((char *)&beta1_, sizeof(double));
      os.write((char *)&beta2_, sizeof(double));
      os.write((char *)&epsilon_, sizeof(double));
    }

    void load(std::ifstream& is){
      is.read((char *)&threshold_, sizeof(double));
      is.read((char *)&k_, sizeof(int));
      is.read((char *)&alpha_, sizeof(double));
      is.read((char *)&beta1_, sizeof(double));
      is.read((char *)&beta2_, sizeof(double));
      is.read((char *)&epsilon_, sizeof(double));
    }

  };
} // namespace coin
#endif
