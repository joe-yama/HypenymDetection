// SingleCluster.cpp

#include "Parameters.h"
#include "Word.h"
#include "Isa.h"
#include "Cluster.h"
#include "Model.h"
#include <fstream>
#include <iostream>

using namespace std;
using namespace coin;

int main(int argc, const char* argv[]){
  string option = argv[1];
  if(option==string("-d")){
    Parameters params(atoi(argv[4]));
    Model model(params);
    model.loadModelCluster(argv[2]);
    //    model.demonstrate(0);
    model.demonstrate_kmeans();
  }else if(option==string("--class")){
    Parameters params(atoi(argv[4]));
    Model model(params);
    model.loadModelCluster(argv[2]);
    model.classificate(argv[3]);
  }else if(option==string("-s")){
    Parameters params(atoi(argv[6]));
    Model model(params);
    cout << "load" << endl;
    model.load_text(argv[2]);
    model.read_pair(argv[3]);
    model.saveWords(argv[4]);
    model.save_model(argv[5]);
  }else if(option==string("-g")){ // add read_test in order to compute MRR in develop data
    Parameters params(atoi(argv[7]));
    params.k_ = atoi(argv[4]);
    params.threshold_ = atof(argv[5]);
    Model model(params);
    model.load_model(argv[2]);
    model.read_test(argv[3]);
    model.JointlyLearning(params.threshold_, atoi(argv[8]));
    model.saveModelCluster(argv[6]);
  }else if(option==string("--val")){ // add read_test in order to compute MRR in develop data
    Parameters params(atoi(argv[6]));
    params.k_ = atoi(argv[4]);
    params.threshold_ = atof(argv[5]);
    Model model(params);
    model.load_model(argv[2]);
    model.read_test(argv[3]);
    model.JointlyLearning(params.threshold_, 30);
  }else if(option==string("--fu")){ // k-means
    Parameters params(atoi(argv[7]));
    Model model(params);
    model.load_model(argv[2]);
    model.read_test(argv[3]);
    model.least_square(atoi(argv[4]), atoi(argv[5]));
    model.saveModelCluster(argv[6]);
  }else if(option==string("-i")){
    Parameters params(atoi(argv[4]));
    Model model(params);
    model.loadModelCluster(argv[2]);
    cerr << "writing....";
    model.writeWordsInCluster(argv[3]);
    cerr << endl << "finished!" << endl;
  }else if(option==string("-p")){
    Timer timer;
    Parameters params(atoi(argv[5]));
    Model model(params);
    model.loadModelCluster(argv[2]);
    model.read_test(argv[3]);
    model.writePair(argv[4]);
    cout << "finished in " << timer.seconds() << "sec." << endl;
  }else if(option==string("-debug")){
    Parameters params(atoi(argv[4]));
    Model model(params);
    model.loadModelCluster(argv[2]);
    model.read_test(argv[3]);
    //cout << model.MRR(model.tests_, false) << endl;
    cout << model.MRR_kmeans(model.tests_) << endl;
  }else{
    cerr << "set option." << endl;
    exit(0);
  }
  return 0;
}
