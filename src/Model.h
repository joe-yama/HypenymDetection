// Model.h
#ifndef MODEL_H_
#define MODEL_H_
#include "Cluster.h"
#include <unordered_map>
#include <random>
#include <thread>
#include <future>
#include <functional>
#include <cfloat>
#include <algorithm>
#include <utility>

 namespace coin{
   typedef struct classificational_word_pair{
     Word* hypo;
     Word* hype;
     int inzicator;
     double sim;
     bool label;
   } Testdata;
   class Grads final{
   public:
    Grads(Matrix phi_grad, double bias_grad): phi_grad_(phi_grad), bias_grad_(bias_grad){}
     Matrix phi_grad_;
     double bias_grad_;
   };
   
   class Model final{
   private:
     Parameters& params_;
     int dim_;
     std::unordered_map<int, Isa*> isa_map_;
     std::vector<Word*> words_;
     std::unordered_map<int, Word*> word_map_;
     std::unordered_map<int, std::string> id_to_string_;
     std::unordered_map<std::string, int> string_to_id_;
     std::vector<Isa*> trains_;
     std::vector<Cluster*> clusters_;
     std::unordered_map<int, Cluster*> cluster_map_;
   public:
   Model(Parameters& params): params_(params), dim_(params_.dim_){}
     ~Model();
     std::vector<Isa*> tests_;
    std::vector<std::string> split(std::string& s, std::string c);
    void save_model(const std::string& filename);
    void load_model(const std::string& filename);
    void saveWords(const std::string& filename);
    void loadWords(const std::string& filename);
    void saveClusters(const std::string& filename);
    void loadClusters(const std::string& filename);
    void saveModelCluster(const std::string& filename);
    void loadModelCluster(const std::string& filename);
    void load_text(const std::string& filename);
    bool InModel(const std::string& word);
    Word* instance(const std::string& word);
    void read_pair(const std::string& filename);
    void read_test(const std::string& filename);
    void writeWordsInCluster(const std::string& filename);
    std::pair<int,double> BelongingCluster(Isa* isa);
    void OutputInsideClusters(const std::string& filename);
    void writePair(const std::string& filename);
    void least_square(int k, int max_iterations);
    void kmeans_init(int k);
    void recompute_global_means();
    void kmeans_clustering(int k);
    double LeastSquareAdam(Isa* isa, Cluster* c);
    
    void InitFirstCluster();
    double ObjectiveValue();
    void Move(Isa* isa, int new_id);
    void Generate(Isa* isa);
    void shuffle();
    void JointlyLearning(double threshold, int max_iterations);
    int MostClosestCluster(Isa* isa);
    Word* RandomWord();
    Word* ClosestNegWord(Isa* isa, Cluster* c);
    void demonstrate(bool);
    void demonstrate_kmeans();
    Eigen::MatrixXd GetPosPhiGrad(Word* hypo, Word* hype, double sim);
    Eigen::MatrixXd GetNegPhiGrad(Word* hypo, Word* hype, double neg_sim);
    double GetPosBiasGrad(double sim);
    double GetNegBiasGrad(double neg_sim);
    Grads NegGrads(Isa* isa, Cluster* c, int n);
    void AdamUpdate(Isa* isa, Cluster* c);

    double MRR(std::vector<coin::Isa*>& tests, bool square_loss);
    double MRR_kmeans(std::vector<coin::Isa*>& tests);
    double classificate(const std::string& filename);
    
    double sigmoid(const double& d){
      return 1. / (1. + exp(-d));
    }
  };


  

} // namespace coin
#endif
