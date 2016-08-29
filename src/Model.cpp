// Model.cpp

#include "Model.h"
#include <iomanip>
using namespace coin;
using namespace std;
using namespace Eigen;

coin::Model::~Model(){
  for(Word* word : words_) delete word;
  for(auto& iter : isa_map_) delete iter.second;
  for(Cluster* cluster : clusters_) delete cluster;
}

vector<string> coin::Model::split(string& s, string c){
  vector<string> ret;
  for(size_t i= 0, n; i <= s.length(); i=n+1){
    n = s.find_first_of(c, i);
    if(n == string::npos) n = s.length();
    assert(n >= i);
    string tmp = s.substr(i, n-i);
    ret.push_back(tmp);
  }
  return ret;
}

void coin::Model::saveWords(const string& filename){
  ofstream os(filename, ios::binary | ios::out | ios::trunc);
  if(!os){
    cout << "can't open file: " << filename << endl;
    exit(0);
  }
  os.write((char *)&dim_, sizeof(int));
  size_t nwords = words_.size();
  os.write((char *)&nwords, sizeof(nwords));
  for(Word* word : words_){
    word->save(os);    
  }
  os.close();
}

void coin::Model::loadWords(const string& filename){
  ifstream is(filename, ios::binary | ios::in);
  if(!is){
    cout << "can't open file: " << filename << endl;
    exit(0);
  }
  is.read((char *)&dim_, sizeof(int));
  size_t nwords;
  is.read((char *)&nwords, sizeof(nwords));
  for(unsigned int i=0; i < nwords; ++i){
    Word* w = new Word(params_, is);
    words_.push_back(w);
    word_map_[w->id()] = w;
    id_to_string_[w->id()] = w->w_str();
    string_to_id_[w->w_str()] = w->id();
    if(i+1 == nwords) break;
  }
  int nisa = 0;
  for(Word* word : words_){
    if(word->hype_id() == -1){
      continue;
    }else{
      nisa++;
      Word* hype = word_map_[word->hype_id()];
      Isa* isa = new Isa(params_, word, hype);
      assert(isa->hypo()->hype_id() == isa->hype()->id());
      isa_map_[word->id()] = isa;
      trains_.push_back(isa);
    }
  }
  is.close();
}

void coin::Model::saveClusters(const string& filename){
  ofstream os(filename, ios::binary | ios::out | ios::trunc);
  if(!os){
    cout << "can't open file: " << filename << endl;
    exit(0);
  }
  size_t nclusters = clusters_.size();
  os.write((char *)&nclusters, sizeof(nclusters));
  for(Cluster* c : clusters_){
    c->save(os);
  }
  os.close();
}

void coin::Model::loadClusters(const string& filename){
  ifstream is(filename, ios::binary | ios::in);
  if(!is){
    cout << "can't open file: " << filename << endl;
    exit(0);
  }
  size_t nclusters = 0;
  is.read((char *)&nclusters, sizeof(nclusters));
  for(size_t i=0; i<nclusters; ++i){
    Cluster* c = new Cluster(params_, is, i+1);
    clusters_.push_back(c);
    cluster_map_[i+1] = c;
  }
  is.close();
}

void coin::Model::save_model(const string& filename){
  ofstream os(filename, ios::binary | ios::out | ios::trunc);
  if(!os){
    cout << "can't open file: " << filename << endl;
    exit(0);
  }
  os.write((char *)&dim_, sizeof(int));
  size_t nwords = words_.size();
  os.write((char *)&nwords, sizeof(nwords));
  for(Word* word : words_){
    word->save(os);    
  }
  int end = 2015;
  os.write((char *)&end, sizeof(int));
  os.close();
}

void coin::Model::saveModelCluster(const string& filename){
  ofstream os(filename, ios::binary | ios::out | ios::trunc);
  if(!os){
    cout << "can't open file: " << filename << endl;
    exit(0);
  }
  os.write((char *)&dim_, sizeof(int));
  size_t nwords = words_.size();
  os.write((char *)&nwords, sizeof(nwords));
  for(Word* word : words_){
    word->save(os);    
  }
  size_t nclusters = clusters_.size();
  os.write((char *)&nclusters, sizeof(nclusters));
  for(Cluster* c : clusters_){
    c->save(os);
  }
  params_.save(os);
  int end = 2015;
  os.write((char *)&end, sizeof(int));
  os.close();
}

void coin::Model::loadModelCluster(const string& filename){
  ifstream is(filename, ios::binary | ios::in);
  if(!is){
    cout << "can't open file: " << filename << endl;
    exit(0);
  }
  is.read((char *)&dim_, sizeof(int));
  size_t nwords;
  is.read((char *)&nwords, sizeof(nwords));
  cout << nwords << "words exist." << endl;
  for(unsigned int i=0; i < nwords; ++i){
    Word* w = new Word(params_, is);
    words_.push_back(w);
    word_map_[w->id()] = w;
    id_to_string_[w->id()] = w->w_str();
    string_to_id_[w->w_str()] = w->id();
    if(i+1 == nwords) break;
  }
  int nisa = 0;
  for(Word* word : words_){
    if(word->hype_id() == -1){
      continue;
    }else{
      nisa++;
      Word* hype = word_map_[word->hype_id()];
      Isa* isa = new Isa(params_, word, hype);
      assert(isa->hypo()->hype_id() == isa->hype()->id());
      isa_map_[word->id()] = isa;
      trains_.push_back(isa);
    }
  }
  size_t nclusters = 0;
  is.read((char *)&nclusters, sizeof(nclusters));
  for(size_t i=0; i<nclusters; ++i){
    Cluster* c = new Cluster(params_, is, i+1);
    clusters_.push_back(c);
    cluster_map_[i+1] = c;
  }
  params_.load(is);
  int end = 2015;
  is.read((char *)&end, sizeof(int));
  assert(end == 2015);
  is.close();
}

void coin::Model::load_model(const string& filename){
  ifstream is(filename, ios::binary | ios::in);
  if(!is){
    cout << "can't open file: " << filename << endl;
    exit(0);
  }
  is.read((char *)&dim_, sizeof(int));
  size_t nwords;
  is.read((char *)&nwords, sizeof(nwords));
  for(unsigned int i=0; i < nwords; ++i){
    Word* w = new Word(params_, is);
    words_.push_back(w);
    word_map_[w->id()] = w;
    id_to_string_[w->id()] = w->w_str();
    string_to_id_[w->w_str()] = w->id();
    if(i+1 == nwords) break;
  }
  int nisa = 0;
  for(Word* word : words_){
    if(word->hype_id() == -1){
      continue;
    }else{
      nisa++;
      Word* hype = word_map_[word->hype_id()];
      Isa* isa = new Isa(params_, word, hype);
      assert(isa->hypo()->hype_id() == isa->hype()->id());
      isa_map_[word->id()] = isa;
      trains_.push_back(isa);
    }
  }
  int end = 2015;
  is.read((char *)&end, sizeof(int));
  assert(end == 2015);
  is.close();
}

void coin::Model::load_text(const string& filename){
  ifstream ifs(filename.c_str());
  if(ifs.fail()){
    cout << "can't open file: " << filename << endl;
    exit(0);
  }
  int nwords = 0;
  string line;
  while(getline(ifs, line)){
    nwords++;
    vector<string> elements = split(line, string(" "));
    if(elements.size() < unsigned(dim_+1)) continue;
    string word = elements[0];
    cerr << word << endl;
    int id = nwords;
    Vector vec = Vector(dim_);
    for(int i=0; i<dim_; i++){
      vec(i) = atof(elements[i+1].c_str());
    }
    vec.normalize();
    Word* w = new Word(params_, id, vec, word);
    words_.push_back(w);
    word_map_[w->id()] = w;
    id_to_string_[w->id()] = w->w_str();
    string_to_id_[w->w_str()] = w->id();
  }
}

bool coin::Model::InModel(const string& word){
  if(string_to_id_.find(word) == string_to_id_.end()){
    return false;
  }else{
    return true;
  }
}

coin::Word* coin::Model::instance(const string& word){
  if(string_to_id_.find(word) == string_to_id_.end()){
    return nullptr;
  }else{
    return word_map_[string_to_id_[word]];
  }
}

void coin::Model::read_pair(const string& filename){
  string line;
  ifstream ifs(filename.c_str());
  if(ifs.fail()){
    cout << "can't open file: " << filename << endl;
    exit(0);
  }
  int in_model = 0, npair = 0;
  while(getline(ifs, line)){
    vector<string> element = split(line, string("\t"));
    string hypernym = element[0], hyponym = element[1];
    if(InModel(hypernym) && InModel(hyponym)){
      Word* hypo = instance(hyponym);
      Word* hype = instance(hypernym);
      hypo->setHypeId(hype->id());
      Isa* isa = new Isa(params_, hypo, hype);
      trains_.push_back(isa);
      isa_map_[hypo->id()] = isa;
      in_model++;
    }
    npair++;
  }
}

void coin::Model::read_test(const string& filename){
  string line;
  ifstream ifs(filename.c_str());
  if(ifs.fail()){
    cout << "can't open file: " << filename << endl;
    exit(0);
  }
  int in_model = 0, npair = 0;
  while(getline(ifs, line)){
    vector<string> element = split(line, string("\t"));
    string hypernym = element[0], hyponym = element[1];
    if(InModel(hypernym) && InModel(hyponym)){
      Word* hypo = instance(hyponym);
      Word* hype = instance(hypernym);
      Isa* isa = new Isa(params_, hypo, hype);
      tests_.push_back(isa);
      isa_map_[hypo->id()] = isa;
      in_model++;
    }
    npair++;
  }
  //  cout << "MRR: " << MRR(tests_, false) << endl;
}  

void coin::Model::writePair(const string& filename){
  ofstream ofs(filename.c_str());
  cout << "#train: " << trains_.size() << endl;
  for(Isa* isa : trains_){
    ofs << isa->hype()->w_str() << "\t" << isa->hypo()->w_str() << endl;
  }
  cout << "#validation: " << tests_.size() << endl;
  for(Isa* isa : tests_){
    ofs << isa->hype()->w_str() << "\t" << isa->hypo()->w_str() << endl;
  }
}
  
void coin::Model::writeWordsInCluster(const string& filename){
  //  string foo;
  //cout << clusters_.size() << endl;
  //  cin >> foo;
  for(Cluster* c : clusters_){
    string dest = filename + "_" + std::to_string(c->id());
    ofstream ofs(dest.c_str());
    const vector<int>& list = c->list();
    for(const int id : list){
      ofs << word_map_[id]->w_str() << endl;
    }
  }
}

pair<int,double> coin::Model::BelongingCluster(Isa* isa){
  vector<pair<int,double>> list;
  for(Cluster* c : clusters_){
    Vector estimated = isa->hypo()->vector()*c->matrix();
    const double& bias = c->bias();
    pair<int,double> id_sim(c->id(), sigmoid(estimated.dot(isa->hype()->vector()) + bias));
    list.push_back(id_sim);
  }
  return *max_element(list.begin(), list.end(), [](const pair<int, double>& lhs, const pair<int, double>& rhs){ return lhs.second > rhs.second; });
}

void coin::Model::OutputInsideClusters(const string& filename){
  ofstream ofs(filename.c_str());
  for(auto id_isa : isa_map_){
    pair<int, double> id_sim = BelongingCluster(id_isa.second);
    // cluster_id TAB similality TAB hypernym TAB hyponym
    ofs << id_sim.first << "\t" << id_sim.second << "\t" << id_isa.second->hype()->w_str() << "\t" << id_isa.second->hypo()->w_str() << endl;
  }
  
}

double coin::Model::LeastSquareAdam(Isa* isa, Cluster* c){
  const int dim = params_.dim_;
  const Matrix& phi = c->matrix();
  const Vector& x = isa->hypo()->vector();
  const Vector& y = isa->hype()->vector();
  Matrix grad = Matrix(dim, dim);
  Vector gap = (x*phi) - y;
  for(int row = 0; row < dim; row++){
    for(int col = 0; col < dim; col++){
      grad(row, col) =  2 * gap(col) * x(row);
    }
  }
  c->update(grad, 0);
  return gap.squaredNorm();
}

void coin::Model::kmeans_init(int k){
  // make #k clusters
  for(int i=1; i <= k; ++i){
    Cluster* c = new Cluster(params_, i);
    c->KmeansInit();
    clusters_.push_back(c);
    cluster_map_[i] = c;
  }
}

void coin::Model::recompute_global_means(){
  for(Cluster* c : clusters_){
    Vector mean = Vector::Zero(params_.dim_);
    for(auto id : c->list()){
      Isa* isa = isa_map_[id];
      mean += isa->hype()->vector() - isa->hypo()->vector();
    }
    mean = mean.array() / c->list().size();
    c->setMean(mean);
  }
}


void coin::Model::kmeans_clustering(int k){
  kmeans_init(k);
  int iters = 0;
  while(1){
    ++iters;
    shuffle();
    int move = 0;
    for(Isa* isa : trains_){
      double close_d = DBL_MAX;
      int close_id = 0;
      for(Cluster* c : clusters_){
	double distance = c->distance(isa);
	if(distance < close_d){
	  close_d = distance;
	  close_id = c->id();
	}
      }
      if(close_id != isa->indicator()){
	move++;
	if(isa->indicator() != -1) cluster_map_[isa->indicator()]->del(isa->id());
	isa->setIndicator(close_id);
	cluster_map_[isa->indicator()]->add(isa->id());
      }
    }
    if(move == 0) break; // converge
    recompute_global_means();
  }
  cout << "kmeans converged in " << iters << " itarations." << endl
       << "total clusters: " << clusters_.size() << endl;
  for(unsigned i = 0; i < clusters_.size(); ++i){
    cout << "#" << i << " : " << clusters_[i]->list().size() << endl;
  }
}

void coin::Model::least_square(int k, int max_iterations){
  kmeans_clustering(k);
  //Adam optimization
  int iters = 0;
  cout << "optimization begin." << endl;
  while(iters < max_iterations){
    Timer time;
    double loss = 0.;
    ++iters;
    cout << "[iter " << iters << "] ";
    for(Cluster* c : clusters_){
      c->shuffle();
      for(auto id : c->list()){
	loss += LeastSquareAdam(isa_map_[id], c); //update cluster
      }
    }
    cout << "loss= " << loss <<  " (" << time.seconds() << "sec.) " << "MRR: " << MRR(tests_, true) << endl;
  }
}

double coin::Model::classificate(const string& filename){
  ifstream ifs(filename.c_str());
  if(ifs.fail()){
    cout << "can't open file: " << filename << endl;
    exit(0);
  }
  string line;
  int ntests = 0;
  vector<Testdata> tests;
  while(getline(ifs, line)){
    vector<string> elements = split(line, string("\t"));
    Word* hypo = instance(elements[0]);
    Word* hype = instance(elements[1]);
    if((hypo == nullptr)||(hype == nullptr)) continue;
    else{
      Testdata test;
      test.hypo = hypo;
      test.hype = hype;
      if(elements[2].find(string("T")) != string::npos) test.label = true;
      else if(elements[2].find(string("F")) != string::npos) test.label = false;
      tests.push_back(test);
      ntests++;
      // compute similarity
      vector<double> sims(clusters_.size());
      for(unsigned ci = 0; ci < clusters_.size(); ci++){
	Cluster* c = clusters_[ci];
	sims[ci] = sigmoid((test.hypo->vector()*c->matrix()).dot(test.hype->vector()) + c->bias());
      }
      test.sim = *max_element(sims.begin(), sims.end());
    }
  }
  cout << ntests << "pairs" << endl;
  for(Testdata t : tests){
    cout << t.hype->w_str() << "\t" << t.hypo->w_str() << "\t" << t.sim << "\t" << int(t.label) << endl;
  }
  // test for each threshold
  vector<double> ps;
  vector<double> rs;
  vector<double> fs;
  vector<double> ds;
  cout << "threshold" << "\t" << "precision" << "\t" << "recall" << "\t" << "f-score" << endl;
  for(double d = 0; d < 1.; d += 0.01){
    ds.push_back(d);
    int tp = 0;
    int tn = 0;
    int fp = 0;
    int fn = 0;
    for(Testdata t : tests){
      if(t.label == true){
	if(t.sim >= d) tp++;
	else fn++;
      }else{
	if(t.sim >= d) fp++;
	else tn++;
      }	
    }
    double p = double(tp) / double(tp + fp);
    double r = double(tp) / double(tp + fn);
    double f = 2.*r*p / (r + p);
    ps.push_back(p);
    rs.push_back(r);
    fs.push_back(f);
    cout << d << "\t" << p << "\t" << r << "\t" << f << "\t" << tp << " " << fp << " " << fn << " " << tn << endl;
  }
  unsigned max_p_id = max_element(ps.begin(), ps.end()) - ps.begin();
  unsigned max_r_id = max_element(rs.begin(), rs.end()) - rs.begin();
  unsigned max_f_id = max_element(fs.begin(), fs.end()) - fs.begin();
  cout << endl
       << "----------" << endl
       << endl
       << "* max precision:" << endl
       << ds[max_p_id] << "\t" << ps[max_p_id] << "\t" << rs[max_p_id] << "\t" << fs[max_p_id] << endl
       << "* max recall:" << endl
       << ds[max_r_id] << "\t" << rs[max_r_id] << "\t" << rs[max_r_id] << "\t" << fs[max_r_id] << endl
       << "* max f-score:" << endl
       << ds[max_f_id] << "\t" << fs[max_f_id] << "\t" << rs[max_f_id] << "\t" << fs[max_f_id] << endl;
  return fs[max_f_id];
}

double coin::Model::MRR_kmeans(vector<Isa*>& tests){
  int nwords = words_.size();
  int nclusters = clusters_.size();
  int ntests = tests_.size();
  vector<int> ranks(ntests);
  vector<Vector> means(nclusters);
  for(int ci = 0; ci < nclusters; ++ci){
    means[ci] = Vector::Zero(dim_);
    for(auto id : clusters_[ci]->list()){
      const Vector& y = isa_map_[id]->hype()->vector();
      means[ci] += y - word_map_[id]->vector();
    }
    means[ci].array() = means[ci].array() / clusters_[ci]->list().size();
  }
  for(int ti = 0; ti < ntests; ++ti){
    Isa* isa = tests_[ti];
    Word* hypo = isa->hypo();
    Word* correct_hype = isa->hype();
    cout << hypo->w_str() << "\t" << correct_hype->w_str() << endl;
    vector<pair<int,double>> estimates(nwords);
    #pragma omp parallel for
    for(int wi = 0; wi < nwords; ++wi){
      Word* w = words_[wi];
      if(w == nullptr){
	estimates[wi] = pair<int, double>(-1, 100000);
	continue;
      }
      if(w->id() == hypo->id()){
	pair<int,double> me(wi, DBL_MAX);
	estimates[wi] = me;
	continue;
      }
      pair<int,double> max_c(-1, DBL_MAX);
      for(int ci = 0; ci < nclusters; ++ci){
	pair<int,double> c_sim(ci,(means[ci] - w->vector() + hypo->vector()).squaredNorm());
	if(c_sim.second < max_c.second || ci == 0){
	  max_c = c_sim;
	}
      }
      estimates[wi] =
	pair<int,double>(wi, (hypo->vector()*clusters_[max_c.first]->matrix() - w->vector()).squaredNorm());
    }
    
    sort(estimates.begin(), estimates.end(),
	 [](const pair<int,double>& lhs, const pair<int,double>& rhs)->bool{
	   return lhs.second < rhs.second;
	 });
    for(int i = 0; i < 5; ++i){
      cout << i+1 << " " << words_[estimates[i].first]->w_str() << "\t" << estimates[i].second << endl;
    }
    auto correct_it = find_if(estimates.begin(), estimates.end(),
	    [&](const pair<int,double>& ref)->bool{
	      return words_[ref.first]->w_str() == correct_hype->w_str(); });
    ranks[ti] = correct_it - estimates.begin() + 1;
    cout << "correct: " << ranks[ti] << endl << endl;
  }
  return accumulate(ranks.begin(), ranks.end(), 0,
		    [](const double sum, const int value)->double{
		      return sum + 1./double(value); }) / double(ntests);
  /*
  int ntests = tests_.size();
  int nclusters = clusters_.size();
  int nwords = words_.size();
  vector<int> ranks(ntests);
  vector<Vector> means(nclusters);
  for(int ci = 0; ci < nclusters; ++ci){
    means[ci] = Vector::Zero(dim_);
    for(auto id : clusters_[ci]->list()){
      const Vector& y = isa_map_[id]->hype()->vector();
      means[ci] += y - word_map_[id]->vector();
    }
    means[ci].array() = means[ci].array() / clusters_[ci]->list().size();
  }

  #pragma omp parallel for
  for(int ti = 0; ti < ntests; ++ti){
    Isa* isa = tests_[ti];
    Word* hypo = isa->hypo();
    Word* correct_hype = isa->hype();
    cout << hypo->w_str() << "\t" << correct_hype->w_str() << endl;
    vector<pair<int,double>> estimates(nwords);
    for(int wi = 0; wi < nwords; ++wi){
      Word* w = words_[wi];
      if(word_map_[wi]->w_str() == hypo->w_str()){
	pair<int,double> me(wi, 10000);
	estimates[wi] = me;
	continue;
      }
      pair<int,double> max_c(-1, 10000);
      for(int ci = 0; ci < nclusters; ++ci){
	pair<int,double> c_sim(ci,(means[ci] - w->vector() + hypo->vector()).squaredNorm());
	if(c_sim.second < max_c.second || ci == 0) max_c = c_sim;
      }
      estimates[wi] =
	pair<int,double>(wi, (hypo->vector()*clusters_[max_c.first]->matrix() - w->vector()).squaredNorm());
    }
    sort(estimates.begin(), estimates.end(),
	 [](const pair<int,double>& lhs, const pair<int,double>& rhs)->bool{
	   return lhs.second < rhs.second;
	 });
    vector<pair<int,double>>::iterator correct_iter = find_if(estimates.begin(), estimates.end(),
							      [&](const pair<int, double>& now)->bool{
								return words_[now.first]->w_str() == correct_hype->w_str();
							      });
    ranks[ti] = correct_iter - estimates.begin() + 1;
  }
  
  double mrr = 0.;
  for(int i = 0; i < ntests; ++i){
    mrr += 1./ranks[i];
  }
  cout << "(ntests=" << ntests << ")" << endl;
  return mrr / ntests;
  */
}

double coin::Model::MRR(vector<Isa*>& tests, bool square_loss = false){
  int ntests = tests_.size();
  int nclusters = clusters_.size();
  int nwords = words_.size();

  vector<int> ranks(ntests);
  #pragma omp parallel for
  for(int ti = 0; ti < ntests; ++ti){
    Isa* isa = tests_[ti];
    Word* hypo = isa->hypo();
    Word* correct_hype = isa->hype();
    vector<Vector> projecteds(nclusters);
    if(!square_loss){
      for(int i = 0; i < nclusters; ++i){
	projecteds[i] = hypo->vector()*clusters_[i]->matrix();
      }
    }else{
      for(int ci = 0; ci < nclusters; ++ci){
	projecteds[ci] = Vector::Zero(dim_);
	for(auto id : clusters_[ci]->list()){
	  projecteds[ci] += words_[words_[id]->hype_id()]->vector() - words_[id]->vector();
	}
	projecteds[ci].array() = projecteds[ci].array() / clusters_[ci]->list().size();
      }
    }
    vector<pair<int,double>> estimates(nwords);
    for(int wi = 0; wi < nwords; ++wi){
      Word* w = words_[wi];
      if(wi == hypo->id()){
	pair<int,double> me(wi, -10000.);
	estimates[wi] = me;
	continue;
      }
      pair<int,double> max_c(-1, 0.);
      for(int ci = 0; ci < nclusters; ++ci){
	pair<int,double> c_sim;
	if(square_loss){
	  c_sim = pair<int, double>(ci, 1 - sigmoid((projecteds[ci] - w->vector() + hypo->vector()).squaredNorm()) );
	}else c_sim = pair<int, double>(ci, projecteds[ci].dot(w->vector())+clusters_[ci]->bias());
	if(c_sim.second > max_c.second || ci == 0){
	  max_c = c_sim;
	}
      }
      pair<int,double> wi_sim(wi, max_c.second);
      if(square_loss) wi_sim = pair<int,double>(wi, 1-sigmoid((hypo->vector()*clusters_[max_c.first]->matrix() - w->vector()).squaredNorm()));
      estimates[wi] = wi_sim;
    }
    sort(estimates.begin(), estimates.end(),
	 [](const pair<int,double>& lhs, const pair<int,double>& rhs)->bool{
	   return lhs.second > rhs.second;
	 });
    vector<pair<int,double>>::iterator correct_iter = find_if(estimates.begin(), estimates.end(),
							      [&](const pair<int, double>& now)->bool{
								return words_[now.first]->w_str() == correct_hype->w_str();
							      });
    ranks[ti] = correct_iter - estimates.begin() + 1;
  }
  double mrr = 0.;
  for(int i = 0; i < ntests; ++i){
    mrr += 1./ranks[i];
  }
  cout << "(ntests=" << ntests << ")" << endl;
  return mrr / ntests;
}
 
void coin::Model::InitFirstCluster(){
  assert(clusters_.size() == 0);
  Cluster* c = new Cluster(params_, 1);
  clusters_.push_back(c);
  cluster_map_[1] = c; 
  for(Isa* isa : trains_){
    c->add(isa->id());
    AdamUpdate(isa, c);
  }
}

double coin::Model::ObjectiveValue(){
  double objval = 0.;
  for(Isa* isa : trains_){
    Cluster* c = cluster_map_[(isa->indicator())];
    objval += log(sigmoid(c->sim(isa)));
  }
  return objval;
}

Word* coin::Model::RandomWord(){
  return word_map_[Random::random_id(1, words_.size())];
}

int coin::Model::MostClosestCluster(Isa* isa){
  assert(clusters_.size() > 0);
  int max_id = 0;
  double max_sim = 0.;
  for(Cluster* c : clusters_){
    double sim = sigmoid(c->sim(isa));
    if(sim > max_sim){
      max_sim = sim;
      max_id = c->id();
    }
  }
  return max_id;
}

Word* coin::Model::ClosestNegWord(Isa* isa, Cluster* c){
  //double max = 0;
  //int id = 0;
  //int hype_id = isa->hype()->id();
  Vector xphi = isa->hypo()->vector() * c->matrix();
  double b = c->bias();
  vector<pair<int, double>> list(words_.size());
  #pragma omp parallel for
  for(unsigned int wi = 0; wi < words_.size(); ++wi){
    Word* w = words_[wi];
    //if(w->id() == hype_id) continue;
    //if(w->id() ==isa->hypo()->id()) continue;
    pair<int,double> w_sim(w->id(), xphi.dot(w->vector())+b);
    //double sim = sigmoid(xphi.dot(w->vector())+b);
    list[wi] = w_sim;
    /*if(sim>max){
      max = sim;
      id = w->id();
    }*/
  }
  sort(list.begin(), list.end(),
       [](const pair<int,double>& lhs, const pair<int,double>& rhs)->bool{
	 return lhs.second > rhs.second;});
  int i = 0;
  while(list[i].first == isa->hypo()->id() || list[i].first == isa->hype()->id()){
    ++i;
  }
  return word_map_[list[i].first];
}

void coin::Model::shuffle(){
  std::shuffle( trains_.begin(), trains_.end(), Random::engine() );
}

void coin::Model::Move(Isa* isa, const int new_id){
  cluster_map_[isa->indicator()]->del(isa->id());
  Cluster* dest = cluster_map_[new_id];
  dest->add(isa->id());
  AdamUpdate(isa, dest);
}

void coin::Model::Generate(Isa* isa){
  cluster_map_[isa->indicator()]->del(isa->id());
  Cluster* new_c = new Cluster(params_, clusters_.size()+1);
  clusters_.push_back(new_c);
  cluster_map_[new_c->id()] = new_c;
  new_c->add(isa->id());
  AdamUpdate(isa, new_c);
}

void coin::Model::JointlyLearning(double threshold, int max_iterations){
  vector<double> fscores(max_iterations);

  Timer init_timer;
  cout << "Jointly Learning" << endl
       << "  lambda: " << threshold << endl
       << "  max iteration: " << max_iterations << endl
       << "  #words: " << words_.size() << endl
       << "  #training isa pairs: " << trains_.size() << endl
       << "  #validation data: " << tests_.size() << endl;
  shuffle();
  InitFirstCluster();
  cout << "[Init] " << init_timer.seconds() << " sec." << endl;
  int iters = 0;
  while(++iters <= max_iterations){
    Timer iter_timer;
    cout << "[iter " << iters << "] ";
    double loss = 0.;
    int generate = 0;
    int move = 0;
    shuffle();
    for(Isa* isa : trains_){
      Cluster* close_c = cluster_map_[MostClosestCluster(isa)];
      if(sigmoid(close_c->sim(isa)) >= threshold){
	Move(isa, close_c->id());
	move++;
	loss += 1-sigmoid((isa->hypo()->vector()*close_c->matrix()).dot(isa->hype()->vector())+close_c->bias());
      }else{
	Generate(isa);
	generate++;
	loss += 1-sigmoid((isa->hypo()->vector()*(*(clusters_.end()-1))->matrix()).dot(isa->hype()->vector())+(*(clusters_.end()-1))->bias());
      }
    }
    cout << "loss: " << loss / double(trains_.size()) << " #clusters: " << clusters_.size();
    if(iters % 1 == 0) cout << " MRR: " << MRR(tests_, false);
    /* validation of classification
    double f = classificate("data/lexical_entailment/baroni2012/data_rnd_val.tsv");
    fscores[iters] = f;
    */
    cout << " (" << iter_timer.seconds() << " sec.)" << endl;
  }
  /* validation of classification
  auto max = max_element(fscores.begin(), fscores.end());
  int max_f_it = max - fscores.begin();
  cout << "* max f:" << endl
       << max_f_it << " " << *max << endl;
  */
}

void coin::Model::demonstrate_kmeans(){
  int nwords = words_.size();
  int nclusters = clusters_.size();
  string input;
  cout << "input word: " << endl;
  while(cin >> input){
    if(input == "q") break;
    if(InModel(input) == false){
      cout << "word is not in the model" << endl
	   << "input word: " << endl;
      continue;
    }
    Word* hypo = instance(input);
    vector<Vector> means(nclusters);
    for(int ci = 0; ci < nclusters; ++ci){
      means[ci] = Vector::Zero(dim_);
      for(auto id : clusters_[ci]->list()){
	const Vector& y = isa_map_[id]->hype()->vector();
	means[ci] += y - word_map_[id]->vector();
      }
      means[ci].array() = means[ci].array() / clusters_[ci]->list().size();
    }
    vector<pair<int,double>> estimates(nwords);
    #pragma omp parallel for
    for(int wi = 0; wi < nwords; ++wi){
      Word* w = words_[wi];
      if(wi == hypo->id()){
	pair<int,double> me(wi, DBL_MAX);
	estimates[wi] = me;
	continue;
      }
      pair<int,double> max_c(-1, DBL_MAX);
      for(int ci = 0; ci < nclusters; ++ci){
	pair<int,double> c_sim(ci,(means[ci] - w->vector() + hypo->vector()).squaredNorm());
	if(c_sim.second < max_c.second || ci == 0){
	  max_c = c_sim;
	}
      }
      estimates[wi] =
	pair<int,double>(wi, (hypo->vector()*clusters_[max_c.first]->matrix() - w->vector()).squaredNorm());
    }
    sort(estimates.begin(), estimates.end(),
	 [](const pair<int,double>& lhs, const pair<int,double>& rhs)->bool{
	   return lhs.second < rhs.second;
	 });
    int limit = 10;
    for(int ei = 0; ei < limit; ++ei){
      pair<int, double> est = estimates[ei];
      cout << words_[est.first]->w_str() << "\t" << est.second << endl;
    }
    cout << endl;
  }
}


void coin::Model::demonstrate(bool square_loss){
  int nwords = words_.size();
  int nclusters = clusters_.size();
  string input;
  cout << "input word: " << endl;
  while(cin >> input){
    if(input == "q") break;
    if(InModel(input) == false){
      cout << "word is not in the model" << endl
	   << "input word: " << endl;
      continue;
    }
    Word* hypo = instance(input);
    
    vector<Vector> projecteds(nclusters);
    for(int i = 0; i < nclusters; ++i){
      projecteds[i] = hypo->vector()*clusters_[i]->matrix();
    }
    vector<pair<int,double>> estimates(nwords);
    for(int wi = 0; wi < nwords; ++wi){
      Word* w = words_[wi];
      if(words_[wi]->w_str() == hypo->w_str()){
	pair<int,double> me(wi, -DBL_MAX);
	estimates[wi] = me;
	continue;
      }
      pair<int,double> max_c(-1, 0.);
      for(int ci = 0; ci < nclusters; ++ci){
	pair<int,double> c_sim;
	if(square_loss) c_sim = pair<int, double>(ci, 1-sigmoid((projecteds[ci] - w->vector()).squaredNorm()) );
	else c_sim = pair<int, double>(ci, sigmoid(projecteds[ci].dot(w->vector())+clusters_[ci]->bias()));
	if(c_sim.second > max_c.second || ci == 0){
	  max_c = c_sim;
	}
      }
      pair<int,double> wi_sim(wi, max_c.second);
      estimates[wi] = wi_sim;
    }
    sort(estimates.begin(), estimates.end(),
	 [](const pair<int,double>& lhs, const pair<int,double>& rhs)->bool{
	   return lhs.second > rhs.second;
	 });
    int limit = 10;
    for(int ei = 0; ei < limit; ++ei){
      pair<int, double> est = estimates[ei];
      cout << words_[est.first]->w_str() << "\t" << est.second << endl;
    }
    cout << endl;
  }
    
    /*
    vector<pair<Word*, double>> ests;
    if(square_loss){ // ks
      for(Cluster* c : clusters_){
	Vector xphi = hypo->vector() * c->matrix();
	for(Word* w : words_){
	  if(w->id() == hypo->id()) continue;	  
	  double distance = (xphi - w->vector()).squaredNorm();
	  pair<Word*,double> est(w, distance);
	  ests.push_back(est);
	}
      }
      sort(ests.begin(), ests.end(), [](const pair<Word*, double>& lhs, const pair<Word*, double>& rhs)->bool{ return lhs.second < rhs.second; });
      for(auto it = ests.begin(); it < ests.begin()+10; it++ ){
	cout << (*it).first->w_str() << "\t" << (*it).second << endl;
      }
    }else{
      for(Cluster* c : clusters_){
	Vector xphi = hypo->vector() * c->matrix();
	const double& b = c->bias();
	for(Word* w : words_){
	  if(w->id() == hypo->id()) continue;
	  double sim = sigmoid( xphi.dot(w->vector()) + b );
	  pair<Word*,double> est(w, sim);
	  ests.push_back(est);
	}
      }
      sort(ests.begin(), ests.end(), [](const pair<Word*, double>& lhs, const pair<Word*, double>& rhs)->bool{ return lhs.second > rhs.second; });
      for(auto it = ests.begin(); it < ests.begin()+10; it++ ){
	cout << (*it).first->w_str() << "\t" << (*it).second << endl;
      }
    }
    cout << endl;
    cout << "input word: " << endl;
  } // while
  */
}

Eigen::MatrixXd coin::Model::GetPosPhiGrad(Word* hypo, Word* hype, double sim){
  return (hypo->vector().transpose() * hype->vector()).array() * (sigmoid(sim)-1);
}

Eigen::MatrixXd coin::Model::GetNegPhiGrad(Word* hypo, Word* hype, double neg_sim){
  return (hypo->vector().transpose() * hype->vector()).array() * sigmoid(neg_sim);
}

double coin::Model::GetPosBiasGrad(double sim){
  return sigmoid(sim) - 1;
}

double coin::Model::GetNegBiasGrad(double neg_sim){
  return sigmoid(neg_sim);
}

Grads coin::Model::NegGrads(Isa* isa, Cluster* c , int n){
  Word* w1 = isa->hype();
  Word* w2 = isa->hypo();
  double neg_sim = c->sim(w1, w2);
  Grads grads = Grads(GetNegPhiGrad(w1, w2, neg_sim), GetNegBiasGrad(neg_sim));
  for(int i = 0; i < n; ++i){
    w1 = isa->hypo();
    w2 = RandomWord();
    neg_sim = c->sim(w1, w2);
    grads.phi_grad_ += GetNegPhiGrad(w1, w2, neg_sim);
    grads.bias_grad_ += GetNegBiasGrad(neg_sim);
  }
  return grads;
}
   
void coin::Model::AdamUpdate(Isa* isa, Cluster* c){
  isa->setIndicator(c->id());
  double pos_sim = c->sim(isa->hypo(), isa->hype());
  double bias_grad = GetPosBiasGrad(pos_sim);
  Matrix phi_grad = GetPosPhiGrad(isa->hypo(), isa->hype(), pos_sim);

  Word* neg_w = ClosestNegWord(isa, c);
  double neg_sim = c->sim(isa->hypo(), neg_w);
  phi_grad += GetNegPhiGrad(isa->hypo(), neg_w, neg_sim);
  bias_grad += GetNegBiasGrad(neg_sim);
  
  c->update(phi_grad, bias_grad);
}
