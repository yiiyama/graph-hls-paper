#include "TTree.h"
#include "TFile.h"
#include "TVector2.h"

#include <vector>
#include <array>
#include <memory>
#include <cmath>
#include <unordered_map>
#include <iostream>
#include <algorithm>

template<class V>
class InputPtr {
public:
  InputPtr() : up_(std::unique_ptr<V>(new V())), addr_(up_.get()) {}
  V** addr() { return &addr_; }
  V& operator*() { return *up_; }
  V* operator->() { return up_.get(); }

private:
  std::unique_ptr<V> up_;
  V* addr_;
};

void
convert(TTree* _input, char const* _outputFileName, double _minPt = 5., long _nEntries = -1)
{
  auto* outputFile(TFile::Open(_outputFileName, "recreate"));
  auto* output(new TTree("tree", ""));

  constexpr unsigned long maxHitsPerCluster(256);
  
  char truth_label;
  short n_cell;
  float features[maxHitsPerCluster][4]; // energy, theta, phi, z

  float cl_pt;
  float cl_eta;
  float cl_phi;
  float g_pt;
  float g_eta;
  float g_phi;

  output->Branch("x", features, TString::Format("x[%lu][4]/F", maxHitsPerCluster));
  output->Branch("n", &n_cell, "n/S");
  output->Branch("y", &truth_label, "y/B");
  output->Branch("cl_pt", &cl_pt, "cl_pt/F");
  output->Branch("cl_eta", &cl_eta, "cl_eta/F");
  output->Branch("cl_phi", &cl_phi, "cl_phi/F");
  output->Branch("gen_pt", &g_pt, "gen_pt/F");
  output->Branch("gen_eta", &g_eta, "gen_eta/F");
  output->Branch("gen_phi", &g_phi, "gen_phi/F");

  _input->SetBranchStatus("*", false);

  typedef std::vector<float> VFloat;
  typedef std::vector<int> VInt;
  typedef std::vector<uint32_t> VID;

  InputPtr<VFloat> gen_eta;
  InputPtr<VFloat> gen_phi;
  InputPtr<VFloat> gen_pt;
  InputPtr<VInt> gen_pdgid;
  InputPtr<VFloat> cl3d_pt;
  InputPtr<VFloat> cl3d_eta;
  InputPtr<VFloat> cl3d_phi;
  InputPtr<std::vector<VID>> cl3d_clusters_id;
  InputPtr<VID> tc_id;
  InputPtr<VFloat> tc_energy;
  InputPtr<VFloat> tc_x;
  InputPtr<VFloat> tc_y;
  InputPtr<VFloat> tc_z;

  _input->SetBranchAddress("gen_eta", gen_eta.addr());
  _input->SetBranchAddress("gen_phi", gen_phi.addr());
  _input->SetBranchAddress("gen_pt", gen_pt.addr());
  _input->SetBranchAddress("gen_pdgid", gen_pdgid.addr());
  _input->SetBranchAddress("hmVRcl3d_pt", cl3d_pt.addr());
  _input->SetBranchAddress("hmVRcl3d_eta", cl3d_eta.addr());
  _input->SetBranchAddress("hmVRcl3d_phi", cl3d_phi.addr());
  _input->SetBranchAddress("hmVRcl3d_clusters_id", cl3d_clusters_id.addr());
  _input->SetBranchAddress("tc_id", tc_id.addr());
  _input->SetBranchAddress("tc_energy", tc_energy.addr());
  _input->SetBranchAddress("tc_x", tc_x.addr());
  _input->SetBranchAddress("tc_y", tc_y.addr());
  _input->SetBranchAddress("tc_z", tc_z.addr());

  long iEntry(0);
  while (iEntry != _nEntries && _input->GetEntry(iEntry++) > 0) {
    if (iEntry % 10000 == 1)
      std::cout << iEntry << std::endl;

    std::unordered_map<uint32_t, unsigned> cellMap;

    for (unsigned iG(0); iG != gen_pt->size(); ++iG) {
      double absEta(std::abs((*gen_eta)[iG]));
      if (absEta < 1.4 || absEta > 2.8)
        continue;
      
      unsigned iMatched(-1);
      double bestDPt(10000.);

      for (unsigned iC(0); iC != cl3d_pt->size(); ++iC) {
        if ((*cl3d_pt)[iC] < _minPt)
          continue;

        double dEta((*gen_eta)[iG] - (*cl3d_eta)[iC]);
        double dPhi(TVector2::Phi_mpi_pi((*gen_phi)[iG] - (*cl3d_phi)[iC]));
        if (dEta * dEta + dPhi * dPhi < 0.01) {
          double dPt(std::abs((*cl3d_pt)[iC] - (*gen_pt)[iG]));
          if (dPt < bestDPt) {
            iMatched = iC;
            bestDPt = dPt;
          }
        }
      }

      if (iMatched >= cl3d_pt->size())
        continue;

      if (cellMap.empty()) {
        for (unsigned iT(0); iT != tc_id->size(); ++iT)
          cellMap.emplace((*tc_id)[iT], iT);
      }

      cl_pt = (*cl3d_pt)[iMatched];
      cl_eta = std::abs((*cl3d_eta)[iMatched]);
      cl_phi = (*cl3d_phi)[iMatched];
      double cl_theta(2. * std::atan(std::exp(-cl_eta)));

      unsigned absPid(std::abs((*gen_pdgid)[iG]));
      if (absPid == 11 || absPid == 22)
        truth_label = 1;
      else
        truth_label = 0;

      auto& constituents((*cl3d_clusters_id)[iMatched]);

      n_cell = std::min(constituents.size(), maxHitsPerCluster);

      std::fill(static_cast<float*>(&features[0][0]), static_cast<float*>(&features[maxHitsPerCluster][0]), 0.);

      for (short iD(0); iD != n_cell; ++iD) {
        unsigned iT(cellMap.at(constituents[iD]));

        double x((*tc_x)[iT]);
        double y((*tc_y)[iT]);
        double z(std::abs((*tc_z)[iT])); // projecting all to positive z
        double r(std::sqrt(x * x + y * y));
        double theta(std::atan2(r, z));
        double phi(std::atan2(y, x));

        features[iD][0] = std::sqrt((*tc_energy)[iT]);
        features[iD][1] = theta - cl_theta;
        features[iD][2] = phi - cl_phi;
        features[iD][3] = (z - 300.) / 200.;
      }

      output->Fill();
    }
  }
  
  outputFile->cd();
  output->Write();
  delete outputFile;
}
