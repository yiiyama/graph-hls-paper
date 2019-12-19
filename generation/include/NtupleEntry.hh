#ifndef NtupleEntry_h
#define NtupleEntry_h

#include "B4Analysis.hh"

#include <algorithm>

struct NtupleEntry {
  NtupleEntry(unsigned nsens) { recoEnergy.resize(nsens); }
  ~NtupleEntry() {}

  void clear() {
    genPid = 0;
    genEnergy = genX = genY = 0.;
    recoEnergy.assign(recoEnergy.size(), 0.);
  }

  void createColumns(G4AnalysisManager&);

  enum Column {
    cGenPid,
    cGenEnergy,
    cGenX,
    cGenY,
    cRecoEnergy,
    nColumns
  };

  G4int genPid{};
  G4double genEnergy{};
  G4double genX{};
  G4double genY{};
  std::vector<double> recoEnergy{};

  G4int columnId[nColumns]{};
};

#endif
