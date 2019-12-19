#include "../include/NtupleEntry.hh"

void
NtupleEntry::createColumns(G4AnalysisManager& manager)
{
  columnId[cGenPid] = manager.CreateNtupleIColumn("genPid");
  columnId[cGenEnergy] = manager.CreateNtupleDColumn("genEnergy");
  columnId[cGenX] = manager.CreateNtupleDColumn("genX");
  columnId[cGenY] = manager.CreateNtupleDColumn("genY");
  columnId[cRecoEnergy] = manager.CreateNtupleDColumn("recoEnergy", recoEnergy);
}
