#include "../include/B4aEventAction.hh"

#include "G4Event.hh"
#include "G4UnitsTable.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

B4aEventAction::B4aEventAction(NtupleEntry& ntuple) :
  G4UserEventAction(),
  ntuple_(ntuple)
{
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

B4aEventAction::~B4aEventAction()
{
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void
B4aEventAction::BeginOfEventAction(const G4Event*)
{  
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void
B4aEventAction::EndOfEventAction(const G4Event*)
{
  auto* analysisManager{G4AnalysisManager::Instance()};

  analysisManager->FillNtupleIColumn(ntuple_.columnId[NtupleEntry::cGenPid], ntuple_.genPid);
  analysisManager->FillNtupleDColumn(ntuple_.columnId[NtupleEntry::cGenEnergy], ntuple_.genEnergy);
  analysisManager->FillNtupleDColumn(ntuple_.columnId[NtupleEntry::cGenX], ntuple_.genX);
  analysisManager->FillNtupleDColumn(ntuple_.columnId[NtupleEntry::cGenY], ntuple_.genY);
  analysisManager->AddNtupleRow();  
}  

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
