#include "../include/B4aEventAction.hh"

#include "G4Event.hh"
#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

B4aEventAction::B4aEventAction(NtupleEntry* ntuple, SensorDescriptions const& sensors) :
  G4UserEventAction(),
  ntuple_{ntuple},
  sensors_{sensors}
{
  if (ntuple_ != nullptr)
    ntuple_->clear();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

B4aEventAction::~B4aEventAction()
{
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void
B4aEventAction::BeginOfEventAction(const G4Event*)
{  
  if (ntuple_ != nullptr)
    return;

  // ntuples can only be filled in events - otherwise I'd do this in DetectorConstruction

  auto* analysisManager{G4AnalysisManager::Instance()};

  for (auto const& sensor : sensors_) {
    analysisManager->FillNtupleIColumn(0, sensor.id);
    analysisManager->FillNtupleFColumn(1, sensor.pos.x() / cm);
    analysisManager->FillNtupleFColumn(2, sensor.pos.y() / cm);
    analysisManager->FillNtupleFColumn(3, sensor.pos.z() / cm);
    analysisManager->FillNtupleFColumn(4, sensor.dxy / cm);
    analysisManager->FillNtupleFColumn(5, sensor.dz / cm);
    analysisManager->AddNtupleRow();
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void
B4aEventAction::EndOfEventAction(const G4Event*)
{
  if (ntuple_ == nullptr)
    return;

  auto* analysisManager{G4AnalysisManager::Instance()};

  analysisManager->FillNtupleIColumn(ntuple_->columnId[NtupleEntry::cGenPid], ntuple_->genPid);
  analysisManager->FillNtupleDColumn(ntuple_->columnId[NtupleEntry::cGenEnergy], ntuple_->genEnergy);
  analysisManager->FillNtupleDColumn(ntuple_->columnId[NtupleEntry::cGenX], ntuple_->genX);
  analysisManager->FillNtupleDColumn(ntuple_->columnId[NtupleEntry::cGenY], ntuple_->genY);
  analysisManager->AddNtupleRow();

  ntuple_->clear();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
