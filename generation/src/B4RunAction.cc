#include "../include/B4RunAction.hh"
#include "../include/B4DetectorConstruction.hh"

#include "G4Run.hh"
#include "G4RunManager.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

B4RunAction::B4RunAction() :
  G4UserRunAction()
{ 
  auto* analysisManager{G4AnalysisManager::Instance()};
  G4cout << "Using " << analysisManager->GetType() << G4endl;

  analysisManager->SetVerboseLevel(1);
  analysisManager->SetNtupleMerging(true);
  analysisManager->SetNtupleRowWise(false);


  // if (nSensors != 0) {
  //   analysisManager->CreateNtuple("events", "");

  //   ntuple_.reset(new NtupleEntry(nSensors));
  //   ntuple_->createColumns(*analysisManager);

  //   analysisManager->FinishNtuple();
  // }

  G4cout << "run action initialised" << G4endl;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

B4RunAction::~B4RunAction()
{
  delete G4AnalysisManager::Instance();  
}

void B4RunAction::saveGeometry()
{
  auto* analysisManager{G4AnalysisManager::Instance()};
  analysisManager->CreateNtuple("detector", "");
 
  analysisManager->CreateNtupleIColumn("id");
  analysisManager->CreateNtupleFColumn("x");
  analysisManager->CreateNtupleFColumn("y");
  analysisManager->CreateNtupleFColumn("z");
  analysisManager->CreateNtupleFColumn("dxy");
  analysisManager->CreateNtupleFColumn("dz");
 
  analysisManager->FinishNtuple();
}

void B4RunAction::bookNtuple(unsigned nSensors)
{
  auto* analysisManager{G4AnalysisManager::Instance()};
  analysisManager->CreateNtuple("events", "");

  ntuple_.reset(new NtupleEntry(nSensors));
  ntuple_->createColumns(*analysisManager);

  analysisManager->FinishNtuple();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void B4RunAction::BeginOfRunAction(const G4Run*)
{ 
  if (!fname_.empty())
    G4AnalysisManager::Instance()->OpenFile(fname_);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void B4RunAction::EndOfRunAction(const G4Run*)
{
  auto* analysisManager{G4AnalysisManager::Instance()};
  analysisManager->Write();
  analysisManager->CloseFile();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
