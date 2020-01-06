#include "../include/B4aSteppingAction.hh"

#include "G4Step.hh"
#include "G4SystemOfUnits.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

B4aSteppingAction::B4aSteppingAction(NtupleEntry* ntuple, SensorDescriptions const& sensors) :
  G4UserSteppingAction(),
  ntuple_{ntuple}
{
  for (auto& sd : sensors) {
    sensors_[sd.sensor] = sd.id;
    absorbers_[sd.absorber] = sd.id;
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

B4aSteppingAction::~B4aSteppingAction()
{
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void
B4aSteppingAction::UserSteppingAction(const G4Step* step)
{
  if (ntuple_ == nullptr)
    return;

  // Collect energy and track length step by step

  // get volume of the current step
  auto* volume{step->GetPreStepPoint()->GetTouchableHandle()->GetVolume()};

  // find the sensor index
  G4double calibration{0.};

  auto vItr{sensors_.find(volume)};
  if (vItr == sensors_.end()) {
    vItr = absorbers_.find(volume);
    if (vItr == absorbers_.end())
      return;
    calibration = absorberCalibration;
  }
  else
    calibration = sensorCalibration;

  ntuple_->recoEnergy[vItr->second] += step->GetTotalEnergyDeposit() / MeV * calibration;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
