#ifndef B4aSteppingAction_h
#define B4aSteppingAction_h 1

#include "G4UserSteppingAction.hh"

#include "NtupleEntry.hh"
#include "B4DetectorConstruction.hh"

/// Stepping action class.
///
/// In UserSteppingAction() there are collected the energy deposit and track 
/// lengths of charged particles in Absober and Gap layers and
/// updated in B4aEventAction.

class B4aSteppingAction : public G4UserSteppingAction {
public:
  B4aSteppingAction(NtupleEntry*, SensorDescriptions const&);
  ~B4aSteppingAction();

  void UserSteppingAction(const G4Step*) override;
    
private:
  NtupleEntry* ntuple_{nullptr};
  G4double sensorCalibration{1.};
  G4double absorberCalibration{1.};
  std::map<G4VPhysicalVolume const*, unsigned> sensors_;
  std::map<G4VPhysicalVolume const*, unsigned> absorbers_;
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif
