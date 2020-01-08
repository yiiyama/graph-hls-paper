#ifndef B4aActionInitialization_h
#define B4aActionInitialization_h 1

#include "G4VUserActionInitialization.hh"
#include "G4String.hh"

#include "B4PrimaryGeneratorAction.hh"
#include "B4DetectorConstruction.hh"

#include <set>

class B4DetectorConstruction;

/// Action initialization class.
///

class B4aActionInitialization : public G4VUserActionInitialization {
public:
  B4aActionInitialization(std::set<B4PrimaryGeneratorAction::particles> const&, SensorDescriptions const&);
  ~B4aActionInitialization();

  void BuildForMaster() const override;
  void Build() const override;

  void setSaveGeometry(bool b) { saveGeometry_ = b; }
  void setFilename(G4String fname) { fFileName = fname; }
  void setEnergy(G4double minE, G4double maxE) { minE_ = minE; maxE_ = maxE; }
  void setPositionWindow(G4double maxX, G4double maxY) { maxX_ = maxX; maxY_ = maxY; }
  void setAddPileup(bool b) { addPileup_ = b; }

private:
  std::set<B4PrimaryGeneratorAction::particles> const& fParticleTypes;
  SensorDescriptions const& fSensors;
  bool saveGeometry_{false};
  G4String fFileName{};
  G4double minE_{10.};
  G4double maxE_{100.};
  G4double maxX_{10.};
  G4double maxY_{10.};
  bool addPileup_{false};
};

#endif

    
