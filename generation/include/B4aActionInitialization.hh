#ifndef B4aActionInitialization_h
#define B4aActionInitialization_h 1

#include "G4VUserActionInitialization.hh"
#include "G4String.hh"

#include "B4PrimaryGeneratorAction.hh"
#include "B4DetectorConstruction.hh"
#include "NtupleEntry.hh"

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

  void setFilename(G4String fname) { fFileName = fname; }

private:
  std::set<B4PrimaryGeneratorAction::particles> const& fParticleTypes;
  SensorDescriptions const& fSensors;
  G4String fFileName{};
};

#endif

    
