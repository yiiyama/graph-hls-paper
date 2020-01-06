#ifndef B4PrimaryGeneratorAction_h
#define B4PrimaryGeneratorAction_h 1

#include "globals.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"

#include "NtupleEntry.hh"

#include <vector>
#include <set>

/// The primary generator action class with particle gun.
///
/// It defines a single particle which hits the calorimeter 
/// perpendicular to the input face. The type of the particle
/// can be changed via the G4 build-in commands of G4ParticleGun class 
/// (see the macros provided with this example).

class B4PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction {
public:
  enum particles {
    elec,
    muon,
    pioncharged,
    pionneutral,
    klong,
    kshort,
    gamma,
    particles_size //leave this one
  };

  B4PrimaryGeneratorAction(std::set<particles> const&, NtupleEntry*);
  ~B4PrimaryGeneratorAction();

  void GeneratePrimaries(G4Event*) override;

  void setZpos(G4double z) { zpos_ = z; }
  void setMaxEnergy(G4double e) { maxEnergy_ = e; }
  void setMinEnergy(G4double e) { minEnergy_ = e; }
  void setMaxX(G4double x) { maxX_ = x; }
  void setMaxY(G4double y) { maxY_ = y; }

  static G4String const particleNames[particles_size];
  static G4String const particleNamesG4[particles_size];

private:
  G4double zpos_{0.};
  G4double maxEnergy_{100.};
  G4double minEnergy_{10.};
  G4double maxX_{10.};
  G4double maxY_{10.};

  G4ParticleGun gun_;
  std::vector<particles> const particlesToGenerate_;
  NtupleEntry* ntuple_{nullptr};
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif
