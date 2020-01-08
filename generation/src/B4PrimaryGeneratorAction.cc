#include "../include/B4PrimaryGeneratorAction.hh"

#include "G4Event.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandPoisson.h"
#include "CLHEP/Random/RandLandau.h"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4String const
B4PrimaryGeneratorAction::particleNames[particles_size] = {
  "electron",
  "muon",
  "pioncharged",
  "pionneutral",
  "klong",
  "kshort",
  "gamma"
};

G4String const
B4PrimaryGeneratorAction::particleNamesG4[particles_size] = {
  "e-",
  "mu-",
  "pi+",
  "pi0",
  "kaon0L",
  "kaon0S",
  "gamma"
};

B4PrimaryGeneratorAction::B4PrimaryGeneratorAction(std::set<particles> const& particleTypes, NtupleEntry* ntuple) :
  G4VUserPrimaryGeneratorAction(),
  //gun_(1), // generate one particle at a time
  gun_(),
  particlesToGenerate_(particleTypes.begin(), particleTypes.end()),
  ntuple_{ntuple}
{
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

B4PrimaryGeneratorAction::~B4PrimaryGeneratorAction()
{
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void
B4PrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
  // This function is called at the begining of event
  if (!particlesToGenerate_.empty()) {
    particles pid{particlesToGenerate_[CLHEP::RandFlat::shootInt(particlesToGenerate_.size())]};
    auto* pdef{G4ParticleTable::GetParticleTable()->FindParticle(particleNamesG4[pid])};

    G4double energy{CLHEP::RandFlat::shoot(minEnergy_, maxEnergy_)};
    G4double xpos{CLHEP::RandFlat::shoot(-maxX_, maxX_)};
    G4double ypos{CLHEP::RandFlat::shoot(-maxY_, maxY_)};

    G4ThreeVector position(xpos, ypos, zpos_);
    position *= cm;

    gun_.SetParticleDefinition(pdef);
    gun_.SetParticleMomentum(energy * GeV);
    gun_.SetParticleMomentumDirection(G4ThreeVector(0., 0., 1.));
    gun_.SetParticlePosition(position);
    gun_.GeneratePrimaryVertex(event);

    if (ntuple_ != nullptr) {
      ntuple_->genPid = pid;
      ntuple_->genEnergy = energy;
      ntuple_->genX = xpos;
      ntuple_->genY = ypos;
    }
  }

  if (addPileup_) {
    // assuming 30cm x 30cm at 80cm radius (eta ~ 2) -> 0.16 eta x phi
    // at PU 200 we expect 250 PU particles per eta x phi unit at eta ~ 2
    // 250 * 0.16 = 40
    auto npu{CLHEP::RandPoisson::shoot(40.)};
    auto* photon{G4ParticleTable::GetParticleTable()->FindParticle("gamma")};
    auto* pion{G4ParticleTable::GetParticleTable()->FindParticle("pi+")};

    int ipu{0};
    while (ipu != npu) {
      // empirical energy distribution I saw
      G4double energy{(CLHEP::RandLandau::shoot() + 0.6) * 0.5};
      if (energy > 20.)
        continue;

      ipu += 1;

      auto ptype{CLHEP::RandFlat::shootInt(3)};
      // PU seems rougly 2:1 hadrons:photons
      if (ptype == 0)
        gun_.SetParticleDefinition(photon);
      else
        gun_.SetParticleDefinition(pion);
      
      G4double xpos{CLHEP::RandFlat::shoot(-maxX_, maxX_)};
      G4double ypos{CLHEP::RandFlat::shoot(-maxY_, maxY_)};

      G4ThreeVector position(xpos, ypos, zpos_);
      position *= cm;

      gun_.SetParticleMomentum(energy * GeV);
      gun_.SetParticleMomentumDirection(G4ThreeVector(0., 0., 1.));
      gun_.SetParticlePosition(position);
      gun_.GeneratePrimaryVertex(event);
    }
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

