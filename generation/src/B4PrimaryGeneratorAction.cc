#include "../include/B4PrimaryGeneratorAction.hh"

#include "G4Event.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"
#include "CLHEP/Random/RandFlat.h"

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
  gun_(1), // generate one particle at a time
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
  particles pid{particlesToGenerate_[CLHEP::RandFlat::shootInt(particlesToGenerate_.size())]};
  auto* pdef{G4ParticleTable::GetParticleTable()->FindParticle(particleNamesG4[pid])};

  G4double energy{CLHEP::RandFlat::shoot(minEnergy_, maxEnergy_)};
  G4double xpos{CLHEP::RandFlat::shoot(-maxX_, maxX_)};
  G4double ypos{CLHEP::RandFlat::shoot(-maxY_, maxY_)};

  G4ThreeVector position(xpos, ypos, zpos_);
  position *= cm;

  gun_.SetParticleDefinition(pdef);
  gun_.SetParticleEnergy(energy * GeV);
  gun_.SetParticleMomentumDirection(G4ThreeVector(0., 0., 1.));
  gun_.SetParticlePosition(position);
  gun_.GeneratePrimaryVertex(event);

  if (ntuple_ != nullptr) {
    ntuple_->clear();
    
    ntuple_->genPid = pid;
    ntuple_->genEnergy = energy;
    ntuple_->genX = xpos;
    ntuple_->genY = ypos;
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

