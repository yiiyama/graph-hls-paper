#include "../include/B4aActionInitialization.hh"
#include "../include/B4PrimaryGeneratorAction.hh"
#include "../include/B4RunAction.hh"
#include "../include/B4aEventAction.hh"
#include "../include/B4aSteppingAction.hh"
#include "../include/B4DetectorConstruction.hh"

#include "G4RunManager.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

B4aActionInitialization::B4aActionInitialization(std::set<B4PrimaryGeneratorAction::particles> const& particleTypes, SensorDescriptions const& sensors) :
  G4VUserActionInitialization(),
  fParticleTypes(particleTypes),
  fSensors{sensors}
{
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

B4aActionInitialization::~B4aActionInitialization()
{
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void B4aActionInitialization::BuildForMaster() const
{
  auto* runAction{new B4RunAction()};
  runAction->setFileName(fFileName);
  if (saveGeometry_)
    runAction->saveGeometry();
  else
    runAction->bookNtuple(fSensors.size());
  SetUserAction(runAction);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void B4aActionInitialization::Build() const
{
  auto* runAction{new B4RunAction()};
  runAction->setFileName(fFileName);
  if (saveGeometry_)
    runAction->saveGeometry();
  else
    runAction->bookNtuple(fSensors.size());
  SetUserAction(runAction);

  auto* genAction{new B4PrimaryGeneratorAction(fParticleTypes, runAction->getNtuple())};
  genAction->setMaxEnergy(maxE_);
  genAction->setMinEnergy(minE_);
  genAction->setMaxX(maxX_);
  genAction->setMaxY(maxY_);
  SetUserAction(genAction);

  auto* eventAction{new B4aEventAction(runAction->getNtuple(), fSensors)};
  SetUserAction(eventAction);

  SetUserAction(new B4aSteppingAction(runAction->getNtuple(), fSensors));

  G4cout << "actions initialised" <<G4endl;
}  

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
