#ifdef G4MULTITHREADED
#include "G4MTRunManager.hh"
typedef G4MTRunManager RunManager;
#else
#include "G4RunManager.hh"
typedef G4RunManager RunManager;
#endif

#include "G4VisExecutive.hh"
#include "G4UImanager.hh"
#include "G4UIExecutive.hh"
#include "FTFP_BERT.hh"

#include "include/B4DetectorConstruction.hh"
#include "include/B4aActionInitialization.hh"
#include "include/B4PrimaryGeneratorAction.hh"

#include <memory>

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

int
main(int argc, char** argv)
{
  auto ui{std::make_unique<G4UIExecutive>(argc, argv)};

  auto runManager{std::make_unique<RunManager>()};

  auto* dc{new B4DetectorConstruction()};
  runManager->SetUserInitialization(dc);
  runManager->SetUserInitialization(new FTFP_BERT);
  auto* ai{new B4aActionInitialization(std::set<B4PrimaryGeneratorAction::particles>(), dc->getSensors())};
  runManager->SetUserInitialization(ai);

  auto visManager{std::make_unique<G4VisExecutive>()};
  visManager->Initialize();

  G4UImanager::GetUIpointer()->ApplyCommand("/control/execute init_vis.mac");
  if (ui->IsGUI())
    G4UImanager::GetUIpointer()->ApplyCommand("/control/execute gui.mac");

  ui->SessionStart();

  return 0;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....
