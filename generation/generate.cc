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
#include "Randomize.hh"

//#include "G4RandomTools.hh"

#include "include/B4DetectorConstruction.hh"
#include "include/B4aActionInitialization.hh"
#include "include/B4PrimaryGeneratorAction.hh"

#include <memory>
#include <set>

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

namespace {
  void PrintUsage() {
    G4cerr << " Usage: " << G4endl;
    G4cerr << " exampleB4a -m macro [-s seed] [-f outfile]" << G4endl;
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

int
main(int argc, char** argv)
{
  G4String macro{};
  G4String outfile{"out"};
  G4long seed{0};

  for (G4int i = 1; i < argc; i = i + 2) {
    if (G4String(argv[i]) == "-m")
      macro = argv[i+1];
    else if (G4String(argv[i]) == "-f")
      outfile = argv[i+1];
    else if (G4String(argv[i]) == "-s")
      seed = G4UIcommand::ConvertToInt(argv[i+1]);      
    else {
      PrintUsage();
      return 1;
    }
  }

  std::unique_ptr<G4UIExecutive> ui{};
  if (macro.size() == 0)
    ui.reset(new G4UIExecutive(argc, argv));

  std::set<B4PrimaryGeneratorAction::particles> particleTypes{{B4PrimaryGeneratorAction::elec, B4PrimaryGeneratorAction::pioncharged}};

  auto engine{std::make_unique<CLHEP::RanecuEngine>()};
  G4Random::setTheEngine(engine.get());
  G4Random::setTheSeeds(&seed);

  auto runManager{std::make_unique<RunManager>()};

  auto* dc{new B4DetectorConstruction()};
  runManager->SetUserInitialization(dc);
  runManager->SetUserInitialization(new FTFP_BERT);
  auto* ai{new B4aActionInitialization(particleTypes, dc->getSensors())};
  ai->setFilename(outfile);
  runManager->SetUserInitialization(ai);

  runManager->SetPrintProgress(1);

  G4UImanager::GetUIpointer()->ApplyCommand("/run/numberOfThreads 1");

  std::unique_ptr<G4VisExecutive> visManager;
  if (ui) {
    visManager.reset(new G4VisExecutive());
    visManager->Initialize();

    G4UImanager::GetUIpointer()->ApplyCommand("/control/execute init_vis.mac");
    if (ui->IsGUI())
      G4UImanager::GetUIpointer()->ApplyCommand("/control/execute gui.mac");

    ui->SessionStart();
  }
  else
    G4UImanager::GetUIpointer()->ApplyCommand("/control/execute " + macro);

  return 0;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....
