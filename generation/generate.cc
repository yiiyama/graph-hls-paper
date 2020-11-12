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

#include "TROOT.h"

#include "include/B4DetectorConstruction.hh"
#include "include/B4aActionInitialization.hh"
#include "include/B4PrimaryGeneratorAction.hh"

#include <memory>
#include <set>
#include <stdexcept>

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

namespace {
  void PrintUsage() {
    G4cerr << "Usage: " << G4endl;
    G4cerr << "generate [-s SEED] [-f FILE] [-j THREADS]" << G4endl;
    G4cerr << " [-e EMIN EMAX] [-x XMAX YMAX] [-p PARTICLES]" << G4endl;
    G4cerr << " [-n EVENTS] [-guh] [macro]" << G4endl;
    G4cerr << G4endl;
    G4cerr << "The options are:" << G4endl;
    G4cerr << "  -s SEED      Set the random number seed to SEED (integer)." << G4endl;
    G4cerr << "  -f FILE      Write the output data file to FILE (path)." << G4endl;
    G4cerr << "  -j THREADS   Run in THREADS (integer) threads." << G4endl;
    G4cerr << "  -e EMIN EMAX Set the minimum and maximum of the generated particle." << G4endl;
    G4cerr << "  -x XMAX YMAX Set the size of the X-Y region where the particles originate." << G4endl;
    G4cerr << "  -p PARTICLES Set particle types to generate (Use comma- or space-separated" << G4endl;
    G4cerr << "               G4 particle names). End the list with a hyphen." << G4endl;
    G4cerr << "  -n EVENTS    Generate EVENTS (integer) events." << G4endl;
    G4cerr << "  -g           Save the geometry to the output file and exit." << G4endl;
    G4cerr << "  -u           Add pileup to the events." << G4endl;
    G4cerr << "  -h           Show this message." << G4endl;
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

int
main(int argc, char** argv)
{
  gROOT->SetBatch(false);

  G4String macro{};
  G4String outfile{"out"};
  G4long seed{0};
  G4String nthreads{"1"};
  bool saveGeometry{false};
  G4double minEnergy{10.};
  G4double maxEnergy{100.};
  G4double maxX{10.};
  G4double maxY{10.};
  std::set<B4PrimaryGeneratorAction::particles> particleTypes{{B4PrimaryGeneratorAction::elec, B4PrimaryGeneratorAction::pioncharged}};
  bool addPileup{false};
  G4int nevents{-1};

  G4int i{1};
  while (i < argc) {
    try {
      G4String opt{argv[i]};
      if (opt == "-f") {
        outfile = argv[i+1];
        i += 2;
      }
      else if (opt == "-s") {
        seed = G4UIcommand::ConvertToInt(argv[i+1]);
        i += 2;
      }
      else if (opt == "-g") {
        saveGeometry = true;
        i += 1;
      }
      else if (opt == "-j") {
        nthreads = argv[i+1];
        i += 2;
      }
      else if (opt == "-e") {
        minEnergy = std::stof(argv[i+1]);
        maxEnergy = std::stof(argv[i+2]);
        i += 3;
      }
      else if (opt == "-x") {
        maxX = std::stof(argv[i+1]);
        maxY = std::stof(argv[i+2]);
        i += 3;
      }
      else if (opt == "-p") {
        particleTypes.clear();
        G4String const* nbegin{B4PrimaryGeneratorAction::particleNames};
        G4String const* nend{B4PrimaryGeneratorAction::particleNames + B4PrimaryGeneratorAction::particles_size};
      
        std::string particles(argv[i+1]);
        if (particles != "-") {
          size_t pos{0};
          while (true) {
            auto comma{particles.find(",", pos)};

            G4String name(particles.substr(pos, comma));
            auto nitr{std::find(nbegin, nend, name)};
            if (nitr == nend)
              throw std::runtime_error("Invalid particle name " + name);
            std::cout << name << " " << (nitr - nbegin) << std::endl;
            particleTypes.insert(B4PrimaryGeneratorAction::particles(nitr - nbegin));

            if (comma == std::string::npos)
              break;
        
            pos = comma + 1;
          }
        }
        
        i += 2;
      }
      else if (opt == "-u") {
        addPileup = true;
        i += 1;
      }
      else if (opt == "-n") {
        nevents = std::stoi(argv[i+1]);
        i += 2;
      }
      else if (opt == "-h") {
        PrintUsage();
        return 0;
      }
      else {
        macro = opt;
        i += 1;
      }
    }
    catch (std::exception& ex) {
      std::cerr << ex.what() << std::endl;
      PrintUsage();
      return 1;
    }
  }

  std::unique_ptr<G4UIExecutive> ui{};
  if (macro.size() == 0 && nevents == -1)
    ui.reset(new G4UIExecutive(argc, argv));

  auto engine{std::make_unique<CLHEP::RanecuEngine>()};
  G4Random::setTheEngine(engine.get());
  G4Random::setTheSeeds(&seed);

  auto runManager{std::make_unique<RunManager>()};

  auto* dc{new B4DetectorConstruction()};
  dc->setCheckOverlaps(false);
  dc->setGeometryType(B4DetectorConstruction::kMiniCalo);
  runManager->SetUserInitialization(dc);

  runManager->SetUserInitialization(new FTFP_BERT);

  auto* ai{new B4aActionInitialization(particleTypes, dc->getSensors())};
  ai->setSaveGeometry(saveGeometry);
  ai->setFilename(outfile);
  ai->setEnergy(minEnergy, maxEnergy);
  ai->setPositionWindow(maxX, maxY);
  ai->setAddPileup(addPileup);
  runManager->SetUserInitialization(ai);

  runManager->SetPrintProgress(1);

  G4UImanager::GetUIpointer()->ApplyCommand("/run/numberOfThreads " + nthreads);

  if (saveGeometry) {
    G4UImanager::GetUIpointer()->ApplyCommand("/run/initialize");
    G4UImanager::GetUIpointer()->ApplyCommand("/run/beamOn 1");
    return 0;
  }

  std::unique_ptr<G4VisExecutive> visManager;
  if (ui) {
    visManager.reset(new G4VisExecutive());
    visManager->Initialize();

    G4UImanager::GetUIpointer()->ApplyCommand("/control/execute macros/init_vis.mac");
    if (ui->IsGUI())
      G4UImanager::GetUIpointer()->ApplyCommand("/control/execute macros/gui.mac");

    ui->SessionStart();
  }
  else if (nevents == -1)
    G4UImanager::GetUIpointer()->ApplyCommand("/control/execute " + macro);
  else {
    G4UImanager::GetUIpointer()->ApplyCommand("/run/initialize");
    G4UImanager::GetUIpointer()->ApplyCommand("/run/printProgress 1");
    G4UImanager::GetUIpointer()->ApplyCommand("/run/verbose 0");
    G4UImanager::GetUIpointer()->ApplyCommand("/run/beamOn " + std::to_string(nevents));
  }
    
  return 0;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo.....
