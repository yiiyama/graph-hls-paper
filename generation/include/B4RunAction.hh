#ifndef B4RunAction_h
#define B4RunAction_h 1

#include "globals.hh"
#include "G4UserRunAction.hh"
#include "G4String.hh"

#include "B4PrimaryGeneratorAction.hh"
#include "NtupleEntry.hh"

/// Run action class
///
/// It accumulates statistic and computes dispersion of the energy deposit 
/// and track lengths of charged particles with use of analysis tools:
/// H1D histograms are created in BeginOfRunAction() for the following 
/// physics quantities:
/// - Edep in absorber
/// - Edep in gap
/// - Track length in absorber
/// - Track length in gap
/// The same values are also saved in the ntuple.
/// The histograms and ntuple are saved in the output file in a format
/// accoring to a selected technology in B4Analysis.hh.
///
/// In EndOfRunAction(), the accumulated statistic and computed 
/// dispersion is printed.
///

class B4RunAction : public G4UserRunAction {
public:
  B4RunAction(unsigned nSensors);
  ~B4RunAction();

  void setFileName(G4String fname) { fname_ = fname; }
  void setAddDetectorData(bool b) { addDetectorData_ = b; }

  void BeginOfRunAction(const G4Run*) override;
  void EndOfRunAction(const G4Run*) override;

  NtupleEntry& getNtuple() { return ntuple_; }

private:
  NtupleEntry ntuple_;
  G4String fname_{""};
  bool addDetectorData_{false};
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif

