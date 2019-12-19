#ifndef B4aEventAction_h
#define B4aEventAction_h 1

#include "globals.hh"

#include "G4UserEventAction.hh"
#include "G4Step.hh"

#include "NtupleEntry.hh"

class B4aEventAction : public G4UserEventAction {
public:
  B4aEventAction(NtupleEntry&);
  ~B4aEventAction();

  void BeginOfEventAction(const G4Event*) override;
  void EndOfEventAction(const G4Event*) override;

private:
  NtupleEntry& ntuple_;
};

#endif
