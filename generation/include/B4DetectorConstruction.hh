#ifndef B4DetectorConstruction_h
#define B4DetectorConstruction_h 1

#include "globals.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VisAttributes.hh"

#include <vector>

struct SensorDescription {
  SensorDescription(unsigned _id, G4VPhysicalVolume* _sensor, G4VPhysicalVolume* _absorber, G4double _x, G4double _y, G4double _z) :
    id{_id},
    sensor{_sensor},
    absorber{_absorber},
    x{_x},
    y{_y},
    z{_z}
  {}

  unsigned id{};
  G4VPhysicalVolume* sensor{};
  G4VPhysicalVolume* absorber{};
  G4double x{};
  G4double y{};
  G4double z{};
};

typedef std::vector<SensorDescription> SensorDescriptions;

class B4DetectorConstruction : public G4VUserDetectorConstruction {
public:
  B4DetectorConstruction();
  ~B4DetectorConstruction();

public:
  G4VPhysicalVolume* Construct() override;
  void ConstructSDandField() override;

  SensorDescriptions const& getSensors() const { return sensors_; }
     
private:
  SensorDescriptions sensors_{};
  bool checkOverlaps_{true};

  G4VisAttributes* layerVisualization[2]{};
  G4VisAttributes* sensorVisualization{};
  G4VisAttributes* absorberVisualization{};
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif

