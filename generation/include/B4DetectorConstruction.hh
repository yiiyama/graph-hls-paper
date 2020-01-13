#ifndef B4DetectorConstruction_h
#define B4DetectorConstruction_h 1

#include "globals.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4ThreeVector.hh"
#include "G4VisAttributes.hh"

#include <vector>

struct SensorDescription {
  SensorDescription(unsigned _id, G4VPhysicalVolume* _sensor, G4VPhysicalVolume* _absorber, G4ThreeVector const& _pos, G4double _dxy, G4double _dz) :
    id{_id},
    sensor{_sensor},
    absorber{_absorber},
    pos{_pos},
    dxy{_dxy},
    dz{_dz}
  {}

  unsigned id{};
  G4VPhysicalVolume* sensor{};
  G4VPhysicalVolume* absorber{};
  G4ThreeVector pos{};
  G4double dxy{};
  G4double dz{};
};

typedef std::vector<SensorDescription> SensorDescriptions;

class B4DetectorConstruction : public G4VUserDetectorConstruction {
public:
  B4DetectorConstruction();
  ~B4DetectorConstruction();

  G4VPhysicalVolume* Construct() override;
  void ConstructSDandField() override;

  enum GeometryType {
    kHGCALish,
    kMiniCalo,
    nGeometryTypes
  };

  SensorDescriptions const& getSensors() const { return sensors_; }
  void setCheckOverlaps(bool check) { checkOverlaps_ = check; }
  void setGeometryType(unsigned t) { geometryType_ = t; }
    
private:
  SensorDescriptions sensors_{};
  bool checkOverlaps_{true};
  unsigned geometryType_{kHGCALish};

  G4VisAttributes* layerVisualization[2]{};
  G4VisAttributes* sensorVisualization{};
  G4VisAttributes* absorberVisualization{};
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif

