#include "../include/B4DetectorConstruction.hh"

#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4Colour.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

B4DetectorConstruction::B4DetectorConstruction() :
  G4VUserDetectorConstruction()
{
}

B4DetectorConstruction::~B4DetectorConstruction()
{
  delete layerVisualization[0];
  delete layerVisualization[1];
  delete sensorVisualization;
  delete absorberVisualization;
}

G4VPhysicalVolume*
B4DetectorConstruction::Construct()
{
  // Geometry parameters
  constexpr G4double caloSizeXY{30. * cm};
  constexpr G4double worldSizeXY{1.2 * caloSizeXY};
  constexpr G4double worldSizeZ{10. * m};

  constexpr G4double caloFaceZ{3.5 * m};

  constexpr unsigned nLayers[2] = {25, 25};
  constexpr G4double layerDZAbs[2] = {0.57 * cm, 4.17 * cm};
  constexpr G4double layerDZSi[2] = {0.03 * cm, 0.03 * cm};

  constexpr G4double sensorEdge{2.5 * cm};
  constexpr G4double sensorHalfDiag{sensorEdge / std::sqrt(2.)};

  constexpr G4double epsilon{10. * um};

  G4ThreeVector origin(0., 0., 0.);
  G4RotationMatrix sensorRotation(G4ThreeVector(0., 0., 1.), CLHEP::pi / 4.);

  // Material
  auto* nist{G4NistManager::Instance()};

  auto* air{nist->FindOrBuildMaterial("G4_AIR")};
  auto* copper{nist->FindOrBuildMaterial("G4_Cu")};
  auto* stainless{nist->FindOrBuildMaterial("G4_STAINLESS-STEEL")};
  auto* silicon{nist->FindOrBuildMaterial("G4_Si")};

  G4Material* absMaterial[2] = {copper, stainless};

  // Clear list of sensors
  sensors_.clear();

  std::vector<G4LogicalVolume*> layers[2];

  // Now the actual geometry
  //
  // World
  //
  auto* worldSolid{new G4Box("World", worldSizeXY * 0.5, worldSizeXY * 0.5, worldSizeZ * 0.5)};
  auto* worldLV{new G4LogicalVolume(worldSolid, air, "World")};
  auto* worldPV{new G4PVPlacement(
    nullptr, // no rotation
    origin,
    worldLV, // logical volume
    "World", // name
    nullptr, // its mother volume
    false, // pMany
    0, // copy number
    checkOverlaps_ // checking overlaps
  )};

  G4String caloName[2] = {"EE", "HE"};
  G4double caloZ{caloFaceZ}; // front z of the calorimeter; to be incremented

  for (unsigned iDet{0}; iDet != 2; ++iDet) {
    //
    // Calorimeter
    //
    auto layerDZ{layerDZAbs[iDet] + layerDZSi[iDet]};
    auto caloSizeZ{nLayers[iDet] * layerDZ + nLayers[iDet] * epsilon * 2.};

    auto* caloSolid{new G4Box(caloName[iDet], caloSizeXY * 0.5, caloSizeXY * 0.5, caloSizeZ * 0.5)};
    auto* caloLV{new G4LogicalVolume(caloSolid, air, caloName[iDet])};
    new G4PVPlacement(
      nullptr, // no rotation
      G4ThreeVector(0., 0., caloZ + caloSizeZ * 0.5),
      caloLV, // logical volume
      caloName[iDet], // name
      worldLV, // its mother volume
      false, // pMany
      0, // copy number
      checkOverlaps_ // checking overlaps
    );

    G4double layerZ{-caloSizeZ * 0.5 + layerDZ * 0.5};
   
    for (unsigned iL{0}; iL != nLayers[iDet]; ++iL) {
      //
      // Layer
      //
      G4String layerName{"Layer_" + caloName[iDet] + "_" + std::to_string(iL)};
      auto* layerSolid{new G4Box(layerName, caloSizeXY * 0.5, caloSizeXY * 0.5, layerDZ * 0.5)};
      auto* layerLV{new G4LogicalVolume(layerSolid, air, layerName)};
      new G4PVPlacement(
        nullptr, // no rotation
        G4ThreeVector(0., 0., layerZ),
        layerLV, // logical volume
        layerName, // name
        caloLV, // its mother volume
        false, // pMany
        0, // copy number
        checkOverlaps_ // checking overlaps
      );

      layers[iDet].push_back(layerLV);

      //
      // Absorber and sensor tiles
      //  
      G4double ypos{caloSizeXY * 0.5 - sensorHalfDiag};
      bool oddrow{true};
      while (ypos >= -caloSizeXY * 0.5 + sensorHalfDiag) {
        G4double xpos;
        if (oddrow)
          xpos = -caloSizeXY * 0.5 + sensorHalfDiag;
        else
          xpos = -caloSizeXY * 0.5 + sensorHalfDiag * 2.;
        
        while (xpos <= caloSizeXY * 0.5 - sensorHalfDiag) {
          auto id{sensors_.size()};

          G4ThreeVector absDisplacement(xpos, ypos, -layerDZ * 0.5 + layerDZAbs[iDet] * 0.5);
          G4ThreeVector sensorDisplacement(xpos, ypos, -layerDZ * 0.5 + layerDZAbs[iDet] + layerDZSi[iDet] * 0.5);
          G4Transform3D absPositioning(sensorRotation, absDisplacement);
          G4Transform3D sensorPositioning(sensorRotation, sensorDisplacement);
  
          // absorber
          G4String absName{"Absorber_" + std::to_string(id)};
          auto* absSolid{new G4Box(absName, sensorEdge * 0.5, sensorEdge * 0.5, layerDZAbs[iDet] * 0.5)};
          auto* absLV{new G4LogicalVolume(absSolid, absMaterial[iDet], absName)};
          auto* absPV{new G4PVPlacement(
            absPositioning,
            absLV, // logical volume
            absName, // name
            layerLV, // its mother volume
            false, // pMany
            0, // copy number
            checkOverlaps_ // checking overlaps
          )};
  
          // sensor
          G4String sensorName{"Sensor_" + std::to_string(id)};
          auto* sensorSolid{new G4Box(sensorName, sensorEdge * 0.5, sensorEdge * 0.5, layerDZSi[iDet] * 0.5)};
          auto* sensorLV{new G4LogicalVolume(sensorSolid, silicon, sensorName)};
          auto* sensorPV{new G4PVPlacement(
            sensorPositioning,
            sensorLV, // logical volume
            sensorName, // name
            layerLV, // its mother volume
            false, // pMany
            0, // copy number
            checkOverlaps_ // checking overlaps
          )};
  
          sensors_.emplace_back(id, sensorPV, absPV, xpos, ypos, caloZ + layerZ + layerDZAbs[iDet] + layerDZSi[iDet] * 0.5);
  
          xpos += sensorHalfDiag + epsilon;
        }
        oddrow = !oddrow;
        ypos -= sensorHalfDiag + epsilon;
      }

      layerZ += layerDZ + epsilon;
    }

    caloZ += caloSizeZ;
  }

  //
  // Visualization attributes
  //
  worldLV->SetVisAttributes(G4VisAttributes::GetInvisible());

  layerVisualization[0] = new G4VisAttributes(G4Colour(0.5, 0.5, 0.));
  layerVisualization[1] = new G4VisAttributes(G4Colour(0., 0.5, 0.5));

  for (unsigned iDet{0}; iDet != 2; ++iDet) {
    layerVisualization[iDet]->SetVisibility(true);
    for (auto* layerLV : layers[iDet])
      layerLV->SetVisAttributes(layerVisualization[iDet]);
  }

  sensorVisualization = new G4VisAttributes(G4Colour(1., 0., 0.));
  sensorVisualization->SetVisibility(false);
  for (auto& sensor : sensors_)
    sensor.sensor->GetLogicalVolume()->SetVisAttributes(sensorVisualization);

  absorberVisualization = new G4VisAttributes(G4Colour(0., 1., 0.));
  absorberVisualization->SetVisibility(false);
  for (auto& sensor : sensors_)
    sensor.absorber->GetLogicalVolume()->SetVisAttributes(absorberVisualization);

  //
  // Always return the physical World
  //
  return worldPV;
}

void
B4DetectorConstruction::ConstructSDandField()
{
  // auto field{std::make_unique<G4GlobalMagFieldMessenger>(G4ThreeVector(0., 0., 3.8 * tesla))};
  // field->SetVerboseLevel(1);
}
