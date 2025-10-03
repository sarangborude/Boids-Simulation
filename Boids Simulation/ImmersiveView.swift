//
//  ImmersiveView.swift
//  Boids Simulation
//
//  Created by Sarang Borude on 10/3/25.
//

import SwiftUI
import RealityKit
import RealityKitContent

struct ImmersiveView: View {
    
    let configuration = SpatialTrackingSession.Configuration(tracking: [.hand])
    let session = SpatialTrackingSession()
    
    private var leftHandAnchor = AnchorEntity(.hand(.left, location: .indexFingerTip))
    private var rightHandAnchor = AnchorEntity(.hand(.right, location: .indexFingerTip))

    var body: some View {
        RealityView { content in
            let markerRadius: Float = 0.03
            var redMaterial = UnlitMaterial(color: .red)
            redMaterial.blending = .transparent(opacity: .init(floatLiteral: 0))
            
            let leftMarker = ModelEntity(mesh: .generateSphere(radius: markerRadius), materials: [redMaterial])
            leftMarker.name = "LeftHandMarker"
            leftMarker.components.set(RepellerComponent())
            leftHandAnchor.addChild(leftMarker)
            
            let rightMarker = ModelEntity(mesh: .generateSphere(radius: markerRadius), materials: [redMaterial])
            rightMarker.name = "RightHandMarker"
            rightMarker.components.set(RepellerComponent())
            rightHandAnchor.addChild(rightMarker)
            
            content.add(leftHandAnchor)
            content.add(rightHandAnchor)
            
            let mesh = MeshResource.generateBox(size: [0.2, 0.2, 0.6])
            
//            var boidMaterial = PhysicallyBasedMaterial()
//            boidMaterial.baseColor = PhysicallyBasedMaterial.BaseColor(tint: .init(white: 1.0, alpha: 1))
//            boidMaterial.metallic.scale = 1
//            boidMaterial.roughness.scale = 0.25
            
            let boidMaterial = UnlitMaterial(color: .blue)
            
            let count = 300
            
            guard let instances = try? LowLevelInstanceData(instanceCount: count) else {
                fatalError("Failed to create instance data")
            }
            
            // Fill transforms for each instance (forward-biased sampling)
            instances.withMutableTransforms { transforms in
                let minForwardOffset: Float = 1.5
                let maxForwardOffset: Float = 2.0
                for i in 0..<count {
                    let x = Float.random(in: -0.8...0.8)
                    let y = Float.random(in: -0.6...0.6)
                    let z = -Float.random(in: minForwardOffset...maxForwardOffset)
                    let position: SIMD3<Float> = [x, y, z]
                    let scale: Float = .random(in: 0.05...0.1)
                    let angle: Float = .random(in: 0..<(2 * .pi))
                    let transform = Transform(
                        scale: .init(repeating: scale),
                        rotation: .init(angle: angle, axis: [0, 1, 0]),
                        translation: position
                    )
                    transforms[i] = transform.matrix
                }
            }
            
            guard let meshInstancesComponent = try? MeshInstancesComponent(mesh: mesh, instances: instances) else {
                fatalError("Failed to create MeshInstancesComponent")
            }
            
            
            let model = ModelEntity(mesh: mesh, materials: [boidMaterial])
            model.position = [0, 1.2, -1]
            model.components.set(meshInstancesComponent)
            content.add(model)
        }
    }
}

#Preview(immersionStyle: .mixed) {
    ImmersiveView()
        .environment(AppModel())
}
