//
//  AppModel.swift
//  Boids Simulation
//
//  Created by Sarang Borude on 10/3/25.
//

import SwiftUI
import RealityKit

/// Maintains app-wide state
@MainActor
@Observable
class AppModel {
    let immersiveSpaceID = "ImmersiveSpace"
    enum ImmersiveSpaceState {
        case closed
        case inTransition
        case open
    }
    var immersiveSpaceState = ImmersiveSpaceState.closed
    
    init() {
        InstanceAnimationSystem.registerSystem()
        RepellerComponent.registerComponent()
    }
}
