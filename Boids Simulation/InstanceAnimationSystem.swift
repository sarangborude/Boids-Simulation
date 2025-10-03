import RealityKit
import simd

/// A marker component indicating an entity acts as a repeller for boids.
struct RepellerComponent: Component {}


/*
Boids System – Changes & Tuning Guide

Overview
- World Bounds: Replaced soft spherical bounds with a 4 x 4 x 4 axis-aligned box centered at the origin. Boids are softly steered inward near faces and hard-clamped after integration so they never leave the box.
- Parameter Docs: Added inline comments describing what each parameter controls (radii, kinematic limits, behavior weights, and world bounds).
- Overcrowding Limit: Added a crowd-limiting rule to prevent large clumps. When a boid detects more than `maxGroupSize` neighbors, it boosts separation, reduces alignment/cohesion, and adds a repulsion away from the local neighbor centroid. This encourages large groups to split naturally.
- Separated Spawn: On first initialization, instances are re-placed with a minimum spacing inside the box and written back into the instance transforms, so the simulation doesn’t begin with all boids clumped together.
- Related Scene Setup (ImmersiveView.swift): Initial instance placement was changed to sample within a local box around the model origin (instead of a small disc), aligning with the system’s 4x4x4 bounds and reducing initial clustering.

Key Parameters (edit in this file)
- neighborRadius (Float): Neighborhood distance for alignment/cohesion checks.
- separationRadius (Float): Personal space distance; separation is applied more strongly within this radius.
- maxSpeed (Float): Upper bound on boid speed (m/s).
- maxForce (Float): Upper bound on steering acceleration magnitude.
- separationWeight, alignmentWeight, cohesionWeight (Float): Relative influence of each rule.
- boundsWeight (Float): Strength of steering used to keep boids inside the box.
- boxHalfExtent (Float = 2.0): Half-extent for the world bounds on each axis (box is [-2, 2] on X/Y/Z).
- boxMargin (Float): Distance from any face where a gentle inward nudge begins.
- maxGroupSize (Int): Maximum tolerated neighbors; exceeding this triggers overcrowding behavior.
- overcrowdingSeparationBoost (Float): Multiplier on separation when overcrowded.
- overcrowdingRepulsionWeight (Float): Strength of extra push away from the local neighbor centroid when overcrowded.
- initialMinSeparation (Float): Minimum spacing enforced during first-time placement.
- initialPlacementMaxTriesPerBoid (Int): Attempts to find a spaced-out spawn location per boid.

How It Works (high level)
1) Per-frame neighbor scan (O(n^2) naive): For each boid, compute separation, alignment, and cohesion contributions based on neighbors within `neighborRadius` and `separationRadius`.
2) Overcrowding: If `neighborCount > maxGroupSize`, boost separation, scale down alignment/cohesion, and add a repulsive steer away from the local neighbor centroid. This makes dense clusters split apart.
3) Box Bounds: Compute the nearest point inside the [-boxHalfExtent, boxHalfExtent] box. If outside, steer strongly toward the nearest point inside. If inside but within `boxMargin` of a face, apply a gentle inward nudge. After integrating velocity, clamp the position to the box.
4) Integration: Apply acceleration to velocity (clamped by `maxSpeed` and `maxForce`) and update transforms. Orient instances to face their velocity.
5) First-time Initialization: Positions are re-sampled with a minimum spacing and written into instance transforms, preserving original scale/rotation, then used to compute the initial flock center.

Tuning Tips
- Too clumpy: Lower `maxGroupSize`, increase `overcrowdingSeparationBoost`, or slightly reduce `cohesionWeight`.
- Too fragmented: Increase `alignmentWeight`, reduce `overcrowdingSeparationBoost`, or raise `maxGroupSize`.
- Boids stick to walls: Increase `boxMargin` and/or `boundsWeight` slightly so nudging starts earlier and is stronger near faces.
- Start more spread out: Increase `initialMinSeparation` and/or enlarge the spawn half-extent in ImmersiveView.swift.

Notes
- The system assumes the boids operate in a local space centered near the origin with box bounds of ±2m. If you need the box centered elsewhere, add an offset to the bounds checks and the clamping step.
- For performance with large instance counts, consider spatial partitioning (grid or BVH) to replace the naive O(n^2) neighbor loop.
*/

struct InstanceAnimationSystem: System {
    private var query = EntityQuery(where: .has(MeshInstancesComponent.self))

    // Boids simulation state cached per entity
    private struct InstanceState {
        var baseTransforms: [simd_float4x4]
        var velocities: [SIMD3<Float>]
        var flockCenter: SIMD3<Float>
    }

    private var cache: [Entity.ID: InstanceState] = [:]

    // Boids parameters (tweak as desired)
    // Neighborhood radii (in meters)
    // - neighborRadius: how far a boid looks to align and cohere with neighbors.
    // - separationRadius: personal space radius; steer away more strongly within this distance.
    private let neighborRadius: Float = 0.15
    private let separationRadius: Float = 0.05

    // Kinematic limits
    // - maxSpeed: upper bound on boid speed (m/s).
    // - maxForce: upper bound on steering acceleration magnitude.
    private let maxSpeed: Float = 0.40
    private let maxForce: Float = 0.60

    // Behavior weights (relative influence of each rule)
    // - separationWeight: keeps boids from crowding.
    // - alignmentWeight: encourages matching neighbors' heading.
    // - cohesionWeight: pulls boids toward neighbors' center of mass.
    // - boundsWeight: strength of steering used to keep boids inside the box.
    private let separationWeight: Float = 1.6
    private let alignmentWeight: Float = 1.0
    private let cohesionWeight: Float = 0.9
    private let boundsWeight: Float = 1.0

    // World bounds (axis-aligned box)
    // Box is centered at the origin with half-extent of 2.0m on each axis,
    // resulting in a 4 x 4 x 4 meter volume: [-2, 2] on X, Y, and Z.
    // - boxMargin: distance from any face where a gentle inward nudge begins.
    private let boxHalfExtent: Float = 1.0
    private let boxMargin: Float = 0.20

    // Initial placement (ensures boids start separated)
    // - initialMinSeparation: minimum distance between any two boids at spawn time.
    // - initialPlacementMaxTriesPerBoid: attempts to find a valid spot per boid.
    private let initialMinSeparation: Float = 0.10
    private let initialPlacementMaxTriesPerBoid: Int = 60

    // Crowd limiting (prevents large clumps from forming)
    // - maxGroupSize: if a boid sees more than this many neighbors within neighborRadius,
    //   it will try to break away from the dense cluster.
    // - overcrowdingSeparationBoost: multiplier applied to separation when overcrowded.
    // - overcrowdingRepulsionWeight: strength of an extra push away from the local neighbor center.
    private let maxGroupSize: Int = 12
    private let overcrowdingSeparationBoost: Float = 2.0
    private let overcrowdingRepulsionWeight: Float = 1.0

    // External repulsion parameters
    // Hands
    // - handRepulsionRadius: radius around each hand where boids get repelled.
    // - handRepulsionWeight: strength of repulsion force from hands.
    private let handRepulsionRadius: Float = 0.20
    private let handRepulsionWeight: Float = 2.0

    init(scene: RealityKit.Scene) {}

    mutating func update(context: SceneUpdateContext) {
        let dt = max(0.001, Float(context.deltaTime))

        let entities = context.entities(matching: query, updatingSystemWhen: .rendering)

        for entity in entities {
            guard var mic = entity.components[MeshInstancesComponent.self] else { continue }

            // Access the first part (index 0) of the mesh instances
            var part = mic[partIndex: 0]
            guard let instanceData = part?.data else { continue }
            let count = instanceData.instanceCount
            guard count > 0 else { continue }

            // Collect all repeller marker positions relative to this entity (hands, head, etc.)
            let repellers = repellerPositions(relativeTo: entity)
            
            if repellers.isEmpty {
                print("No head repellers found")
            } else {
                // Print the first one’s local position and distance from model origin
                let repeller = repellers[0]
                //print("Repeller local:", repeller, "distance:", length(repeller), "Total Count: ", repellers.count)
            }

            // Initialize cache for this entity if needed
            if cache[entity.id] == nil {
                var bases: [simd_float4x4] = Array(repeating: .init(1), count: count)
                var positions: [SIMD3<Float>] = Array(repeating: .zero, count: count)

                instanceData.withTransforms { src in
                    for i in 0..<count {
                        bases[i] = src[i]
                        let t = Transform(matrix: src[i])
                        positions[i] = t.translation
                    }
                }

                // Reposition instances with minimum initial separation using a forward hemisphere (local -Z)
                let minSpawnRadius: Float = 0.7
                let maxSpawnRadius: Float = max(0.75, min(boxHalfExtent - 0.25, 1.6))
                var placed: [SIMD3<Float>] = []
                placed.reserveCapacity(count)
                for _ in 0..<count {
                    var candidate = SIMD3<Float>.zero
                    var ok = false
                    for _ in 0..<initialPlacementMaxTriesPerBoid {
                        // Sample a random unit direction; ensure it's in the forward hemisphere (z <= 0)
                        var dir = randomUnitVector()
                        if dir.z > 0 { dir.z = -dir.z } // reflect to forward hemisphere
                        let r = Float.random(in: minSpawnRadius...maxSpawnRadius)
                        candidate = dir * r

                        // Ensure candidate lies within the box bounds (it should if r <= boxHalfExtent - margin)
                        if abs(candidate.x) > boxHalfExtent || abs(candidate.y) > boxHalfExtent || abs(candidate.z) > boxHalfExtent {
                            continue
                        }

                        var good = true
                        for p in placed {
                            if distance(p, candidate) < initialMinSeparation { good = false; break }
                        }
                        if good { ok = true; break }
                    }
                    if !ok {
                        // Fallback: place directly forward at min radius with a small lateral jitter
                        let jitter: Float = 0.10
                        candidate = SIMD3<Float>(
                            Float.random(in: -jitter...jitter),
                            Float.random(in: -jitter...jitter),
                            -minSpawnRadius
                        )
                        // Clamp to box just in case
                        candidate = simd_max(-SIMD3<Float>(repeating: boxHalfExtent), simd_min(SIMD3<Float>(repeating: boxHalfExtent), candidate))
                    }
                    placed.append(candidate)
                }

                // Write separated positions back to instance transforms, preserving scale/rotation from base
                instanceData.withMutableTransforms { dst in
                    for i in 0..<count {
                        let baseT = Transform(matrix: bases[i])
                        let t = Transform(scale: baseT.scale, rotation: baseT.rotation, translation: placed[i])
                        dst[i] = t.matrix
                    }
                }

                // Use the separated positions going forward
                positions = placed

                // Compute initial flock center from base positions
                let center = positions.reduce(SIMD3<Float>.zero, +) / Float(max(1, positions.count))

                // Small random initial velocities
                var velocities: [SIMD3<Float>] = []
                velocities.reserveCapacity(count)
                for _ in 0..<count {
                    let rand = randomUnitVector() * Float.random(in: 0.05...0.15)
                    velocities.append(rand)
                }

                cache[entity.id] = InstanceState(baseTransforms: bases, velocities: velocities, flockCenter: center)
            }

            guard var state = cache[entity.id] else { continue }

            // Read current positions from transforms
            var positions: [SIMD3<Float>] = Array(repeating: .zero, count: count)
            var scales: [SIMD3<Float>] = Array(repeating: .one, count: count)
            var previousRotations: [simd_quatf] = Array(repeating: simd_quatf(angle: 0, axis: [0,1,0]), count: count)
            instanceData.withTransforms { src in
                for i in 0..<count {
                    let tr = Transform(matrix: src[i])
                    positions[i] = tr.translation
                    scales[i] = tr.scale
                    previousRotations[i] = tr.rotation
                }
            }

            // Update the cached flock center lazily
            state.flockCenter = positions.reduce(SIMD3<Float>.zero, +) / Float(max(1, positions.count))

            // Boids update: compute acceleration for each instance
            var newVelocities = state.velocities

            // Precompute neighbor lists (naive O(n^2))
            for i in 0..<count {
                let pi = positions[i]
                var separation = SIMD3<Float>.zero
                var alignment = SIMD3<Float>.zero
                var cohesionAccum = SIMD3<Float>.zero

                var neighborCount: Int = 0
                var separationCount: Int = 0

                for j in 0..<count where j != i {
                    let pj = positions[j]
                    let toNeighbor = pj - pi
                    let dist = length(toNeighbor)

                    if dist < neighborRadius {
                        neighborCount += 1
                        alignment += state.velocities[j]
                        cohesionAccum += pj
                    }

                    if dist < separationRadius && dist > 0.0001 {
                        separationCount += 1
                        // Strongly steer away, inversely proportional to distance
                        separation -= normalize(toNeighbor) / max(dist, 0.0001)
                    }
                }

                var accel = SIMD3<Float>.zero

                let overcrowded = neighborCount > maxGroupSize

                if separationCount > 0 {
                    let sepSteer = limitMagnitude(separation / Float(separationCount), maxForce)
                    let sepW: Float = overcrowded ? (separationWeight * overcrowdingSeparationBoost) : separationWeight
                    accel += sepSteer * sepW
                }

                if neighborCount > 0 {
                    // Alignment: steer toward average heading
                    let avgVel = alignment / Float(neighborCount)
                    let alignSteer = steer(current: state.velocities[i], towards: avgVel, maxSpeed: maxSpeed, maxForce: maxForce)
                    let alignScale: Float = overcrowded ? 0.4 : 1.0
                    accel += alignSteer * (alignmentWeight * alignScale)

                    // Cohesion: steer toward average position of neighbors
                    let center = cohesionAccum / Float(neighborCount)
                    let desired = center - pi
                    let cohSteer = steer(current: state.velocities[i], towards: desired, maxSpeed: maxSpeed, maxForce: maxForce)
                    let cohScale: Float = overcrowded ? 0.3 : 1.0
                    accel += cohSteer * (cohesionWeight * cohScale)

                    if overcrowded {
                        // Push away from the local neighbor centroid when too many are nearby
                        let center = cohesionAccum / Float(neighborCount)
                        let away = pi - center
                        if length_squared(away) > 1e-8 {
                            let desired = normalize(away) * maxSpeed
                            var crowdSteer = limitMagnitude(desired - state.velocities[i], maxForce)
                            // Scale by how much the local group exceeds the cap (softly clamped)
                            let densityFactor = min(2.0, Float(neighborCount - maxGroupSize) / Float(max(1, maxGroupSize)))
                            crowdSteer *= (overcrowdingRepulsionWeight * max(0.3, densityFactor))
                            accel += crowdSteer
                        }
                    }
                }

                // Axis-aligned box bounds: steer back inside a 4x4x4 box centered at origin
                let minB = SIMD3<Float>(-boxHalfExtent, -boxHalfExtent, -boxHalfExtent)
                let maxB = SIMD3<Float>( boxHalfExtent,  boxHalfExtent,  boxHalfExtent)

                // Compute the nearest point inside the box to current position
                let clampedPos = simd_max(minB, simd_min(maxB, pi))
                let toInside = clampedPos - pi

                if length_squared(toInside) > 1e-8 {
                    // Outside the box: steer strongly toward the nearest point inside
                    let desired = normalize(toInside) * maxSpeed
                    let boundSteer = limitMagnitude(desired - state.velocities[i], maxForce)
                    accel += boundSteer * boundsWeight
                } else {
                    // Inside the box: gently nudge inward when within a margin of any face
                    var nudge = SIMD3<Float>(repeating: 0)

                    let dxMin = pi.x - minB.x
                    let dxMax = maxB.x - pi.x
                    if dxMin < boxMargin { nudge.x += (boxMargin - dxMin) }
                    if dxMax < boxMargin { nudge.x -= (boxMargin - dxMax) }

                    let dyMin = pi.y - minB.y
                    let dyMax = maxB.y - pi.y
                    if dyMin < boxMargin { nudge.y += (boxMargin - dyMin) }
                    if dyMax < boxMargin { nudge.y -= (boxMargin - dyMax) }

                    let dzMin = pi.z - minB.z
                    let dzMax = maxB.z - pi.z
                    if dzMin < boxMargin { nudge.z += (boxMargin - dzMin) }
                    if dzMax < boxMargin { nudge.z -= (boxMargin - dzMax) }

                    if length_squared(nudge) > 1e-8 {
                        let desired = normalize(nudge) * maxSpeed * 0.5
                        let boundSteer = limitMagnitude(desired - state.velocities[i], maxForce)
                        accel += boundSteer * (boundsWeight * 0.5)
                    }
                }

                // External repulsion from hand markers
                for rp in repellers {
                    accel += repulsion(from: rp, at: pi, radius: handRepulsionRadius, weight: handRepulsionWeight, currentVel: state.velocities[i])
                }

                // Integrate velocity
                var v = state.velocities[i] + accel * dt
                // Limit speed
                let speed = length(v)
                if speed > maxSpeed { v = (v / speed) * maxSpeed }
                newVelocities[i] = v
            }

            // Integrate positions and write back transforms
            instanceData.withMutableTransforms { dst in
                for i in 0..<count {
                    var pos = positions[i] + newVelocities[i] * dt
                    // Hard clamp to ensure boids remain within the 4x4x4 box
                    let minB = SIMD3<Float>(-boxHalfExtent, -boxHalfExtent, -boxHalfExtent)
                    let maxB = SIMD3<Float>( boxHalfExtent,  boxHalfExtent,  boxHalfExtent)
                    pos = simd_max(minB, simd_min(maxB, pos))

                    // Orient to face velocity if moving; otherwise keep previous rotation
                    let v = newVelocities[i]
                    let rot: simd_quatf
                    if length_squared(v) > 1e-6 {
                        rot = orientation(from: v)
                    } else {
                        rot = previousRotations[i]
                    }

                    // Preserve scale, update translation and rotation
                    let animated = Transform(
                        scale: scales[i],
                        rotation: rot,
                        translation: pos
                    )
                    dst[i] = animated.matrix
                }
            }

            // Write back the mutated instance data into the part and component
            part?.data = instanceData
            mic[partIndex: 0] = part
            entity.components.set(mic)

            // Persist new velocities
            state.velocities = newVelocities
            cache[entity.id] = state
        }
    }
}

// MARK: - Boids helpers

private func limitMagnitude(_ v: SIMD3<Float>, _ maxMag: Float) -> SIMD3<Float> {
    let m2 = length_squared(v)
    let max2 = maxMag * maxMag
    if m2 > max2 && m2 > 0 { return normalize(v) * maxMag }
    return v
}

private func steer(current: SIMD3<Float>, towards desired: SIMD3<Float>, maxSpeed: Float, maxForce: Float) -> SIMD3<Float> {
    if length_squared(desired) < 1e-8 { return .zero }
    let desiredVel = normalize(desired) * maxSpeed
    let steer = desiredVel - current
    return limitMagnitude(steer, maxForce)
}

private func randomUnitVector() -> SIMD3<Float> {
    var v: SIMD3<Float>
    repeat {
        v = SIMD3(Float.random(in: -1...1), Float.random(in: -1...1), Float.random(in: -1...1))
    } while length_squared(v) < 1e-6
    return normalize(v)
}

private func orientation(from direction: SIMD3<Float>) -> simd_quatf {
    // Build a rotation that points the local +Z axis toward `direction`, with Y up.
    // If your mesh faces a different axis, adjust the basis here.
    let f = normalize(direction) // forward
    let up = SIMD3<Float>(0, 1, 0)
    var r = cross(up, f) // right
    let rLen2 = length_squared(r)
    if rLen2 < 1e-6 {
        // Direction is parallel to up; choose any orthogonal right
        r = normalize(cross(SIMD3<Float>(1, 0, 0), f))
    } else {
        r = r / sqrt(rLen2)
    }
    let u = cross(f, r) // corrected up
    let rot = float3x3(columns: (r, u, f))
    return simd_quatf(rot)
}

// MARK: - Repeller marker helpers

private func repellerPositions(relativeTo entity: Entity) -> [SIMD3<Float>] {
    guard let scene = entity.scene else { return [] }
    let query = RealityKit.EntityQuery(where: .has(RepellerComponent.self))
    let markers = Array(scene.performQuery(query))
    var positions: [SIMD3<Float>] = []
    positions.reserveCapacity(markers.count)
    for marker in markers {
        let local = marker.position(relativeTo: entity)
        positions.append(local)
    }
    return positions
}


/// Calculates a repulsion acceleration from a point if within radius.
/// The repulsion falls off linearly with distance and points away from the repulsor point.
/// - Parameters:
///   - point: The repulsor point position.
///   - position: The boid's current position.
///   - radius: The effective radius of repulsion.
///   - weight: The strength multiplier of the repulsion.
///   - currentVel: The boid's current velocity.
/// - Returns: Acceleration vector to add to the boid's steering.
private func repulsion(from point: SIMD3<Float>, at position: SIMD3<Float>, radius: Float, weight: Float, currentVel: SIMD3<Float>) -> SIMD3<Float> {
    let offset = position - point
    let dist = length(offset)
    if dist >= radius || dist < 1e-6 {
        return .zero
    }
    let dir = offset / dist
    // Desired velocity away from the repulsor, scaled by proximity (stronger when closer)
    let strength = (1.0 - dist / radius) * weight
    let desiredVel = dir * strength * radius * 5.0 // scale factor to get a reasonable force magnitude
    let steer = desiredVel - currentVel
    return limitMagnitude(steer, strength * 2.0)
}

