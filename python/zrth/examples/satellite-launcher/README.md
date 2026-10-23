# Probabilistic Satellite Launcher

This is a naive attempt to translate SDCPN of the [probabilistic satellite
launcher][psl] into a system of reactive modules.

```rust
module Satellite[id]
  external mu, planet_radius, altitude, initial_velocity: R,
           launch_event, collision_event: Event
  interface x, y, direction, velocity: R
  private state: { prelaunch, space, debris }, x, y, direction, velocity: R

  atom state, x, y, direction, velocity
      awaits launch_event reads launch_event
      reads mu, planet_radius, altitude, initial_velocity
      
    init
      let distance = planet_radius + altitude
      let angle = Distribution.Uniform(0, Math.PI * 2)
      state' := prelaunch;
      x' := cos(angle) * distance;
      y' := sin(angle) * distance;
      direction' := Uniform(0, Math.PI * 2);
      velocity' := Gaussian(initial_velocity, initial_velocity * 0.1);

    update
      [] state = prelaunch && launch_event? ->
         state' := space;

      [] state = space && collision_event? ->
         state' := debris;

    flow
        [] state = space ->
            // We need a way to define intermediate variables and functions in ODEs.
            // Distance to planet center
            let r = hypot(x, y)
            // Gravitational acceleration vector (points toward origin)
            let ax = (-mu * x) / (r * r * r)
            let ay = (-mu * y) / (r * r * r)
            d(x) := velocity * cos(direction);
            d(y) := velocity * cos(direction);
            d(direction) := (-ax * sin(direction) + ay * cos(direction)) / velocity;
            d(velocity) := ax * cos(direction) + ay * sin(direction);


module LaunchSatellite[id]
  // these are parameters
  interface launch_rate: Real, launch_event: Event
  
  private deadline: Real, clk: Clock
  atom x[id], y[id], direction[id], velocity[id], deadline
    init
      deadline' := Poisson(launch_rate);
      launch_event' := false
    update
      [] clk = deadline ->
        // launch the satellite
        launch_event' := not(launch_event)
    delay
      // the upper bound is a variable?
      [] true -> clk' <= deadline


// TODO: this is still WIP
module Collision[id1, id2]
  // these are parameters
  interface launch_rate: Real, launch_event: Event

  external x[id1], y[id1], x[id2], y[id2]: Real
  
  private deadline: Real, clk: Clock
  atom x, y, direction, velocity, deadline
    init
      deadline' := Poisson(launch_rate);
      launch_event' := false
    update
      [] ... ->
        // there is a collision
        collision_event[id1]!;
        collision_event[id2]!;

    delay
      // the upper bound is a variable?
      [] true -> clk' <= deadline
```

[psl]: https://demo.petrinaut.org/local/151b7bdb-f199-4980-a1df-d33377809421