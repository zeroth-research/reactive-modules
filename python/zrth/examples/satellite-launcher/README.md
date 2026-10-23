# Probabilistic Satellite Launcher

This is a naive attempt to translate SDCPN of the [probabilistic satellite
launcher][psl] into a system of reactive modules.

Our findings so far, assuming that we use the encoding below (where we model
each colored token as a reactive module):

 1. We have to introduce reactive modules for the transitions. We have to
 connect token modules to the transition modules via external variables. A
 workaround to this could be: introduce lists/arrays of ids as external
 variables (or events) + quantification or broadcast in reactive modules. This
 would be something similar to [Cubicle][].
 
 1. If we chose to parameterize the number of colored tokens and transitions, we
 cannot express infinitely many firings of a transition (not when a PN
 transition interacts/produces an infinite number of tokens).
 
 1. We could avoid introducing reactive modules for transitions by encoding the
 update/flow logic of each transition directly in the reactive module of each
 token.  In this case, every token module has to be connected with every other
 token module. This would imply explicit unfolding of an existential quantifier
 (checking another satellite). Again, we could introduce arrays and existential
 transitions.
 
 1. An alternative to the above point is to introduce a communication
 circut/switch.
 
```rust
// Global constants (YAML defaults; substituted before module execution)
const planet_radius: R = 50.0;
const satellite_radius: R = 4.0;
const collision_threshold: R = 10.0;
const crash_threshold: R = 5.0;
const mu: R = 400000.0; // gravitational_constant in the YAML
const altitude: R = 40.0;
const launch_rate: R = 0.5;
const initial_velocity: R = 67.0;

module Satellite[id]
  external launch_event, collision_event: Event
  interface x, y, direction, velocity: R
  private state: { prelaunch, space, debris }

  atom state, x, y, direction, velocity
      awaits launch_event, collision_event
      reads state, x, y, direction, velocity,
            launch_event, collision_event
      
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
         direction' := 0;
         velocity' := 0;

    flow
      [] state = space ->
          // We need a way to define intermediate variables and functions in ODEs.
          // Distance to planet center
          let r = hypot(x, y)
          // Gravitational acceleration vector (points toward origin)
          let ax = (-mu * x) / (r * r * r)
          let ay = (-mu * y) / (r * r * r)
          d(x) := velocity * cos(direction);
          d(y) := velocity * sin(direction);
          d(direction) := (-ax * sin(direction) + ay * cos(direction)) / velocity;
          d(velocity) := ax * cos(direction) + ay * sin(direction);


module LaunchSatellite[id]
  interface launch_event: Event
  
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
  interface launch_event: Event

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
      // -> outside Alur/Henzinger "propositional timed" fragment
      [] true -> clk' <= deadline
```

[psl]: https://demo.petrinaut.org/local/151b7bdb-f199-4980-a1df-d33377809421
[Cubicle]: https://usr.lmf.cnrs.fr/cubicle/