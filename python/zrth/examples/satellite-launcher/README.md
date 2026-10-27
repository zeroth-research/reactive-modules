# Probabilistic Satellite Launcher

## First attempt

This is a naive attempt to translate the (S)DCPN of Petrinaut's [probabilistic
satellite launcher example][psl] into a system of reactive modules.

Our findings so far, assuming that we use the encoding below (where we model
each colored token as a reactive module):

 1. We have to introduce reactive modules for the transitions. We have to
 connect token modules to the transition modules via external variables. A
 workaround to this could be: introduce lists/arrays of ids as external
 variables (or events) + quantification or broadcast in reactive modules. This
 would be something similar to [Cubicle][].
 
 1. We can express infinitely many firings with finitely many reactive modules.
 However, with the encoding below, we can only launch as many satellites as we
 have token modules. Each launch adds a token, and collisions and crashes do
 not remove tokens from the net (they move them to Debris). Even if we
 only consider a finite time horizon, there is no fixed bound on the number of
 launches: for a positive launch rate, the count is almost surely finite.
 
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

      [] state = prelaunch || state = debris ->
          d(x) := 0;
          d(y) := 0;
          d(direction) := 0;
          d(velocity) := 0;

    delay
      [] true -> true


module LaunchSatellite[id]
  interface launch_event: Event
  
  private deadline: Real, clk: Clock, fired: Boolean
  atom launch_event, deadline, clk, fired
    init
      clk' := 0;
      deadline' := Exponential(rate = launch_rate);
      launch_event' := false;
      fired' := false;
    update
      [] !fired && clk = deadline ->
        // Launch this satellite exactly once.
        launch_event' := not(launch_event);
        fired' := true;
        // Round-insensitivity requires sleeping to remain possible when no
        // awaited variable changes, even at the deadline. The default update
        // only applies when no guard is enabled, so we state it explicitly:
        [] true -> /* leave all controlled variables unchanged */
    delay
      // (a sampled upper bound is outside the Alur/Henzinger "propositional timed" fragment)
      [] !fired && clk <= deadline -> clk' <= deadline
      // After firing, impose no restriction on elapsed time.
      [] fired -> true


// TODO: this is still WIP
module Collision[id1, id2]
  external x[id1], y[id1], x[id2], y[id2]: Real
  
  init
    deadline' := Poisson(launch_rate);
    launch_event' := false
  update
      [] ... -> // there is a collision
        collision_event[id1]!;
        collision_event[id2]!;

    delay
      // the upper bound is a variable
      [] true -> clk' <= deadline
```

### Limitations

- **Incorrect aggregate launch rate.** All `LaunchSatellite[id]` modules above
  start their independent timers at initialization. With N unused satellite
  modules, the first launch therefore has rate `N * launch_rate`, not
  `launch_rate`; the total rate then decreases as modules fire. The launch
  variants below avoid this problem.
- **Double consumption of a satellite token.** Independent collision modules can
  fire for overlapping pairs, such as `(i, j)` and `(i, k)`, in the same update
  round, effectively removing satellite `i` from Space twice.

## Second attempt

We give launcher variants below are alternatives using the same token representation.

### Token representation

Each `Satellite[i]` represents one potential token. Its discrete mode records the
Petri-net place, with `⊥` meaning that the token does not yet exist.

```rust
module Satellite[i]
  external launch_event: Event

  // Petri net places become a discrete mode component
  private location[i]: {⊥, space, debris}

  // Continuous-valued components
  interface x[i], y[i], direction[i], velocity[i]: Real

  atom location[i], x[i], y[i], direction[i], velocity[i]
    awaits launch_event
    reads location[i], x[i], y[i], direction[i], velocity[i], launch_event

    init
      location[i]' := ⊥  // token does not exist
      x[i]' := 0
      y[i]' := 0
      direction[i]' := 0
      velocity[i]' := 0

    update
      [] location[i] = ⊥ && launch_event? ->
         let distance = planet_radius + altitude
         let angle ~ Uniform(0, Math.PI * 2)
         location[i]' := space
         x[i]' := cos(angle) * distance
         y[i]' := sin(angle) * distance
         direction[i]' ~ Uniform(0, Math.PI * 2)
         velocity[i]' ~ Gaussian(initial_velocity, initial_velocity * 0.1)

      // the default update leaves all controlled variables unchanged

    flow
      [] location[i] = space ->
         let r = hypot(x[i], y[i])
         let ax = (-mu * x[i]) / (r * r * r)
         let ay = (-mu * y[i]) / (r * r * r)
         d(x[i]) := velocity[i] * cos(direction[i])
         d(y[i]) := velocity[i] * sin(direction[i])
         d(direction[i]) :=
           (-ax * sin(direction[i]) + ay * cos(direction[i])) / velocity[i]
         d(velocity[i]) := ax * cos(direction[i]) + ay * sin(direction[i])

      [] location[i] != space ->
         d(x[i]) := 0
         d(y[i]) := 0
         d(direction[i]) := 0
         d(velocity[i]) := 0

    delay
      [] true -> true
```

Alternatively, we could fold the N satellite modules into a single module with
indexed vectors `location[1..N]`, `x[1..N]`, `y[1..N]`, `direction[1..N]`, and
`velocity[1..N]`. Its module body would be the parallel composition of the
per-token atoms. Since these atoms do not await one another, we could also merge
them into one vector-valued atom

### Launch transition

The DCPN's `LaunchSatellite` has no input arcs and a constant rate, so it is
always pre-enabled. Its successive waiting times are independent exponential
samples with rate `launch_rate`.

#### Replicated launcher chain

Give each satellite a one-shot launcher. Only the first launcher
starts armed; each subsequent launcher is activated by its predecessor's firing.

```text
Launch[1] --launch[1]--> Launch[2] --launch[2]--> Launch[3] ...
              |                         |
              v                         v
         Satellite[1]              Satellite[2]
```

```rust
module Launch[i]
  external launch[i-1]: Event
  interface launch[i]: Event
  private state: {idle, armed, done}
  private clk: Clock
  private w: discrete Real

  atom state, clk, w, launch[i]
    awaits launch[i-1]
    reads state, clk, w, launch[i], launch[i-1]

    init
      state' := if i = 1 then armed else idle
      clk' := 0
      w' ~ Exponential(rate = launch_rate)
      launch[i]' := false

    update
      [] state = idle && launch[i-1]? ->
         state' := armed
         clk' := 0

      [] state = armed && clk = w ->
         launch[i]!
         state' := done

    delay
      [] state = idle || state = done -> true
      [] state = armed && clk <= w -> clk' <= w
```

For `i==1`, start immediately in armed.

#### Indexed launcher

A singleton launcher indexes into the instances of `Satellite`. Each firing
increments `selected`, identifying the next unused satellite, and restarts its
timer. The initial value 0 means that no satellite has been selected yet.

```rust
module IndexedLauncher
  interface launch_event: Event
  interface selected: {0, ..., N}
  private clk: Clock
  private w: discrete Real

  atom launch_event, selected, clk, w
    init
      launch_event' := false
      selected' := 0
      clk' := 0
      w' ~ Exponential(rate = launch_rate)

    update
      [] selected < N && clk = w ->
         launch_event!
         selected' := selected + 1
         clk' := 0
         w' ~ Exponential(rate = launch_rate)

      [] true -> skip

    delay
      [] selected < N && clk <= w -> clk' <= w
      [] selected = N -> true
```

Specialize the satellite module as follows:

- Add `external selected: {0, ..., N}` and await `selected` as well as `launch_event`:
- Replace its launch guard with
  `location[i] = ⊥ && launch_event? && selected' = i`.

#### Existential selection

Instead of counting launches, the singleton launcher can choose an unused
satellite. Here `∃ i ∈ 1..N : location[i] = ⊥` tests whether such a slot exists;
`choose i ...` nondeterministically selects exactly one witness. An existential
guard alone must not broadcast a launch to every unused satellite.

```rust
module ExistentialLauncher
  external location[1..N]: {⊥, space, debris}
  interface launch_event: Event
  interface selected: {1, ..., N}
  private clk: Clock
  private w: discrete Real

  atom launch_event, selected, clk, w
    // Read latched locations; do not await the satellites' updated locations.
    reads location[1..N], launch_event, selected, clk, w

    init
      launch_event' := false
      selected' := 1  // immaterial until a launch event occurs
      clk' := 0
      w' ~ Exponential(rate = launch_rate)

    update
      [] clk = w && (∃ i ∈ 1..N : location[i] = ⊥) ->
         choose i ∈ 1..N such that location[i] = ⊥
         selected' := i
         launch_event!
         clk' := 0
         w' ~ Exponential(rate = launch_rate)

      [] clk = w && !(∃ i ∈ 1..N : location[i] = ⊥) ->
         // No available slot: renew the timer without emitting a launch.
         clk' := 0
         w' ~ Exponential(rate = launch_rate)

    delay
      [] clk <= w -> clk' <= w
```

### Limitations

- **Finite capacity.** Both variants implement only the first N launches,
  corresponding to a bounded number of tokens.

[psl]: https://demo.petrinaut.org/local/151b7bdb-f199-4980-a1df-d33377809421
[Cubicle]: https://usr.lmf.cnrs.fr/cubicle/
