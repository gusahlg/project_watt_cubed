# Refining of Game Features

The element / composition / recipe / mixture model this draft explored is retired.
Matter is a resource lattice under one integer law:
[`documentation/material-model.md`](../../material-model.md).
Workbench applies, holdings, and worldgen regions are described there.

What remains below are still-open gameplay questions that are **not** the
material kernel (signals, heat as a toy, motion). Do not reintroduce authored
element stats, named recipes, or multiple block kinds.

# Thoughts

## Should Electricity be called something else?
Pros of calling it something else is that it makes it more unique and playful and the downside is that it makes is slightly more ambiguous for new people. Minecraft has
redstone which is super iconic so maybe this game should have something as well describing the transferring of an instant signal between blocks.

### @MrOcelotGuy
The con of having another name though would be that it can turn this less intuitive of a feature for the player

## Should temperature be in the game at all?
A survival temperature meter is out of scope. Heat as a later machine/event kind
is possible only if it is an `EventKind` (or a new law parameter), not a
per-element authored stat.

### @MrOcelotGuy
A placeholder of hot/cold for processing, if any, before a more complicated system.

## Density, friction, and motion
Observations already expose hardness, friction, and liquid/flow. Block velocity,
pistons-as-a-block, and collision damage are still open and must not grow a
second material model.

## Unresolved (outside the law)
- Signals / electricity naming
- Block movement and velocity
- Whether a later event kind should encode heat or pressure
