# The Game's Core Features

This file describes the core features that align with the core philosophy to define the soul of the game.

### The World
The world of the game should have much variety, be aesthetically interesting and provide resources and other rewards in a balanced and fun way.

### Matter
A voxel holds a configuration of lattice points. One law turns neighbouring configurations into new ones; operational properties (solid, liquid, hardness, light, sound) are **observations** of that law, not authored stats. Worldgen places **regions** (families of stable configurations). Breaking yields the configuration; a workbench applies the same events the world uses. Full model: [`documentation/material-model.md`](../../material-model.md).

### The Inventory
The inventory is intentionally a list of held configurations (counts, no grid). Without a mod it is inaccessible; the default inventory mod is enabled as shipped. Starting capacity is 100 and is meant to be upgradeable.

### Crafting
There is no recipe table. The workbench mod applies a physical event (`NewContact`, `Collision`, `Moved`) between two held configurations through `interact`. Discovered procedures are journal knowledge. Machines emit the same events through `ModContext::emit_material_event`. On a server the authority evaluates the apply.
