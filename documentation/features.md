# The game's core features
This file describes the core features that align with the core philopshy to define the soul of the game.

## The world
The world of the game should have much variety, be aesthetically interesting and provide resources and other rewards in a balanced and fun way.

### Elements
Instead of the traditional way of there being different kinds of blocks that all do different things this game will instead have different elements. The elements
have different properties and react with other elements in different ways. Elements can be combined in different ways with other elements to create blocks.

### The properties of an element
There are different core types of properties that elements can have. There are the here-after referred to as core properties, all elements have these and these
properties are inherited by blocks as the average of the elements they are made by's properties. So if a block has equal parts of one element that has durability level
1, another one with level 2 and then another with 3 the blocks' durability will be equal to (1+2+3)/3 which equals 2. This is the way these properties are calculated.

These are the core properties:
- Durability, how much damage it can take until it breaks
- Hardness, how hard it is to damage it, acts as a floor for the least amount of damage it can take, if it is attacked with something lower it takes no damage
- Conductivity, how well it can transfer electricity (waaay more about electricity later)
- Thermal Conductivity, how well it transfers heat 
- Density, how heavy it is, will affect how easily it is moved
- Temperature resistance, the range of temperatures that it survives before taking damage (hardness does not protect against this), the temp range is 0 to u8::max 
- Friction, how much it grips to adjacent blocks, if the value is higher than an adjacent blocks density that blocks moves with it (if the pushing force is adequate)
- Light emission, the amount of light something gives out, also a 0 to u8::max range
- Transparency, how much light travels through it, a range of 0% to 100%

Then there are special properties that not all elements have. These properties are more complex and the strength of their effects are scaled with how big percentage
that element is of the entire block. If two elements have the same special property then the strength of the property for the block is an average of the calculated
strength of the property for the elements that have it.

These are some examples of examples of things that might be such properties:
- Explosion at breakage
- Magnetism, attracts some other elements
- Corrosion, wears down blocks around it
- Makes heat into electricity
- Can store electricity

Then there are also those properties that only come when two or more elements are combined. These properties are derived from the right percentages of certain elements
in a block. These properties are from now on called reactions. The requirement for a reaction to be active is merely that all of its associated elements are present in 
the block. But the reaction can only achieve its highest effect if its associated elements are in the right amounts in the block, the preferred percentages of the elements
together always equal 100%.

### All about blocks
Blocks may seem simple but they are more nuanced than they might seem. Since the game is supposed to allow players to explore many different kinds of combination of elements
there has to be a framework for exactly what combining elements into blocks means in a practical way and what different kinds of blocks there should be. For the sake of
simplicity and memory there should be blocks called natural blocks that look different depending on which combination of elements they hold. Instead of holding elements in 
specific amounts or arangement they simply say which elements they hold and assume that they have equal parts of all components. Then there are the sort of blocks that are
made by a player through deliberate combinations of given percentages of different elements. These are called mixtures and they can have the special properties and reactions
that the natural blocks can't have. The third type of block is called a configuration. A configuration block is more complex than the other two kinds of blocks. Like the 
mixture blocks the configuration blocks have precise control over the exact amounts of different elements it contains. Unlike the mixture blocks the configuration blocks also
specify the arangement of the elements in the block. This enables for the arangement of elements with certain properties in specific ways that means that the block can
handle way more complicated behaviour. For example a material with high conductivity can be wrapped with a material with low conductivity to make electricity only travel in 
specified directions through the block. The final kind of block is called a computational block. They are another step of complexity that includes many really small components
that add together to a more complicated block with intricate functionality. These computational blocks can be used as small computational units and the components that make
them up are made in a specific crafter. Instead of being built of an arangement of elements they are made of an arangement of complex components. These components can are
logic gates and the computational blocks are made to have electricity pass through them and their logic gates to activate other things around them. 
SIDENOTES: I am thinking if there should be some built in programming language for logic instead or if this less abstract system is more fun?

This boils down to these different kinds of blocks:
- Natural blocks, contains elements of unspecified amounts and arangement
- Mixture blocks, contains elements, specifies percentage amounts and can have special properties and reaction properties
- Configuration blocks, contains elements, specifies percentage amounts and the arangement of the elements
- Computational blocks, takes in electricity and passes it through logic gates in a chosen arangement and sending out electric signals in one or more directions

## The inventory
The inventory is an important part of any game that involves the handling of many resources. What is unique about this games implementation of the inventory is that it is
intentionally as bare bones as possible. Think of an inventory, you are likely thinking of the classic minecraft grid style inventory. Remove the grid, what is the inventory now?
The answer is that it is merely a list containing all of your items. That is exactly what project watt cubed's inventory is. Without any mod it is inaccessable, therefore there
shall be a default inventory mod that is installed and enabled by default in the game. The size of the inventoy should at the start be 100. This might seem like a lot but 
compared to games like minecraft with stacking it is actually quite little. The inventory size should be very upgradeable as well with time so the player might only have this
inventory size for the first few hours or so of the game.

## crafters
There are also different kinds of crafters for crafting different kinds of things in the game. There is one crafter for every kind of block and some crafters are harder to
obtain than others. Crafting the natural blocks can be done by the player without the access to a crafter. The player should have a method that simply takes in any amount of
unique elements (there cannot be more than one of the same kind since natural blocks do not care about ratios) and then outputs a block that. Just like with the inventory there
is no built in interface for the crafting, just a function. There should be a mod though that makes there be a simple crafting interface but you should be able to disable it as
a player. For crafting the mixture blocks there should be a special crafting block that you can put in the elements you want to make a mixture block of. The process of making a
mixture block is simple, all you do is add in the elements and the amount of them (in % of the entire block) if all of the elements don't add up to 100% (a full block) it didn't
work and the function returns an error instead of outputting a completed block. This too should not have an interface by default and instead be moddable but have a default mod.
There should also be similar crafters for the other kinds of blocks.
