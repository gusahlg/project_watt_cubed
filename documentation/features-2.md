# Refining of game features
I am rewriting key parts of the features file entirely in a new draft that has more thought through and refined versions of the features described in the original
feature file. After this I will write a new feature file that more extensively goes through the final versions of the core features of the game.

## Core properties
I think the core properties that any element should have are these:
- Durability, how much damage it can take until it breaks
- Hardness, how hard it is to damage it, acts as a floor for the least amount of damage it can take, if it is attacked with something lower it takes no damage
- Conductivity, how well it can transfer electricity (waaay more about electricity later)
- Density, how heavy it is, will affect how easily it is moved
- Friction, how much it grips to adjacent blocks, if the value is higher than an adjacent blocks density that blocks moves with it (if the pushing force is adequate)

## Other types of properties, yes or no?

# Thoughts
## Should Electricity be called something else?
Pros of calling it something else is that it makes it more unique and playful and the downside is that it makes is slightly more ambigious for new people. Minecraft has
redstone which is super iconic so maybe this game should have something as well describing the transfering of an instant signal between blocks affected by their conductivity.

## Should temperature be in the game at all?
Removing it would simplify things a lot since it removes 2 entire core element properties. This would also be good for memory as well which is a bit of a concern for this
kind of game. The real question is, is the game going to provide ways for temperature to be important and fun in any way? Currently we do not even have the concept of smelting
or really any specific material processing at all other than just 'combining' in different forms. I think temperature should be in the game if we add in smelting and multiple
element combining/processing methods, otherwise no.

## Should we have multiple combining methods for elements?
Currently there is only the generic 'combine' essentially meaning that you simply take some elements and combine them into a block, it is very simple which is good but it is
not realistic and it makes things quite simple for the player to make and later automate things. If multiple new methods are added it is essential that they are not
abstract. For example we can not make it so that smelting simply means that you put something into a special block that then makes the end product pop out after some time.
Instead it would have to be something more fundamental like if something is in a certain state it will make something else. it cannot be too specific or complicated either
as that sacrifices a lot of simplicity and makes things less funcamentalist. My final answer to all this is that something like this is too complicated for now and that we
should try without it and see if the game is fun and then maybe add it later on when we have figured out if it is a good fit for the game and what the implementation should
look like.

## Should we have light transmission and transparency?
Would be kind of cool but it kind of seems like a very insignificant feature that does not afect the core gameloop at all so I think we shall remove it for now.

## Should we have density and friction?
Bad thing is that it makes things more complicated and worse performance and such so once again the question is whether or not it is interesting for gameplay or not. Density
making it harder to move certain things makes things more realistic but also maybe annoying in some cases and you might have to calculate exactly when something has enough
friction to move things around it which might be slightly tedious. But I want it to be possible to have moving things and complex mechanics in a way that is more interesting
and nuanced than in Minecraft so this seems better than the minecraft can/cannot be moved and does/doesn't drag blocks with it model in that way.
