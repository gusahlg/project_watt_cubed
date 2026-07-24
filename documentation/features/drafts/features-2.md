# Refining of Game Features
I am rewriting key parts of the features file entirely in a new draft that has more thought through and refined versions of the features described in the original
feature file. After this I will write a new feature file that more extensively goes through the final versions of the core features of the game.

## Elements
Elements make up the blocks in the world. These are the different kinds of properties that define elements

### Core Properties
I think the core properties that any element should have are these:
- Durability, how much damage it can take until it breaks
- Hardness, how hard it is to damage it, acts as a floor for the least amount of damage it can take, if it is attacked with something lower it takes no damage
- Conductivity, how well it can transfer electricity (waaay more about electricity later)
- Density, how heavy it is, will affect how easily it is moved
- Friction, how much it grips to adjacent blocks, if the value is higher than an adjacent blocks density that blocks moves with it (if the pushing force is adequate)

### Special Properties
Special properties are properties that not every element has.

### Reaction Properties
Reaction properties are properties that do not belong to a single element, rather they come active when a certain combinations of elements are present at once in a block.

## Blocks
Blocks contain elements and inherit their properties in different ways depending on what kind of block it is. There are two different kinds of blocks that behave differently.

### Natural Blocks
Natural blocks are combinations between mutliple elemtns that 

# Thoughts
Random thoughts and debates that need to be resolved to be able to write another draft.

## Should Electricity be called something else?
Pros of calling it something else is that it makes it more unique and playful and the downside is that it makes is slightly more ambiguous for new people. Minecraft has
redstone which is super iconic so maybe this game should have something as well describing the transferring of an instant signal between blocks affected by their conductivity.

### @MrOcelotGuy
The con of having another name though would be that it can turn this less intuitive of a feature for the player

## Should temperature be in the game at all?
Removing it would simplify things a lot since it removes 2 entire core element properties. This would also be good for memory as well which is a bit of a concern for this
kind of game. The real question is, is the game going to provide ways for temperature to be important and fun in any way? Currently we do not even have the concept of smelting
or really any specific material processing at all other than just 'combining' in different forms. I think temperature should be in the game if we add in smelting and multiple
element combining/processing methods, otherwise no.

### @MrOcelotGuy
My own thoughts are that a temperature system for block processing would be wise but having temperatures for survival gameplay where you have to adapt to the cold or something may be
both unfun and take up too much processing power, perhaps have a more complicated temperature system added much later in the game's production  and have a placeholder of just hot/cold
before that point or maybe like hot/lukewarm/cold or something like that before having a more complicated kelvin system

## Should we have multiple combining methods for elements?
Currently there is only the generic 'combine' essentially meaning that you simply take some elements and combine them into a block, it is very simple which is good but it is
not realistic and it makes things quite simple for the player to make and later automate things. If multiple new methods are added it is essential that they are not
abstract. For example we can not make it so that smelting simply means that you put something into a special block that then makes the end product pop out after some time.
Instead it would have to be something more fundamental like if something is in a certain state it will make something else. it cannot be too specific or complicated either
as that sacrifices a lot of simplicity and makes things less fundamentalist. My final answer to all this is that something like this is too complicated for now and that we
should try without it and see if the game is fun and then maybe add it later on when we have figured out if it is a good fit for the game and what the implementation should
look like.

### @MrOcelotGuy
My own thoughts are that a much more elaborate system can be made further on but for now there should be some sort of a crafting bench as a placeholder so that players cant
just combine things together by hand with some sort of super strength.

### @gusahlg
The thing is that crafting is very much so an extremely important feature and it defines how the game is played quite heavily. An elaborate and solid solution for it is
therefore key. Although I do undersand that some of these ideas are very ambitious and we could maybe fully leave out different crafting methods for now as they do not
interfere with anything else. Later on when the more important things are done we can focus much more on this subject.

## Should we have light transmission and transparency?
Would be kind of cool but it kind of seems like a very insignificant feature that does not affect the core game loop at all so I think we shall remove it for now.

### @MrOcelotGuy
I agree, should be left for later.

## Should we have density and friction?
Bad thing is that it makes things more complicated and worse performance and such so once again the question is whether or not it is interesting for gameplay or not. Density
making it harder to move certain things makes things more realistic but also maybe annoying in some cases and you might have to calculate exactly when something has enough
friction to move things around it which might be slightly tedious. But I want it to be possible to have moving things and complex mechanics in a way that is more interesting
and nuanced than in Minecraft so this seems better than the Minecraft can/cannot be moved and does/doesn't drag blocks with it model in that way.

### @MrOcelotGuy
I think density and friction are kinda very key or at least some sort of rudimentary system that way you can have things like vehicles that you intuitively make yourself with
the game mechanics or all sorts of contraptions similar to create aeronautics I guess.

### @gusahlg
Yes then we go with density and friction, it seems even more obvious for us to have this system after some considering. 

## Other types of properties, yes or no?
Like I am not sure because adding in more kinds of properties add complexity quickly and is worse for memory. We want a simple core that players can expand onto. This
makes it less simple but gives way more oppertunities for people to make things on top. I don't want people to have to constantly look up the specific abilities of
elements all the time, that is just annoying so if there were some much simpler mechanic that allows there to be equally much experimentation and cool mechanics than this
that would be more welcome. The general rule of the game is that things should be as un-abstract as possible and be part of a solid and logical core, this makes it so that
players are never limited and have the most room for experimentation as possible. Okay I am quite convinced now, I will come up with a genius idea that fits perfectly instead.
Every part of the current idea should be critiqued and refined.

Okay so to refine the core idea of the game I have to lay out the current version in a concise and concrete paragraph:

World is made of blocks.

Blocks are made of elements.

There are core properties of elements. The average of all elements in a blocks properties is the blocks properties.

Special properties of elements are just like core properties but not all elements have them and the strength of their effect is calculated by the standard effect strength
multiplied by how big part that element makes up of the entire block (if the element makes up 50% of the block it would be multiplied by 0.5).

There is also the so called reaction properties that come when a specific selection of elements are present in a block at once and it gets stronger if the ratios are closer
to the preferred distribution.

Now that I see all of this I see that the problem is probably not the concept of properties because it is actually pretty neat and is a good interesting simplification
of the properties of real life things. But here are two things that feel a bit off and that have to be fixed for us to achieve a truly perfect core feature set. The first is
that there are different kinds of blocks that have different rulesets. I do not like this because it adds complexity and feels like an abstraction onto the core element
and block model. And the second is that it would make more sense and be cooler if reactions weren't just that elements where in the same block but that they were also combined
in a certain way through smelting or something. Hmmm ok I have an idea, what if you can effectively make new elements by smelting or processing other elements together and that
these new elements have the so called reaction properties plus the special and core properties of its components. This would make temperature in some form relevant again and
maybe other methods of combination could be added.

The first problem that there are mutliple kinds of blocks. Currently there are natural blocks that are very simple and basically won't be used in technical stuff ever because
they are not very practical in any way. There are also mixture, configuration and computational blocks and all have different structure and rules.
This is basically how the differnt blocks behave:
- Natural blocks basically only care about core properties and assumes that all of it components have the same ratio of the same block.
- Mixture blocks are like natural blocks but ratios of elements can be controled (as long as they together equal 100%) and these also have special and reaction properties.
- Configuration blocks also allows precise specification of where in the block the elements are spread out.
- Computational blocks are the most abstract block as they do not build upon elements but rather something made by logic gates and takes in electricity.

The first simplification is to remove mixture blocks and simply have there only be configuration blocks. The reason for this simplification is that anything that can be done
with mixture blocks can be done with configuration blocks as well so there is no sacrifice in functionality, configuration blocks are only mixture blocks with more control.
I also think that configuration blocks should be removed for now as I think it is an abstraction too big. This leaves us with two kinds of blocks: Natural blocks and
configuration blocks. The natural blocks feel necessary for memory to not be too atrocious so they probably have to stay a separate kind of block but they will not really
be used much at all for most tasks since they are very limited and do not really offer anything that the configuration blocks do not offer. The role of natural blocks shall
be that they are maybe cool for building sometimes and that they are mined and refined into other things. They have the core properties but nothing else because that would
not really be useful or worth it memory wise.

## How should crafting work?
As with all other things the solution has to be elegant and not abstract but also not super complex or hard to understand (this is hard to achieve of course). Crafting
cannot be tied to a block or a specific menu, it has to be something that can be automated but is not abstract. This means it has to be the result of some element or block
level core mechanic that causes the elements to combine into a block. I am not sure if there should even be manual crafting now that I think about it... Wouldn't it be cool
and kinda make sense if the player could effectively equip blocks in some way to be able to 'use' them? So you could make a block that crafts something and the player could
carry it around and use it. But this is probably overambitious and also not helpful since designing a general purpose crafter can probably not fit into whatever ruleset I
decide on anyway. Speaking of rulesets, what exactly should the conditions be for something to be crafted following the previous discussed goals?

## We gotta have block movement, but how?
We are not adding in some piston block as in minecraft, that is against the philosophy of the game and a more elaborate system is required. The issues that have to be resolved
to answer how this is supposed to work are what it is that makes a block move and how that should look and behave. We could have some system for making something be pushed out
of a block kind of how pistons in minecraft are but generalising that seems like an absolute pain and very complex so I think we should keep blocks independent (holy shit I
can't possibly remember the word I wanna use here, I feel like I can almost fine it but not quite), a general movement system could with this model only involve multy block
contraptions, not sure what I think about that. Maybe there should be a sort of pressure system so that high pressure can make blocks get pushed? The question is whether or
not a pressure system is fun or if we want something simpler and completely different, we can be original and totally unrealistic. Unrealistic does not mean unfun, Minecraft
proves this and that game is a great inspiration for this game.

## I think we might want to have a velocity system.
So basically walnutty asked me if there should be automining with a tunnnel bore or something and I realised that there was no collision damage system and then I realised
there is no velocity system at all and I want it to be possible to propell away a block due to an explosion or something and then have a collision with something else equal
damage. I think we just got to have this in the game to be honest, thanks walnut. We will have some sort of air resistance system as well though so that blocks do not travel
indefinitely.

## Also I think we will have alloys in the game.
So basically the concept of an alloy in the game is just the action of basically creating a new element (but instead of having the element label it has the alloy lable instead)
the difference between an alloy and an element is that it has the average core properties of the elements it is made of - all special properties (of the elements it is made of)
+ any reaction properties that exist between the elements it is made of. The basic condition for an alloy to be created is that it would have one or more reaction properties and
then every reaction property also has a specific environment condition that decides when it can be crafted. This condition can be for example reaching a certain temperature or
pressure or both at the same time, or something more specific like it being surrounded by a certain kind of element or something like that.

## Also I just realised we have to make it possible for elements to be moved around kind of like in a pipe.
Currently we only have block movement meaning the block can have velocity and be moved around but elements themselves can in no way be propelled within blocks and from blocks to
other blocks. Whilst this would be cool I think we gotta make some sort of decision around what is a block dynamic and element dynamic, is movement going to be only blocks or both?

## Overall concern, what is an element in a confiutation block, exactly how is the block "configured"?
As I have seen it so far I have thought of the configuration block basically being like a block with a bunch of small blocks in it being the elements. And configuration blocks can
also have alloys in them but the blocks themselves can not have special or reaction properties. But the question is how detailed the grid of elements in the configuration block
should be. If it is too detailed it will be tedious to edit it anyway and annoying + worse for memory and significantly harder to craft. It has to be simple, yet powerful, that
is the main thing. I think something like a 30x30x30 within grid for elements within the block might be an ideal compromise.

## Is there any way to simplify and go from blocks being one thing and elements inside of them being another dimension entirely and make some kind of shape generalisation?
This is interesting since it could mean that we could maybe have sub block things be crafted. This would mean that instead of there being only elements and blocks there are instead
instead inbetween things kind of. What I mean is that you can craft small fractions of blocks as well. This just means that the configuration block crafting should allow you to leave
a bunch of air in the blocks as well. It also means that these components should be reusable when crafting a new block. So basically configuration blocks that fit inside of other
configuration blocks can be placed there as well. I think the sub blocks should be nameable and reusable easily and that there should be a good interface for the configuration block
crafting but most of that is up to modding I'd say. I also think that multy block things should also be nameable and reproducable in the same way kind of like schematics.

## Maybe we should just have a crafting block or something?
I am starting to think that that might be the most fun solution for configuration blocks at least. Since like configuration blocks are immmensely complex things creating a general
crafting formula for them in a fun way seems like something close to an impossibiliy. I think we should maybe add in something like a crafting block for them that allows both player
and machines to craft blocks. Player can do so through an interface (mod) and machines can autocraft things by basically sending in signals to it through electricity. We could like
create some lore around this somehowk, saying that we got it from like an alien spaceship or somthing idk. Could be cool still. And it appeals to simplicity which after all is a major
goal although adding in a specific block for it is non-preferable for some reasons. I am saying maybe we do this for now and then someone can have a stroke of genius later on and make
something better

## Ok genius moment
What if we have an element called like stardust or whatever that acts as a determiner for how things are suppoesed to be crafted. Hmm idk this might be complex to I think my brain is
just fried but if we somehow could like put in a bit of stardust into a block to signal.... WAIT what if configuration blocks start off as being empty explicitly letting other blocks move
through it and that somehow elements can be pushed into place in the block and then the stardust signal thing is done and it kinda just says, ok I AM COMPLETE! and switches internal state
and done. Something like this could maybe work. Please give some feedback lol.

Ok I have to specify kinda exactly how this crafting thing should work now so that we can have a complete solution to one of the hardest problems in the game that we can then review and
refine. Okay so lets set up the details of how this works. First of all how is a configuration submitted into a configuration block? I had this idea that maybe the elements would be pushed
into the block that will be crafted and then craft signal leading to new block being made. Instead of a signal there could also be a condition that makes crafting trigger. And instead of
physically pushing elements into the block there could instead by some sort of way to like put multiple blocks together into one new block in a certain way where the placement can be translated
into element space. Or the third option is submitting a configuration and resources seperately. In a more organized format this could be expressed as:

Crafting execution solutions:
- Signals
- Conditions

Configuration and resources solutions:
- Input confiugration and resources seperately
- Somehow make it possible to put configuration into new block by surrounding it with other blocks that code for the configuration
- physically move elments into the block

A solution is to full on add in movement to all elements and not make movement block only. This makes me wonder whether we would want elements to be able to move though. Like the thing is
If elements moved configuration blocks would not have to exist anyway and it would be super messy and annoying I think so what about just saying no to this. I have another idea though. What if we can
push a natural block into a confiuration block and having it convert into elements. So that if we have a pure stone natural block pushing it into... wait I think I know how to do this. WHat if we have
the signaling be for when to convert a natural block into elements in a configuratin block? So like what if a signal is inputted into the configuration block, the configuration block changes some
state and then the next natural block that is pushed into it gets converted to elements in that block. Okay sure sure but how does one make precise things out of this? This gives little control over
where the elements go, which is a problem since that is the whole other half of the problem, controling configuration. Maybe the signal that the block gets in before-hand can hold information about
how the next elements are going to be placed? Yeah that might work. Maybe we can have an element called something cool. I think this idea is okay actuallly it is basically a sort of variation of the
combination of having a sort of signal and inputting configuration and resources seperately. I like it. The only real problem is that we have to have a way of editing the configuration by a player and
placing it out as a block. I am thinking it will look like a block made of element that are bluish transparent (kind of like a typical blueprint) this block should basically be its own kind of block
called something like a bluepring block or something and it should be possible to walk through and it should have very simple core properties. It should be super easy to push it into a configuration
block and it should then get deleted and update the place the next stuff go to in the configuration block. blueprint blocks take no resources to craft. I am thinking it would be good if they were
also craftable without the player there as well somehow. It doesn't matter if that is a little tedious as much since it is not the primary method. I am thinking that it should be possible to
automatically craft them by laying out natural blocks that are pure (made up of only one element) in a 30x30x30 configuration so that it is almost like a big block made out of blocks. Then there has
to be some signal or something that makes the 30x30x30 blocks combine into a single block. The signal would have to be something specific or there could be a condition of some sort.

OKAY so I want an alternative path because this is not purist enough to the original vision. THINKTHINKTHINK

# Unresolved problems list
There's probably a lot more here that is missing so just try adding in something whenever you come to think of something.
- Crafting
    - Processes
- Simulation
    - Electricity
    - Temperature
    - Block movement
- Block breaking
