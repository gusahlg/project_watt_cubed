# Design philosophy
Everything in this game should be designed around a few core principles: Simplicity, Freedom, Community and Exploration
Simplicity represents the urge to make great things from small things and not making things bloated with edge cases. This is important because
simplicity is very satisfying and allows for things to be built on top more easily. The game aims to be a perfect thin core that is interesting
but allows for things to be built on top with ease. Freedom is the value of everything being transparent. The code is entirely free and open
source. Freedom also means that people can easily build things around the game and in the game in a totally free way. The game should support
as much customisation as possible and there should be few limitations for what things you can make of the gameplay. Community represents the
value of a strong and active community. The game is nothing without people playing it and with all the freedom that the community has it is also
and invaluable asset for creating more content and diversifying the game and the gameplay into many new areas. The community aspect is also why
must have support for massive multiplayer server in mind from the start and also make sure to have good and fun in game conmmunication systems 
like proximity chat and a good chat system. Exploration is the value of letting the player explore almost endless different possibilities in the
game. We want the game to allow for many different things and experiences that the players themselves define. Exploration also means that the
world should be interesting to explore.

# Code philosophy
The code should always be very modular and may heavily use macros and generics to achieve minimal repetition. An important part is that the code
should be maintainable and practical long term but also expose key parts to a simple api that can be used for writing mods. This means that the
complexity should be varied with more simple code as well. Naming should be very simple, no jargon or anything just the most simple name that
works with naming conventions and conveys pupose in a clear and non-contradictory way. The comment style is conservative, write overall purpose
of the file at the top of the file (this matches way with the modular rust thinking where one file often equals one explicit sub system)
concisely and short comments explaining more complex parts of the code in short. The short explanations should convey purpose and surface level
technical stuff but never go into explain underlying concepts as that is the responsibility of the person viewing the comments to know. The most
important part of the code philosophyis that optimisationa and effiency always goes ahead of readability and ease. It is essential that the code
is as optimized as possible as that enables more powerful things within the game.
