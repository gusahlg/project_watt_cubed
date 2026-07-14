# World generation
I had the genius idea that since we have already decided upon infinite world generation in all 3 axes why not build more on top of this?
Something that I always wanted to do in Minecraft when I was little was to make a space ship and fly up forever to see how far I could go.
This feeling is magical, you feel like you are exploring entirely uncharterd territory and you don't know what you might find if you go far
enough. What I want to do with the game is to make the starting place be an extremely big circular-ish planet and then also add in a few other
remote planets that you can get to through your own contraptions. This would be very fun. Although this does basically require me to implement
gravity since being able to fall through space at an ever increasing space for no particular reason is quite odd and annoying behaviour. I do
not want to add in gravity for blocks (I don't think, please argue with me on this) but maybe there could be a simplified form of gravity for the
player. Another slightly smaller issue to fix is that we do not want the player to be walking around upside down when it's on the other side of a
planet. My fix for this is to have the player's feet be phasing the direction of the gravitational pull and having the camera also move with it.
