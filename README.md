This is a hopefully much improved version of a neural network meant to play space invaders. The original in "dump of old potentially useful junk" just dumps raw coordinates into a dense perceptron with zero feature engineering. Recipie for desaster. This will be a CNN. New name too BTW:

# Deep space

This version is much improved, and you can see it chase enemies down, but it is too trigger happy because it has not learned ammo conservation. Next steps are to implement better reward shaping so it understands that movement is not just for killing enemies faster, but also so you don't run out of ammo.