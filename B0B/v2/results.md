Ha! Bingo. B0B is essentially being shown experiances where it performs badly mostly, so it just always guesses reward is a little below zero. Maybe we need to tone L2 reg down again but that doesn't seem to be the main issue. Maybe I need to separate experiances into positive and negative and use a mix during replay or maybe prioratise based on reward magnitude. It seems like cheating but maybe it's the best one can do with a 2017 Macbook Air.

Trained:
[[[-0.00822015 -0.0086507  -0.00923265 -0.00853496]]]
[[[-0.00821115 -0.0086414  -0.00922295 -0.00852574]]]
[[[-0.00816278 -0.00859141 -0.00917078 -0.00847618]]]
[[[-0.00808253 -0.00850848 -0.00908423 -0.00839396]]]
[[[-0.00806322 -0.00848853 -0.0090634  -0.00837417]]]
[[[-0.00797531 -0.00839769 -0.0089686  -0.0082841 ]]]
[[[-0.0079155  -0.00833589 -0.0089041  -0.00822283]]]

Untrained:
[[[239.49915 269.72925 266.89404  60.05403]]]
[[[245.56863 272.65695 276.13675  65.62411]]]
[[[231.84955 267.48978 280.5889   66.72857]]]
[[[174.61317 218.87865 289.01718 114.2593 ]]]
[[[ 61.56957  74.52809 264.6938  145.31021]]]
[[[ 51.241863  59.548843 243.24483  134.54106 ]]]
[[[ 56.808125  71.93408  245.88681  136.31119 ]]]

[[[ 177.53777 -392.87213 -357.0297   155.36858]]]
[[[ 144.65582 -325.09097 -356.66232  133.65662]]]
[[[ 103.61538 -621.9319  -519.3682    87.05024]]]
[[[ 155.40512 -449.8373  -495.6135   202.26126]]]
[[[ 110.83379 -600.88916 -518.8685   174.16151]]]
[[[  67.5563  -722.5067  -533.5925   128.44443]]]
[[[ 189.97603 -419.55045 -405.916    145.66806]]]

After this weight analysis I created qvals.txt for all versions.

Interestingly enough, this newer version of B0B produces q-values EVEN CLOSER TO ZERO!
That can't be from REDUCING L2 so it must be dropout? Random Chance?

Yea, results not so good as you can see by the reward guesses. Stayed in place again. 

What to try:
* Serious overhaul of reward shaping
* Massive improvement to replay.

Trial details:
* 3000 Training steps
* 3M action steps
* ~36 Hours (10:15 PM May 31 - 10:15 AM Jun 2 2025)
* 1 New way not to do this
* AWS