This run went terribly! B0B guessed *around* the same value of rewards for **every** state. This suggests
I used too much L2 regularization. Punishing large weights too much pushed them all close to zero and
destroyed the model's ability to react to diverse situations. Guessing approximately the same reward every 
time means B0B chose to hide in the bottom right hand corner, which it repeatedly guessed rendered the highest
reward. 

Another thing I will try in B0B v2 is using He initialization instead of Glorot-Uniform, which I hear works
better.

Trial details:
* 3000 Training steps
* 3M action steps
* ~23.3 Hours (4 PM Apr 18 - 3:20 Apr 19 2025)
* 1 New way not to do this