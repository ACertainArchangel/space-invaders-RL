### An excerpt from the B0B v1 `results.md`

>This run went terribly! B0B guessed *around* the samevalue of rewards for **every** state. This suggests
I used too much L2 regularization. Punishing large weights too much pushed them all close to zero and
destroyed the model's ability to react to diverse situations. Guessing approximately the same reward every 
time means B0B chose to hide in the bottom right hand corner, which it repeatedly guessed rendered the highest
reward.

The first run on T1 went very similarly, even with TD error prioratised replay.

>Another thing I will try in B0B v2 is using He initialization instead of Glorot-Uniform, which I hear works
better.

I will try these things for T1 v2 as well.

Trial details:
* 3000 Training steps
* 3M action steps
* 72.5 hours (3:30 PM Apr 17 - 4:00 PM Apr 20 2025)

Takeaways:
* Glorot uniform < He
* Tone down the L2 regularisation
* shape rewards a bit more
* Maybe try leaky RELU
* Fix the trainer, which I am realising now doesn't reset ammo or make it game over on 0 ammo. 

