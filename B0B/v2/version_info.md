B0B v1 had L2 regularisation of 0.002; this version has 0.0001 (20 times lower). 
This may be radical but we will see where it goes.

B0B v1 had a dropout of 0.3 on all layers but here we reduce it to 0.2 in the second layer and 0.1 in the last, since
dropout becomes less usefull deeper in the network.

De-cluttered some code and removed remenants of TD error that just slowed us down.