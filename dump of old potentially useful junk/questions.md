# Questions about my project for ~~Dr. Shukla~~ Future Me

1. Do you think a project like this could feasibly continue with a $0 budget? If you were working on this, would a $0 budget constrain you and if not, how would you overcome the challenge? -> I have google colab student now. Problem solved.

2. What do you think of my current approach of parallelization? Do you have a better way that immediately comes to mind? Can my program be parallelized on a GPU (specifically the T4 I have access to) or would I have to do a major structural overhaul. -> You need a GPU. You have an A100 now so you're all set. It's not parallelisation of the model you need btw see below.

3. What do you think about my implementation of TD error prioritized repay? Is there a way to optimize it? Recomputing every training cycle is a bit computationally expensive but from what I can tell it seems standard. -> Cut it for now and add it back once fixing fundemental flaws.

4. Seeing as enough computational power for hyperparameter tuning is clearly out of my budget, do you think I am on the right track with my current method of testing T1 and B0B and adjusting their hyperparameters based on how I see them perform? -> Budget problem solved.

5. **Do you have any general advice about the project?** I consider this the most important question as you will for sure know best what it is I need to hear. -> Yes. Raw coordinates are AWEFUL input. Make this a CNN.

6. Do you think there are any important questions I have failed to ask you? -> You failed to ask about feature engineering, because the current feature engineering is AWEFUL.

Questions answered by a me that knows more about deep learning now.