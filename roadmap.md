# durak-reinforcement-learning

Ok so  what's the point of  this project?  
Majorly to learn abotu concepts in ML  and agentic modelling.  In  this case we create a model for durak where we can create and launch agents to  playagainst  a player. So  final product is  a game  with agetns against who you can play.

So  what do  we need for this?

2 aims  - First - Learn, Second - create a cool product.

First things  first - create a rough outline of a plan.

To learn we need to  understand the game,  understand the concepts of  ML and agentic modelling, and then apply them to create agents that can play durak.
So  where do we start with learning?
1.  Get key readings on maths of durak and  ML and agentic modelling.  -  Understand why RL is applicable to this problem. 
2. Go through them, make key notes, polish the plan. 
3. Begin implemtnation.


Last:
To create a cool product we need to design the user interface and link it to the  backend.


Design:

Game engine for training agents and  playing against them  - can  be written in C++ for perforamnce and learning, then  binded with Python through pybind11. We might have millions of games running so performance is key.

1. Two agents play against each other (can be copies of the same network)
2. Each game produces a sequence of (state, action, reward) tuples
3. The network trains on those tuples
4. The updated agent plays more games against itself
5. Repeat — it gets better by playing itself