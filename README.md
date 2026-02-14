### Creating a Monte Carlo Tree Search with persistent node data for chess positions starting from 0.

 Still working on the mcts aspect of this, tree.py theoretically should do the job but it needs work.

## treeBuilder.py will build tree data to a local redis container in serializable form.  
just use pickle.dumps() to convert it back to the {string:node} dict once you've built and retreived it.
It will make a dictionary of unassociated nodes that each have stats inside them.  

I want to make a way to visualize the nodes, and how they connect as it plays games in real time but this may be out of reach for now.

In theory, this should be able to get a good direction for what early moves are good based on continuously simulated percentages.
It's executing effeciently and wrangling the node data that's the real challenge