### Using a detached monte carlo tree to generate heatmaps through game data
Store positions and their info in duckdb, once i have enough, use it to generate heat maps for pieces in different stages of the game.
Need to rebuild analysis and quiescence search in addition to strengthen all sides

## treeBuilder.py will build tree data to a file in serializable form.  
just use pickle.dumps() to convert it back to the {string:node} dict once you've built and retreived it.
It will make a dictionary of unassociated nodes that each have stats inside them.
It also stores info in a local duckdb