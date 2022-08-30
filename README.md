# Snail-Jumper
This project aims to use the evolutionary algorithm for training a neural network in an environment where there is not enough data to train the network. One of these environments is the game, where something new is always happening; therefore, it is impossible to generate enough data for training the model.
To run the game with the help of neural evolution, we need to design a neural network that takes important decision-making parameters in its input and then produces the corresponding output. In the end, the output produced is like pressing the space button and will guide the game agent on its way to getting higher scores.

## Project description

<img width="603" alt="image" src="https://user-images.githubusercontent.com/72692826/178299331-0202a8c7-5285-451b-8204-65a0160ca70b.png">

The game is simple! a runner agent needs to jump over obstacles(snails and bugs) to survive. You can run the game in 2 modes: 1)manual mode, which you can play and enjoy(the agent jumps left and right using "space bar", see? not that sophisticated). 2)Neuroevoloution mode, which is even more exciting because you don't have to bother tapping on space 🥳! a generation of 300 agents is produced at the very beginning. In the next generations, you'll see that agents act better because of evolution! 
### The goal of this project is to implement an accurate neuroevolution algorithm, so agents get better in each new generation.
Note that the more an agent stays alive, the better fitness value it will earn.

## Project steps
1) Designing a proper neural network
- My NN is implemented in the nn.py file. You can see that initialization of weights and biases is done in the init method. Feedforwarding function and activation function are also defined in this class. Note that the sigmoid activation function is used in the hidden layer while softmax is used in the output layer(feel free to change it✌🏻). 
- The size of my NN is [8, 20, 2]. You can see the input values in the "Think" method in player.py(Think of this method as the brain of our model! it actually affects the results a lot). These input parameters define the important parameters that our model will consider to calculate the output(which is jumping left or right, that's why the output layer has only two neurons).
2) Designing a good evoloutionary algorithm (see evolution.py). 
- Next population selection --> Roulette wheel, SUS, and Q-tournament are implemented. Uncomment to see their results.
- Generating new population --> using cross-over and mutation(single point, two-point, and  three-point cross-over algorithms are implemented)

## Results
You can run this game using the game.py file and see the results of the Neuroevolution algorithm(Depending on the number of generations that passes) using plot.py.
Remember, you must delete the result directory before running the game to see the new result. I hope you can improve the results.🤓

<img width="633" alt="image" src="https://user-images.githubusercontent.com/72692826/178298609-f93f8df8-6d27-40df-917e-ff4ec804e1bc.png">
