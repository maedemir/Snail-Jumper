# Snail-Jumper
This project is an implementation of evolutionary algorithms and neural networks(Neuroevolution) for Computational Intelligence course at AUT.

## Project description

<img width="603" alt="image" src="https://user-images.githubusercontent.com/72692826/178299331-0202a8c7-5285-451b-8204-65a0160ca70b.png">


The game is simple! a runner agent needs to jump over obstacls(snails and bugs) to survive. you can run the game in 2 modes: 1)manual mode which you can play and enjoy(the agent jumps left and right using "space bar", see? not that sophisticated). 2)Neuroevoloution mode which is even more exciting because you don't have to bother tapping on space 🥳! a generetion of 300 agents are produced at the very beginning. In the next generations, you'll see that agents are acting better because of evoloution! 
### The goal of this project is to implement an accurate neuroevoloution algorithm so that agents get better in each new generation.
Note that the more an agent stays alive, the better fittness value it will earn.
## Project steps
1) designing a proper neural network
- my NN is implemented in nn.py file. you can see that initialization of weights and biases is done in init method. feedforwarding function and activation function are also defined in this class. note that sigmoid activation function is used in hidden layer while softmax is used in output layer(feel free to change it✌🏻). 
- The size of my NN is [8, 20, 2]. you can see the input values in "Think" method in player.py(Think of this method as the brain of our model! it actually effects the results alot). this input parameters define the important parameters that our model will consider to calculate the output(which is jumping left or right, thats why output layer has only 2 neurons)
2) designing a good evoloutionary algorithm

## Results
<img width="633" alt="image" src="https://user-images.githubusercontent.com/72692826/178298609-f93f8df8-6d27-40df-917e-ff4ec804e1bc.png">
