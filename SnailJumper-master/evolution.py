import copy
import os
from player import Player
import numpy as np
from operator import attrgetter


class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"

    def next_population_selection(self, players, num_players):  # entekhab e bazmandegan
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
        # selection based o fitness
        # players.sort(key=lambda x: x.fitness, reverse=True)

        # RW
        # players = self.roulette_wheel_selection(players, num_players)

        # Sus
        players = self.sus_selection(players, num_players)

        # Q-tournament algorithm implementation
        # players = self.Q_tournament_selection(players, num_players, 4)

        # to plot learning curve
        self.save_results(players)
        return players[: num_players]

    def generate_new_population(self, num_players, prev_players=None):  # parent selection and child production
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """

        first_generation = prev_players is None
        if first_generation:
            return [Player(self.game_mode) for _ in range(num_players)]
        else:
            # TODO ( Parent selection and child generation )
            # parent selection
            new_players = self.cross_over(prev_players)  # first we apply cross over then we apply mutation
            for child in new_players:
                self.mutation(child)
            return new_players

    def cross_over(self, prev_players):
        new_players = []
        for i in range(0, len(prev_players), 2):
            parent_1 = prev_players[i]
            parent_2 = prev_players[i + 1]
            new_child1 = self.clone_player(prev_players[i])
            new_child2 = self.clone_player(prev_players[i + 1])

            # crossover with one point
            '''
            for j in range(len(new_child1.nn.w)):  # j = 0 is ws between input and hidden layer
                # , j = 1 between hidden layer and output
                shape = new_child1.nn.w[j].shape  # shape[1] is column
                # first we replace 1/2 to the end of columns
                new_child1.nn.w[j][:, int(shape[1] / 2):] = parent_2.nn.w[j][:, int(shape[1] / 2):]
                new_child2.nn.w[j][:, int(shape[1] / 2):] = parent_1.nn.w[j][:, int(shape[1] / 2):]

            for k in range(len(new_child1.nn.b)):
                shape = new_child1.nn.b[k].shape  # shape[1] is column
                # first we replace 1/2 to the end of columns
                new_child1.nn.b[k][:, int(shape[1] / 2):] = parent_2.nn.b[k][:, int(shape[1] / 2):]
                new_child2.nn.b[k][:, int(shape[1] / 2):] = parent_1.nn.b[k][:, int(shape[1] / 2):]
            '''
            # crossover with 2 point
            for j in range(len(new_child1.nn.w)):  # j = 0 is ws between input and hidden layer
                # , j = 1 between hidden layer and output
                shape = new_child1.nn.w[j].shape  # shape[1] is column
                new_child1.nn.w[j][:, int(shape[1] / 3): 2 * int(shape[1] / 3)] =\
                    parent_2.nn.w[j][:, int(shape[1] / 3): 2 * int(shape[1] / 3)]
                new_child2.nn.w[j][:, int(shape[1] / 3): 2 * int(shape[1] / 3)] = \
                    parent_1.nn.w[j][:, int(shape[1] / 3): 2 * int(shape[1] / 3)]

            for k in range(len(new_child1.nn.b)):
                shape = new_child1.nn.w[j].shape  # shape[1] is column
                new_child1.nn.b[j][:, int(shape[1] / 3): 2 * int(shape[1] / 3)] = \
                    parent_2.nn.b[j][:, int(shape[1] / 3): 2 * int(shape[1] / 3)]
                new_child2.nn.b[j][:, int(shape[1] / 3): 2 * int(shape[1] / 3)] = \
                    parent_1.nn.b[j][:, int(shape[1] / 3): 2 * int(shape[1] / 3)]

            # crossover with 3 point
            '''for j in range(len(new_child1.nn.w)):  # j = 0 is ws between input and hidden layer
                # , j = 1 between hidden layer and output
                shape = new_child1.nn.w[j].shape  # shape[1] is column
                # first we replace 1/2 to the end of columns
                new_child1.nn.w[j][:, int(shape[1] / 4): int(shape[1] / 2)] =\
                    parent_2.nn.w[j][:, int(shape[1] / 4):int(shape[1] / 2)]
                new_child1.nn.w[j][:, 3 * int(shape[1] / 4):] =\
                    parent_2.nn.w[j][:, 3 * int(shape[1] / 4):]

                new_child2.nn.w[j][:, int(shape[1] / 4): int(shape[1] / 2)] = \
                    parent_1.nn.w[j][:, int(shape[1] / 4):int(shape[1] / 2)]
                new_child2.nn.w[j][:, 3 * int(shape[1] / 4):] = \
                    parent_1.nn.w[j][:, 3 * int(shape[1] / 4):]

            for k in range(len(new_child1.nn.b)):
                shape = new_child1.nn.w[j].shape  # shape[1] is column
                # first we replace 1/2 to the end of columns
                new_child1.nn.b[k][:, int(shape[1] / 4): int(shape[1] / 2)] = \
                    parent_2.nn.b[k][:, int(shape[1] / 4):int(shape[1] / 2)]
                new_child1.nn.b[k][:, 3 * int(shape[1] / 4):] = \
                    parent_2.nn.b[k][:, 3 * int(shape[1] / 4):]

                new_child2.nn.b[k][:, int(shape[1] / 4): int(shape[1] / 2)] = \
                    parent_1.nn.b[k][:, int(shape[1] / 4):int(shape[1] / 2)]
                new_child2.nn.b[k][:, 3 * int(shape[1] / 4):] = \
                    parent_1.nn.b[k][:, 3 * int(shape[1] / 4):]
            '''
            new_players.append(new_child1)
            new_players.append(new_child2)
        # print("number of new players", len(new_players))
        return new_players

    def mutation(self, child):

        '''
        mutation_threshold_row = 0.6  # for each neuron
        mutation_threshold_entry = 0.7  # for each weight of a specific neuron

        for i in range(len(child.nn.w)):
            if np.random.normal(0, 1) >= mutation_threshold_row:
                for j in range(len(child.nn.w[i])):
                    if np.random.normal(0, 1) >= mutation_threshold_entry:
                        child.nn.w[i][j] += np.random.normal(0, 1)
        for i in range(len(child.nn.b)):
            if np.random.normal(0, 1) >= mutation_threshold_row:
                for j in range(len(child.nn.b[i])):
                    if np.random.normal(0, 1) >= mutation_threshold_entry:
                        child.nn.b[i][j] += np.random.normal(0, 1)
        '''

        mutation_threshold = 0.7
        # mutation for each row of weights and biases
        for i in range(len(child.nn.w)):
            if np.random.normal(0, 1) >= mutation_threshold:
                child.nn.w[i] += np.random.normal(0, 1, size=(child.nn.w[i].shape))

        for i in range(len(child.nn.b)):
            if np.random.normal(0, 1) >= mutation_threshold:
                child.nn.b[i] += np.random.normal(0, 1, size=(child.nn.b[i].shape))

    def save_results(self, players):
        if not os.path.exists('result'):
            os.makedirs('result')

        f = open("result/result.txt", "a")
        for p in players:
            f.write(str(p.fitness))
            f.write(" ")
        f.write("\n")
        f.close()

    def roulette_wheel_selection(self, players, num_player):
        next_generation = []
        probs = []
        sum_fitness = 0
        for player in players:
            sum_fitness += player.fitness
        for player in players:
            probs.append(player.fitness / sum_fitness)
        selected = np.random.choice(a=players, size=num_player, replace=True, p=probs)
        for p in selected:
            child = self.clone_player(p)
            next_generation.append(child)
        return next_generation

    def sus_selection(self, players, num_players):
        next_generation = []
        probs = []
        sum_fitness = 0
        for player in players:
            sum_fitness += player.fitness
        for player in players:
            probs.append(player.fitness / sum_fitness)
        for i in range(1, len(players)):
            probs[i] += probs[i - 1]

        # defining second ruler in sus selection
        start_second_ruler = np.random.uniform(0, 1 / num_players, 1)  # random number between 0 , 1/n2
        second_ruler = []
        distance = 1 / num_players
        for i in range(num_players):  # building second ruler
            second_ruler.append(float(i * distance) + start_second_ruler)

        for point in second_ruler:
            for i, prob in enumerate(probs):
                if i == 0:
                    if point < probs[i]:
                        next_generation.append(self.clone_player(players[i]))
                else:
                    if probs[i] >= point > probs[i - 1]:
                        next_generation.append(self.clone_player(players[i]))

        return next_generation

    def Q_tournament_selection(self, players, num_players, q):
        selected = []
        for i in range(num_players):
            q_selections = np.random.choice(players, q)
            selected.append(max(q_selections, key=attrgetter('fitness')))
        return selected

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player
