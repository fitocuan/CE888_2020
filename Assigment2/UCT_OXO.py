# This is a very simple implementation of the UCT Monte Carlo Tree Search algorithm in Python 2.7.
# The function UCT(rootstate, iteramax, verbose = False) is towards the bottom of the code.
# It aims to have the clearest and simplest possible code, and for the sake of clarity, the code
# is orders of magnitude less efficient than it could be made, particularly by using a
# state.GetRandomMove() or state.DoRandomRollout() function.
#
# Example GameState classes for Nim, OXO and Othello are included to give some idea of how you
# can write your own GameState use UCT in your 2-player game. Change the game to be played in
# the UCTPlayGame() function at the bottom of the code.
#
# Written by Peter Cowling, Ed Powley, Daniel Whitehouse (University of York, UK) September 2012.
#
# Licence is granted to freely use and distribute for any sensible/legal purpose so long as this comment
# remains in any distributed code.
#
# For more information about Monte Carlo Tree Search check out our web site at www.mcts.ai

from math import *
import random
import pandas as pd
import numpy as np
import pickle
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
import sys
import matplotlib.pyplot as plt


class GameState:
    """ A state of the game, i.e. the game board. These are the only functions which are
        absolutely necessary to implement UCT in any 2-player complete information deterministic
        zero-sum game, although they can be enhanced and made quicker, for example by using a
        GetRandomMove() function to generate a random move during rollout.
        By convention the players are numbered 1 and 2.
    """

    def __init__(self):
        self.playerJustMoved = 2  # At the root pretend the player just moved is player 2 - player 1 has the first move

    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = GameState()
        st.playerJustMoved = self.playerJustMoved
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerJustMoved.
        """
        self.playerJustMoved = 3 - self.playerJustMoved

    def GetMoves(self):
        """ Get all possible moves from this state.
        """

    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm.
        """

    def __repr__(self):
        """ Don't need this - but good style.
        """
        pass


class OXOState:
    """ A state of the game, i.e. the game board.
        Squares in the board are in this arrangement
        012
        345
        678
        where 0 = empty, 1 = player 1 (X), 2 = player 2 (O)
    """

    def __init__(self):
        self.playerJustMoved = 2  # At the root pretend the player just moved is p2 - p1 has the first move
        self.board = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # 0 = empty, 1 = player 1, 2 = player 2

    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = OXOState()
        st.playerJustMoved = self.playerJustMoved
        st.board = self.board[:]
        return st

    def getSample(self):
        x = self.board[:]
        x.append(self.playerJustMoved)
        return x[:]

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerToMove.
        """
        assert move >= 0 and move <= 8 and move == int(move) and self.board[move] == 0
        self.playerJustMoved = 3 - self.playerJustMoved
        self.board[move] = self.playerJustMoved

    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        return [i for i in range(9) if self.board[i] == 0]

    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm.
        """
        for (x, y, z) in [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]:
            if self.board[x] == self.board[y] == self.board[z]:
                if self.board[x] == playerjm:
                    return 1.0
                else:
                    return 0.0
        if self.GetMoves() == []: return 0.5  # draw
        return False  # Should not be possible to get here

    def __repr__(self):
        s = ""
        for i in range(9):
            s += ".XO"[self.board[i]]
            if i % 3 == 2: s += "\n"
        return s


class Node:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
        Crashes if state not specified.
    """

    def __init__(self, move=None, parent=None, state=None):
        self.move = move  # the move that got us to this node - "None" for the root node
        self.parentNode = parent  # "None" for the root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.untriedMoves = state.GetMoves()  # future child nodes
        self.playerJustMoved = state.playerJustMoved  # the only part of the state that the Node needs later

    def UCTSelectChild(self):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        s = sorted(self.childNodes, key=lambda c: c.wins / c.visits + sqrt(2 * log(self.visits) / c.visits))[-1]
        return s

    def AddChild(self, m, s):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        n = Node(move=m, parent=self, state=s)
        self.untriedMoves.remove(m)
        self.childNodes.append(n)
        return n

    def Update(self, result):
        """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.wins += result

    def __repr__(self):
        return "[M:" + str(self.move) + " W/V:" + str(self.wins) + "/" + str(self.visits) + " U:" + str(
            self.untriedMoves) + "]"

    def TreeToString(self, indent):
        s = self.IndentString(indent) + str(self)
        for c in self.childNodes:
            s += c.TreeToString(indent + 1)
        return s

    def IndentString(self, indent):
        s = "\n"
        for i in range(1, indent + 1):
            s += "| "
        return s

    def ChildrenToString(self):
        s = ""
        for c in self.childNodes:
            s += str(c) + "\n"
        return s


def UCT(rootstate, iteramax, verbose=False, model=None, model2 = None):
    """ Conduct a UCT search for iteramax iteraations starting from rootstate.
        Return the best move from the rootstate.
        Assumes 2 alternating players (player 1 starts), with game results in the range [0.0, 1.0]."""

    rootnode = Node(state=rootstate)


    for i in range(iteramax):
        node = rootnode
        state = rootstate.Clone()

        # Select

        turn = False
        # Expand
        if node.untriedMoves != []:  # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.untriedMoves)
            state.DoMove(m)
            node = node.AddChild(m, state)  # add child and descend tree
        else:
            while node.untriedMoves == [] and node.childNodes != []:  # node is fully expanded and non-terminal
                node = node.UCTSelectChild()
                state.DoMove(node.move)


        # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
        while state.GetMoves() != []:  # while state is non-terminal

            if model != None:

                board = np.array(state.getSample()[:])
                board[-1] = 3 - board[-1]

                # Predict move and generate a random move
                board = np.array(board)
                pred_move = int(model.predict(board.reshape(1, -1)))

                rand_move = random.choice(state.GetMoves())

                move = np.random.choice([pred_move, rand_move], p=[0.9, 0.1])

                #Alternating Policies
                if turn:
                    move = rand_move
                else:
                    move = move if (move in state.GetMoves()) else rand_move
            else:
                move = random.choice(state.GetMoves())

            turn = not turn

            state.DoMove(move)

        # Backpropagate
        while node != None:  # backpropagate from the expanded node and work back to the root node
            node.Update(state.GetResult(
                node.playerJustMoved))  # state is terminal. Update node with result from POV of node.playerJustMoved
            node = node.parentNode

    # Output some information about the tree - can be omitted
    # if verbose: print(rootnode.TreeToString(0))
    # else: print(rootnode.ChildrenToString())

    return sorted(rootnode.childNodes, key=lambda c: c.visits)[-1].move  # return the move that was most visited


agents = []
results = [0, 0, 0]
data_set = []
data_itera = []
count = 1
data_set_count = 0


def UCTPlayGame(agent1, agent2, training, expand=None, itera = 100):
    """ Play a sample game between two UCT players where each player gets a different number
        of UCT iteraations (= simulations = tree nodes).
    """
    global data_set
    global data_itera
    global count
    global data_set_count

    state = OXOState()  # uncomment to play OXO

    while state.GetMoves() != []:
        # print(str(state))
        if state.playerJustMoved == 1:
            m = UCT(rootstate=state,iteramax= itera, verbose=False, model=agent2, model2 = agent1)
        else:
            m = UCT(rootstate=state,iteramax= itera, verbose=False, model=agent1, model2 = agent2)
        # print("Best Move: " + str(m) + "\n")
        state.DoMove(m)

        #Adding instance to dataset
        if training:
            board = state.getSample()[:]
            board[m] = 0
            board.append(m)
            data_itera.append(board)

        if state.GetResult(state.playerJustMoved) != False:
            # print(str(state))
            break

    # Creating new agent
    if training and count % data_set_count == 0:


        data_arr = np.array(data_itera)

        # Generate global dataset
        if len(data_set) == 0:
            data_set = data_arr
        else:
            data_arr_sample = data_arr[-int(expand*len(data_set)):,:]
            data_set = np.vstack((data_set, data_arr_sample))
            

        #Train Decision tree

        X = data_set[:, :-1]
        y = data_set[:, -1]
        clf = tree.DecisionTreeClassifier()
        #X, X_test, y, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        clf.fit(X, y)

        #Create new agent
        agents.append(clf)
        data_itera = []
        count = 0

    count += 1

    result = [0, 0, 0]

    if state.GetResult(state.playerJustMoved) == 1.0:
        #print("Player " + str(state.playerJustMoved) + " wins!")
        result[state.playerJustMoved] += 1
    elif state.GetResult(state.playerJustMoved) == 0.0:
        #print("Player " + str(3 - state.playerJustMoved) + " wins!")
        result[3 - state.playerJustMoved] += 1
    else:
        # print("Nobody wins!")
        result[0] += 1

    return np.array(result)


if __name__ == "__main__":
    """ Play a single game to the end using UCT for both players. 
    """

    #Parameters
    data_set_count = 200 #Number of games from which data is gathered to train each agent
    expand = 0.5 #Percetange of expansion for the data set
    agents_num = 10 #Number of agents to create
    games_test = 250 #Number of games to play for testing

    overall_res = np.zeros((1, 3))

    print("Starting....")

    #Create first agent
    while (len(agents) == 0):
        UCTPlayGame(None, None, True, expand, 100)

    print("First Agent Created")

    overall_res = np.zeros((1, 3))

    sys.stdout.write("Creating Agents... ")

    #Create remaining agents
    c = 0
    while len(agents) != agents_num:
        UCTPlayGame(agents[-1], agents[-1], True, expand, 100)
        if (len(agents) > c):
            sys.stdout.write("#")
            c += 1

    print(" Finish Creating Agents")
    print("Testing...")


    #Testing Agents Matches
    overall_res = np.zeros((1, 3))
    res_total = np.empty((0, 3), int)

    for p_agent in range(agents_num):

        overall_res = np.zeros((1, 3))

        for i in range(games_test):
            overall_res += UCTPlayGame(agents[p_agent], None, False, expand, 30)
        res_total = np.append(res_total, overall_res, axis=0)
        sys.stdout.write(str(p_agent) + ") ")

        print(overall_res)

    #Save Results in CSV file
    #np.savetxt("lastagt.csv", res_total, delimiter=",")


    #Plot Results
    plt.plot(range(agents_num), res_total[:, 0], color='r')
    plt.plot(range(agents_num), res_total[:, 1], color='b')
    plt.plot(range(agents_num), res_total[:, 2], color='g')
    plt.show()

