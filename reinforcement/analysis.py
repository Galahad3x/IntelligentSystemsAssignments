# analysis.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    """
        ANSWER : answerNoise = 0

        By setting the noise value to 0, we are indicating that it will always jump where 
        expected. In this way, when calculating the action values, the negative values due to 
        a random jump out of the way will not influence the the agent's decision. This prevents 
        the agent from tending to go to the westernmost cell (with value 1.0) in order to maximize 
        the reward.
    """
    answerDiscount = 0.9
    answerNoise = 0
    return answerDiscount, answerNoise

def question3a():
    """
        ANSWER : 
        answerDiscount = 0.9
        answerNoise = 0.1
        answerLivingReward = -5

        Negatively penalizing staying alive, causes the agent to approach the terminal position 
        as soon as possible with a score of 1.0 in order to maximize the reward.
    """
    answerDiscount = 0.9
    answerNoise = 0.1
    answerLivingReward = -5
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3b():
    """
        ANSWER : 
        answerDiscount = 0.5
        answerNoise = 0.2
        answerLivingReward = -2

        Reducing the impact of the lifetime penalty allows the agent a range of options to explore. 
        However, the influence of a random jump over the ciff can make the agent tend to achieve
        the objective of the previous question, therefore it is necessary to reduce this impact 
        by reducing the discount factor.
    """
    answerDiscount = 0.5
    answerNoise = 0.2
    answerLivingReward = -2
    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    """
        ANSWER : 
        answerDiscount = 0.8
        answerNoise = 0.1
        answerLivingReward = -0.5

        By slightly negatively penalizing staying alive, the agent tends to try to explore 
        more space by missing the terminal position with a reward of 1.0. But tending to go to 
        the position with value 10 so as not to end up with a negative reward.

    """
    answerDiscount = 0.8
    answerNoise = 0.1
    answerLivingReward = -0.5
    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    """
        ANSWER : 
        answerDiscount = 0.9
        answerNoise = 0.2
        answerLivingReward = 0

        By not receiving a reward for living, the agent will not be forced to go to the nearest
        terminal position, however, the cliff penalty effects received by a random jump cause 
        the agent to try not to play it. On the other hand, as it approaches the terminal reward 
        position 10, its influence due to not receiving any other reward will tend for the agent 
        to go to that position.

    """
    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward  

def question3e():
    """
        ANSWER : answerLivingReward = 10
        
        By setting the value of answerLivingRewards to 10 as the agent seeks to maximize 
        the reward, it will tend to avoid taking actions that lead to a terminal state. 
        Making it never stop
    """
    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 10 
    return answerDiscount, answerNoise, answerLivingReward

def question8():
    answerEpsilon = None
    answerLearningRate = None
    return answerEpsilon, answerLearningRate
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
