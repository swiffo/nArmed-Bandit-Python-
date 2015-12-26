import math, random

class Bandit:
    """A one-armed bandit that outputs a normally distributed reward when used.
    Distribution is N(mu, sigma).
"""
    mu = 0
    sigma = 1

    def __init__(self, mu, sigma):
        """Sets the normal distribution parameters for the reward scheme"""
        self.mu = mu
        self.sigma = sigma

    def getReward(self):
        """Returns normaily distributed reward"""
        return random.gauss(self.mu, self.sigma)

class RewardEstimator:
    """Tracks estimates of rewards for n bandits."""

    estimates = None

    def initialize(self, firstEstimates):
        """Sets the first estimate of each bandit's reward based on the input list"""
        self.estimates = [(e,1) for e in firstEstimates]

    def receiveReward(self, banditNumber, reward):
        """Updates the reward estimate for bandit no. banditNumber based on the new reward"""
        (currentEstimate, currentCount) = self.estimates[banditNumber]
        newEstimate = (currentEstimate*currentCount + reward) / (currentCount + 1)
        self.estimates[banditNumber] = (newEstimate, currentCount + 1)

    def rewardEstimates(self):
        """Returns list of reward estimates"""
        return [x[0] for x in self.estimates]
            
class EpsilonExplorer(RewardEstimator):
    """A strategy which is greedy most of the time (1-epsilon) and exploratory some of the time (epsilon)."""
    epsilon = 0
    
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def chooseBandit(self):
        """Outputs bandit number to play"""
        rewardEstimates = self.rewardEstimates()
        numBandits = len(rewardEstimates)
        
        if random.random() < self.epsilon:
            return random.randrange(numBandits)
        else:
            bestReward = rewardEstimates[0]
            bestBandit = 0
            for k,v in enumerate(rewardEstimates):
                if v > bestReward:
                    bestReward = v
                    bestBandit = k
            return bestBandit

class BoltzmannExplorer(RewardEstimator):
    """A strategy which constantly explores but weights its choices towards high-reward choices based on
the Boltzmann-Gibbs distribution.

Each reward (/energy level) is weighted proportionally to exp(-Energy/temperature). Temperature is a
parameter of the explorer.
"""
    def __init__(self, temperature):
        self.temperature = temperature

    def chooseBandit(self):
        """Outputs bandit number to play."""

        weights = [math.exp(-e/self.temperature) for e in self.rewardEstimates()]
        total = sum(weights)
        choice = total * random.uniform(0, total)
        for banditNumber, weight in enumerate(weights):
            choice -= weight
            if choice < 0:
                break
        return banditNumber            

def main():
    bandits = [Bandit(0.5,1), Bandit(0,2), Bandit(1,2)]
    rewardEstimates = [b.getReward() for b in bandits]

    strategies = [
        EpsilonExplorer(0.01),
        EpsilonExplorer(0.1),
        BoltzmannExplorer(50),
        BoltzmannExplorer(270),
        BoltzmannExplorer(10),
    ]
    
    for strat in strategies:
        strat.initialize(rewardEstimates)

    gains = [0 for s in strategies]
    for n in range(10000):
        for numberStrat, strat in enumerate(strategies):
            chosenBandit = strat.chooseBandit()
            reward = bandits[chosenBandit].getReward()
            gains[numberStrat] += reward
            strat.receiveReward(chosenBandit, reward)

    print(gains)
     

    
         
            
