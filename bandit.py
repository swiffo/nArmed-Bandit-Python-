import math, random, matplotlib.pyplot as plt

class Bandit:
    """A one-armed bandit that outputs a normally distributed reward when used.
    Distribution is N(mu, sigma).
"""
    def __init__(self, mu, sigma):
        """Sets the normal distribution parameters for the reward scheme"""
        self.mu = mu
        self.sigma = sigma

    def getReward(self):
        """Returns normaily distributed reward"""
        return random.gauss(self.mu, self.sigma)

    def describe(self):
        return "Bandit({mu:.1f}, {sigma:.1f})".format(mu=self.mu, sigma=self.sigma)

class RewardEstimator:
    """Tracks estimates of rewards for n bandits.

Must be initialized with one observation for each bandit. The rewards are estimates as simple
averages of past rewards.
"""

    estimates = None # Set to None to make it clear whether the object was initialized

    def initialize(self, firstEstimates):
        """Sets the first estimate of each bandit's reward based on the input list"""

        # We store estimates as (current estimate, number of observations the estimate is based on)
        # The observation count is needed when updating the estimate with new observations.
        self.estimates = [(e,1) for e in firstEstimates]

    def receiveReward(self, banditNumber, reward):
        """Updates the reward estimate for bandit no. banditNumber based on the new reward"""
        (currentEstimate, currentCount) = self.estimates[banditNumber]
        newEstimate = currentEstimate + (reward - currentEstimate ) / (currentCount+1)
        self.estimates[banditNumber] = (newEstimate, currentCount + 1)

    def rewardEstimates(self):
        """Returns list of reward estimates"""
        return [x[0] for x in self.estimates]
            
class EpsilonExplorer(RewardEstimator):
    """A strategy which is greedy most of the time (1-epsilon) and exploratory some of the time (epsilon)."""

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def describe(self):
        """Human-readable descriptor"""
        return "Epsilon-explorer({:.1%})".format(self.epsilon)

    def chooseBandit(self):
        """Returns bandit number to play"""
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

class DescendingEpsilonExplorer(EpsilonExplorer):
    """A strategy which is greedy some of the time and exploratory the rest of the time.

Chance of exploration is epsilon which decreases with time based on the descent rate.
"""
    def __init__(self, epsilon, descentRate):
        self.originalEpsilon = epsilon
        EpsilonExplorer.__init__(self, epsilon)
        self.descentRate = descentRate

    def describe(self):
        """Human-readable descriptor"""
        return "Descending epsilon-explorer(eps={:.1%}, desc={:.3f})".format(self.originalEpsilon, self.descentRate)

    def chooseBandit(self):
        """Returns bandit numer to play"""
        self.epsilon *= self.descentRate
        return EpsilonExplorer.chooseBandit(self)
        

class BoltzmannExplorer(RewardEstimator):
    """A strategy which constantly explores but weights its choices towards high-reward
outcomes based on the Boltzmann-Gibbs distribution.

Each reward (/energy level) is weighted proportionally to exp(-Energy/temperature). Temperature is a
parameter of the explorer.
"""
    def __init__(self, temperature):
        self.temperature = temperature # The higher the temperature, the more equalisation of choice weights

    def describe(self):
        """Returns human-readable descriptor"""
        return "Boltzmann-explorer({:.1f}K)".format(self.temperature)

    def chooseBandit(self):
        """Returns bandit number to play."""

        weights = [math.exp(estimate/self.temperature) for estimate in self.rewardEstimates()]
        total = sum(weights)
        choice = random.uniform(0, total)
        for banditNumber, weight in enumerate(weights):
            choice -= weight
            if choice < 0:
                break
        return banditNumber            

def main(numBandits, iterations):
    """Run the n-bandit propblem with (numBandits) bandits and with (iterations) games."""
    
    bandits = [Bandit(random.uniform(1,10), random.uniform(1, 5) ) for _b in range(numBandits)]
    rewardEstimates = [b.getReward() for b in bandits]
    expectedRewards = [b.mu for b in bandits]
    bestBanditChoice = expectedRewards.index(max(expectedRewards))

    print("*** Bandits and initial reward estimates ***")
    print("\n".join(["{bandit} with estimate {est:.1f}".format(bandit=b.describe(), est=e) for (b,e) in zip(bandits, rewardEstimates)]))
          
    strategies = [
        EpsilonExplorer(0),
        EpsilonExplorer(0.01),
        EpsilonExplorer(0.1),
        DescendingEpsilonExplorer(0.1, 0.999),
        BoltzmannExplorer(1),
        BoltzmannExplorer(0.5),
        BoltzmannExplorer(2),
    ]
    
    for strat in strategies:
        strat.initialize(rewardEstimates)

    # Run the simulation for each strategy, recording the gains
    gainHistories = [[0] for s in strategies]
    gains = [0 for s in strategies]
    choiceCorrectness = [[0] for s in strategies] # Stores average correctness of choice; starts with 0 for ease of implementation
    
    for n in range(iterations):
        for numberStrat, strat in enumerate(strategies):
            chosenBandit = strat.chooseBandit()
            reward = bandits[chosenBandit].getReward()
            choiceCorrectness[numberStrat].append(
                choiceCorrectness[numberStrat][len(choiceCorrectness[numberStrat])-1]*n/(n+1) + (chosenBandit==bestBanditChoice)/(n+1)
            )
            gains[numberStrat] += reward
            gainHistories[numberStrat].append(gains[numberStrat]/(n+1))
            strat.receiveReward(chosenBandit, reward)

    # Print out total gain for each strategy as the simplest measure of success
    print("\n*** Total rewards accumulated ***")
    for (s,g) in zip(strategies, gains):
        print( "{} gained {:.0f}".format(s.describe(), g) )

    # Plot the gains history for each strategy to see how quickly each strategy learned
    # and what slope it ended settling on.
    handles = []
    for (hist, s) in zip(gainHistories, strategies):
        h, = plt.plot(hist, label=s.describe())
        handles.append(h)
    plt.legend(handles=handles, loc=4) # Lower right
    plt.title('Average Rewards')
    plt.show()

    # Plot the average number of best choices over time    
    handles = []
    for (correctness, s) in zip(choiceCorrectness, strategies):
        h, = plt.plot(correctness, label=s.describe())
        handles.append(h)
    plt.legend(handles=handles, loc=4) # Lower right
    plt.title('Average Correctness of Choice')
    plt.show()
     

    
         
            
