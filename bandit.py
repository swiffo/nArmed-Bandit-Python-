import math
import random
import matplotlib.pyplot as plt

class Bandit:
    """A one-armed bandit that outputs a normally distributed reward when used.
    Distribution is N(mu, sigma).
    """
    def __init__(self, mu, sigma):
        """Sets the normal distribution parameters for the reward scheme"""
        self.mu = mu
        self.sigma = sigma

    def get_reward(self):
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

    def initialize(self, first_estimates):
        """Sets the first estimate of each bandit's reward based on the input list"""

        # We store estimates as (current estimate, number of observations the estimate is based on)
        # The observation count is needed when updating the estimate with new observations.
        self.estimates = [(e, 1) for e in first_estimates]

    def receive_reward(self, bandit_number, reward):
        """Updates the reward estimate for bandit no. bandit_number based on the new reward"""
        (current_estimate, current_count) = self.estimates[bandit_number]
        new_estimate = current_estimate + (reward - current_estimate ) / (current_count+1)
        self.estimates[bandit_number] = (new_estimate, current_count + 1)

    def reward_estimates(self):
        """Returns list of reward estimates"""
        return [x[0] for x in self.estimates]
            
class EpsilonExplorer(RewardEstimator):
    """A strategy which is greedy most of the time (1-epsilon) and exploratory some of the time (epsilon)."""

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def describe(self):
        """Human-readable descriptor"""
        return "Epsilon-explorer({:.1%})".format(self.epsilon)

    def choose_bandit(self):
        """Returns bandit number to play"""
        reward_estimates = self.reward_estimates()
        num_bandits = len(reward_estimates)
        
        if random.random() < self.epsilon:
            return random.randrange(num_bandits)
        else:
            best_reward = reward_estimates[0]
            best_bandit = 0
            for k,v in enumerate(reward_estimates):
                if v > best_reward:
                    best_reward = v
                    best_bandit = k
            return best_bandit

class DescendingEpsilonExplorer(EpsilonExplorer):
    """A strategy which is greedy some of the time and exploratory the rest of the time.

    Chance of exploration is epsilon which decreases with time based on the descent rate.
    """
    def __init__(self, epsilon, descent_rate):
        self.original_epsilon = epsilon
        EpsilonExplorer.__init__(self, epsilon)
        self.descent_rate = descent_rate

    def describe(self):
        """Human-readable descriptor"""
        return "Descending epsilon-explorer(eps={:.1%}, desc={:.3f})".format(self.original_epsilon, self.descent_rate)

    def choose_bandit(self):
        """Returns bandit number to play"""
        self.epsilon *= self.descent_rate
        return EpsilonExplorer.choose_bandit(self)
        

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

    def choose_bandit(self):
        """Returns bandit number to play."""

        weights = [math.exp(estimate/self.temperature) for estimate in self.reward_estimates()]
        total = sum(weights)
        choice = random.uniform(0, total)
        for bandit_number, weight in enumerate(weights):
            choice -= weight
            if choice < 0:
                break
        return bandit_number


def get_strategies(reward_estimates):
    """Return list of strategies initialized wiht reward_estimates"""
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
        strat.initialize(reward_estimates)
    return strategies

def main(num_bandits, iterations):
    """Run the n-bandit propblem with (num_bandits) bandits and with (iterations) games."""
    
    bandits = [Bandit(random.uniform(1,10), random.uniform(1, 5) ) for _b in range(num_bandits)]
    reward_estimates = [b.get_reward() for b in bandits]
    expected_rewards = [b.mu for b in bandits]
    best_bandit_choice = expected_rewards.index(max(expected_rewards))

    print("*** Bandits and initial reward estimates ***")
    print("\n".join(["{bandit} with estimate {est:.1f}".format(bandit=b.describe(), est=e) for (b,e) in zip(bandits, reward_estimates)]))

    strategies = get_strategies(reward_estimates)
 
    # Run the simulation for each strategy, recording the gains
    gain_histories = [[0] for s in strategies]
    gains = [0 for s in strategies]
    choice_correctness = [[0] for s in strategies] # Stores average correctness of choice; starts with 0 for ease of implementation
    
    for n in range(iterations):
        for number_strat, strat in enumerate(strategies):
            chosen_bandit = strat.choose_bandit()
            reward = bandits[chosen_bandit].get_reward()
            choice_correctness[number_strat].append(
                choice_correctness[number_strat][len(choice_correctness[number_strat])-1]*n/(n+1) 
                + (chosen_bandit==best_bandit_choice)/(n+1)
            )
            gains[number_strat] += reward
            gain_histories[number_strat].append(gains[number_strat]/(n+1))
            strat.receive_reward(chosen_bandit, reward)

    # Print out total gain for each strategy as the simplest measure of success
    print("\n*** Total rewards accumulated ***")
    for (s,g) in zip(strategies, gains):
        print( "{} gained {:.0f}".format(s.describe(), g) )

    # Plot the gains history for each strategy to see how quickly each strategy learned
    # and what slope it ended settling on.
    handles = []
    for (hist, s) in zip(gain_histories, strategies):
        h, = plt.plot(hist, label=s.describe())
        handles.append(h)
    plt.legend(handles=handles, loc=4) # Lower right
    plt.title('Average Rewards')
    plt.show()

    # Plot the average number of best choices over time    
    handles = []
    for (correctness, s) in zip(choice_correctness, strategies):
        h, = plt.plot(correctness, label=s.describe())
        handles.append(h)
    plt.legend(handles=handles, loc=4) # Lower right
    plt.title('Average Correctness of Choice')
    plt.show()
     

    
         
            
