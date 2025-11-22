# Josh Zatz | UIC RAM Lab
#
# This script creates a generalized trust model to impliment into games.
# It uses POMDP for state predictions, Bayesian inference for state updates.
# Classes store the initial + current state, predict future state, how state changes, what is observed, updates state belief, applies risk to state, and tracks history.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as betaDist
from scipy.special import digamma, gammaln

#-----------------------------------------------------------#

#Store current state
class TrustState:
    def __init__(self, alpha = 2.0, beta = 2.0):
        #distribution of trust (beta continuous random variable on interval [0,1])
        self.alphaVar = alpha
        self.betaVar = beta
        self.mean = self.getMean()
        self.variance = self.getVariance()

    # mean value of the trust state
    def getMean(self):
        return self.alphaVar / (self.alphaVar + self.betaVar)

    # variance of the trust state
    def getVariance(self):
        return (self.alphaVar * self.betaVar) / (((self.alphaVar + self.betaVar) ** 2) * (self.alphaVar + self.betaVar + 1))

    # alpha and beta parameters 
    def getParameters(self):
        return self.alphaVar, self.betaVar
  
    # Update parameters bayesianInference class
    def updateParameters(self, newAlpha, newBeta):
        self.alphaVar = newAlpha
        self.betaVar = newBeta
        self.mean = self.getMean()
        self.variance = self.getVariance()

    # Reset parameters (if needed)
    def resetParameters(self, alpha=2.0, beta=2.0):
        self.alphaVar = alpha
        self.betaVar = beta
        self.mean = self.getMean()
        self.variance = self.getVariance()

#-----------------------------------------------------------#

#Model what the robot observes (emissions)
class ObservationModel:
    def __init__(self, observationType = 'binary', noiseStdDev = 0.1, interventionParam=None, acceptanceParam=None):
        self.observationType = observationType #in this case, intervention (1 if i intervene, 0 if i accept robot action)
        self.acceptanceParam = acceptanceParam
        self.interventionParam = interventionParam or { 'lowTrustThreshold' : 0.3, 'highTrustThreshold' : 0.7, 'urgencyMultipler': 1.5}
    
    # Calculate the probability of human intervention based on trust and action
    def getInterventionProbability(self, trustLevel, action):
        midpoint = (self.interventionParam['lowTrustThreshold'] + self.interventionParam['highTrustThreshold']) / 2   #midpoint of trust thresholds (intervention prob=50%)

        steepness = 10  #sharpness of sigmoid transition
        baseProb = 1 / (1 + np.exp(steepness * (trustLevel - midpoint))) #sigmoid function (maps trust to intervention probability)

        actionModifier =  {'take_control': 1.2, 'suggest_only': 1.5, 'urgent': 0.5, 'assist':0.9} #lower intervention probability = human is less likely to intervene (urgent the lowest)
        modifier = actionModifier.get(action, 1.0) #associate an action to one of the 4 categories, otherwise, take control (1.0)
        return np.clip(baseProb * modifier, 0.0, 1.0) #clip probability [0,1]
    
    # Calculate the likelihood of an observation given a trust level and action
    def getObservationLikelihood(self, observation, trustLevel, action):
        interventionProb = self.getInterventionProbability(trustLevel, action) #get intervention probability for a SPECIFIC trust AND action

        if observation == 1: #if human intervenes
            return interventionProb #how likely was this intervention?
        else:   
            return 1 - interventionProb #how likely was robot action accepted?
    
    # Generate a sample observation based on a trust level and action
    def sampleObservation(self, trustLevel, action):
        interventionProb = self.getInterventionProbability(trustLevel, action) #get intervention probability for a SPECIFIC trust AND action
        return 1 if np.random.random() < interventionProb else 0 #get random number [0,1] and compare to intervention probability. 1 if human intervened, 0 otherwise.

    # Calculate the expected observation for a given trust level and action (wrapper function)
    def getExpectedObservation(self, trustLevel, action):
        return self.getInterventionProbability(trustLevel, action) #expected value of observation

    # Calculate the probability of a human accepting a robot's suggestion
    def getAcceptanceProbability(self, trustLevel, action):
        return 1 - self.getInterventionProbability(trustLevel, action) #associate intervention probability with acceptance probability
    
#-----------------------------------------------------------#

#Model how trust changes (transitions)
class TransitionModel:
    def __init__(self, learningRate = 2.0, isTimeVarying = False, decayRate = 0.0, contextWeight = None):
        self.learningRate = learningRate
        self.isTimeVarying = isTimeVarying
        self.decayRate = decayRate
        self.timeStep = 0 #Change later for time window
        self.contextWeight = contextWeight or {
            'take_control': 1.2,
            'collision': 3.5,
            'player_manual_action': 1.0
        }
    # Update trust state based on an action and its observed performance
    def transition(self, trustState, action, performance, timeStep, speed=None, ttc=None):
        alpha, beta = trustState.getParameters() #get beta distribution params

        if speed is not None and ttc is not None: #speed and time to collision contexts)
            speedRisk = min(speed / 100 , 1.0) #normalize speed risk (100 is placeholder for max speed right now)
            ttcRisk = max(0.0, 1.0 - (ttc / 10.0)) #normalize ttc risk (low ttc = high risk)

            combinedRisk = (speedRisk + ttcRisk) / 2.0 #average risks of both contexts
            contextMultiplier = 0.5 + (combinedRisk * 2.5) #map risk to learning weight [0.5,3.0] low and high risk respectively
        else:
            contextMultiplier = self.contextWeight.get(action,1.0) #fincase the context weights arent known
        effectiveLearningRate = self.learningRate * contextMultiplier #effective state learning rate
        
        if performance == 1: #success (perform conjugate updates on parameters)
            alpha += effectiveLearningRate 
        else:
            beta += effectiveLearningRate

        if self.isTimeVarying and self.decayRate > 0: # apply time decay if enabled
            alpha, beta = self.applyDecay(alpha, beta)
        
        trustState.updateParameters(alpha,beta) #update trust state
        self.timeStep = timeStep #update timesteps

        return trustState 

    # Predict the next trust state given an action and its success probability
    def predictState(self, trustState, action, successProbability, speed = None, ttc = None):
        alpha, beta = trustState.getParameters() #get beta distribution params
        if speed is not None and ttc is not None:
            speedRisk = min(speed / 100.0, 1.0)
            ttcRisk = max(0.0, 1.0 - (ttc / 10.0))
            combinedRisk = (speedRisk + ttcRisk) / 2.0
            contextMultiplier = 0.5 + (combinedRisk * 2.5)
        else:
            contextMultiplier = self.contextWeight.get(action, 1.0)
        effectiveLearningRate = self.learningRate * contextMultiplier

        predictedAlpha = alpha + (successProbability * effectiveLearningRate) #predict alpha param
        predictedBeta = beta + ((1-successProbability) * effectiveLearningRate) #predict beta param
                
        if self.isTimeVarying and self.decayRate > 0: #if time decay is enabled
            predictedAlpha, predictedBeta = self.applyDecay(predictedAlpha, predictedBeta) 
        
        predictedState = TrustState(alpha=predictedAlpha, beta=predictedBeta) #return new trust state prediction (keep orginal though! ->Logging memory)
        
        return predictedState

    # Reset the internal timestep counter
    def resetTimestep(self):
        self.timeStep = 0

    # Get a context-specific weight based on task difficulty
    def getWeight(self, difficulty):
        return self.contextWeight.get(difficulty,1.0) #retrieve the contexts, otherwise default to 1.0

    # Apply a decay factor to the trust parameters over time (models regression towards neutral trust, avoids trust sticking to over or under trust!)
    def applyDecay(self, alpha, beta):
        #these params (alpha and beta) of 2.0 are for a neutral trust state (50%)
        neutralAlpha = 2.0
        neutralBeta = 2.0 

        decayedAlpha = alpha + self.decayRate * (neutralAlpha - alpha)
        decayedBeta = beta + self.decayRate * (neutralBeta - beta)
        return decayedAlpha, decayedBeta

#-----------------------------------------------------------#

#Updates belief
class BayesianInference:
    def __init__(self, tolerance = 1e-5, maxIterations=100, method='conjugate', discreteBins=50):
        self.tolerance = tolerance #set tolerance for convergence in computation 
        self.maxIterations = maxIterations 
        self.method = method #bayesian inference method
        self.discreteBins = discreteBins #specifically for grid method (not conjugate)

    # Update the belief distribution given new observations and models
    def updateBelief(self, trustState, observations, observationModel, transitionModel, action):
        if not isinstance(observations, list): #check if current observation is in the list of observations
            observations = [observations] #incase its not in list, observation list is still useable. observation is 0 or 1.
        
        for observation in observations:
            if self.method == 'conjugate':
                self.performConjugateUpdate(trustState, observation, observationModel, action) #updates belief
            elif self.method == 'grid':
                self.performGridUpdate(trustState, observation, observationModel, action) #update belief
            else:
                raise ValueError
        return trustState #belief update depending on method
    
    # Perform a belief update using the conjugate prior (Beta-Bernoulli)
    def performConjugateUpdate(self, trustState, observation, observationModel, action):
        alpha,beta = trustState.getParameters() #get current alpha beta params

        learningRate = observationModel.learningRate if hasattr(observationModel, 'learnRate') else 1.0 #default learning rate for conjugate update
        if observation == 0: #check if human accepted (no intervention)
            alpha += learningRate * 0.5 #if accepted, add 0.1 to existing alpha (tune)         [Trust goes up slightly due to acceptance]
        else:
            beta += learningRate * 2.0 #if intervention, add 0.5 to existing beta (tune)       [Trust goes down slightly due to intervention]
        trustState.updateParameters(alpha,beta) #updates alpha beta params to model, recalculates trust state.

        return trustState 

    # Perform a belief update using grid-based approximation
    def performGridUpdate(self, trustState, observation, observationModel, action):
        alpha,beta = trustState.getParameters() #get current alpha beta params
        trustGrid = np.linspace(0.01,0.99, self.discreteBins) #create evenly spaced numbers for grid
        prior = betaDist.pdf(trustGrid,alpha,beta) #pdf of beta distribution. Gives array of probabilities for each grid point.
        likelihood = np.array([observationModel.getObservationLikelihood(observation, trust, action) for trust in trustGrid]) #loop goes through each trust value in grid, calculates how likely observation was, collects all results into list -> then numpy array.

        posterior = prior * likelihood #Bayes rule (posterior * likelihood)
        posterior = posterior / np.sum(posterior) #Normalized for all posterior beliefs (combination of all prior beliefes with new evidence)

        posteriorMean = np.sum(trustGrid * posterior) #weighted average of posterior distribution
        posteriorVar = np.sum((trustGrid - posteriorMean)**2 * posterior)  #variance of posterior distibution

        if posteriorVar > 0: #prevent division by zero
            #alpha and beta are now calculated with method of moments formula
            newAlpha = posteriorMean * (posteriorMean * (1 - posteriorMean) / posteriorVar - 1) 
            newBeta = (1 - posteriorMean) * (posteriorMean * (1 - posteriorMean) / posteriorVar - 1)
            newAlpha = max(0.1, newAlpha) #returns either 0.1 or new alpha, whichever is larger
            newBeta = max(0.1, newBeta) #returns either 0.1 or new beta, whichever is larger
        else: #if new values cannot be calculated (variance is 0 or less -> no change from last action)
            newAlpha = alpha 
            newBeta = beta
        
        trustState.updateParameters(newAlpha,newBeta) #belief of trust state with new params.
        return trustState

    # Learn model parameters from historical performance data
    def learnParameters(self, performanceTracker, trustStateHistory=None):
        history = performanceTracker.getHistory(window = None, actionFilter = None) #get list of performance history
        if len(history) == 0: #check if there is any history to begin with
            return {'learningRate': 2.0, 'message' : 'No data available'} #dictionary of history
        
        successRate = performanceTracker.getSuccessRate(window = None, actionFilter = None) #empirical success rate 0 to 1
        total = performanceTracker.getTotalObservations() #retrieves total number of successes

        if total > 10: #need sufficent data to learn from history (tune)
            recentHistory = performanceTracker.getRecentHistory(min(20, total))     # gets most amount of recent records

            successes = sum(1 for record in recentHistory if record['success']) #loop through each record and bind 1 to success, 0 otherwise
            variance = (successes / len(recentHistory)) * (1 - successes / len(recentHistory)) #variance for random variables (VarX = p(1-x) Bernouilli random variable probability)
            
            learnedRate = 0.5 + (1.5 * (1 - variance))     # Map variance to learning rate (0.5 to 2.0)         [low variance = consistent performance]

        else:
            learnedRate = 1.0  #otherwise default learning rate
        
        return {
            'learningRate': learnedRate,
            'successRate': successRate,
            'sampleSize': total,
            'message': 'Parameters learned from data'
        } #returns dictionary of learned parameters and data
 
    # Calculate the KL-Divergence between two beta distributions (how different two beliefs are)
    def calculateKLDivergence(self, alpha1, beta1, alpha2, beta2):
        #divergence formula for beta distributions (normalization of first and second distribution, contribution to alpha and beta difference, cross term respectively)
        kl = (gammaln(alpha1 + beta1) - gammaln(alpha1) - gammaln(beta1) - gammaln(alpha2 + beta2) + gammaln(alpha2) + gammaln(beta2) + (alpha1 - alpha2) * digamma(alpha1) + (beta1 - beta2) * digamma(beta1) + (alpha2 - alpha1 + beta2 -beta1) * digamma(alpha1 + beta1))
        return kl
    
#-----------------------------------------------------------#

#Track history of variables (trust state, observation, input, etc)
class PerformanceTracker:
    def __init__(self, windowSize, trackContext=True):
        self.history = []  # List of dictionaries, each recording one event
        self.windowSize = windowSize
        self.trackContext = trackContext
        self.successCount = 0
        self.failureCount = 0
        self.totalCount = 0
    
    #record a new performance
    def recordPerformance(self, action, success, timestep, context=None):
        # Create record dictionary
        record = {
            'action': action,
            'success': bool(success),  # Convert to boolean
            'timestep': timestep
        }
        
        # Add context if tracking is enabled and context provided
        if self.trackContext and context is not None:
            record['context'] = context
        
        # Add to history
        self.history.append(record)
        
        # Update counters
        if success:
            self.successCount += 1
        else:
            self.failureCount += 1
        self.totalCount += 1
        
        # Maintain window size if specified
        if self.windowSize is not None and len(self.history) > self.windowSize:
            self.removeOldest()

    #Calculate success rate for filtered records
    def getSuccessRate(self, window=None, actionFilter=None, contextFilter=None):
        records = self.filterRecords(window, actionFilter, contextFilter)
        
        if len(records) == 0:
            return 0.0
        
        successes = sum(1 for record in records if record['success'])
        return successes / len(records)

    #Retrieve history records 
    def getHistory(self, window=None, actionFilter=None):
        return self.filterRecords(window, actionFilter, contextFilter=None)
    
    #Get n most recent records
    def getRecentHistory(self, n):
        if n >= len(self.history):
            return self.history.copy()
        else:
            return self.history[-n:].copy()

    #Total number of recorded observations
    def getTotalObservations(self):
        return self.totalCount

    #summary of performance metrics
    def getSummary(self):
        summary = {
            'totalObservations': self.totalCount,
            'successes': self.successCount,
            'failures': self.failureCount,
            'successRate': self.successCount / self.totalCount if self.totalCount > 0 else 0.0,
            'historyLength': len(self.history)
        }
        
        # Add action-specific summaries if we have data
        if len(self.history) > 0:
            actions = set(record['action'] for record in self.history)
            summary['uniqueActions'] = len(actions)
            summary['actionBreakdown'] = {}
            
            for action in actions:
                actionRecords = [r for r in self.history if r['action'] == action]
                actionSuccesses = sum(1 for r in actionRecords if r['success'])
                summary['actionBreakdown'][action] = {
                    'count': len(actionRecords),
                    'successRate': actionSuccesses / len(actionRecords)
                }
        
        return summary

    #trends over time windows
    def getRecentTrend(self, window=10):
        recent = self.getRecentHistory(window)
        
        if len(recent) == 0:
            return {'trend': 'no_data', 'direction': 0}
        
        # Split into first and second half
        midpoint = len(recent) // 2
        firstHalf = recent[:midpoint] if midpoint > 0 else []
        secondHalf = recent[midpoint:]
        
        # Calculate success rates
        firstRate = sum(1 for r in firstHalf if r['success']) / len(firstHalf) if len(firstHalf) > 0 else 0.0
        secondRate = sum(1 for r in secondHalf if r['success']) / len(secondHalf) if len(secondHalf) > 0 else 0.0
        
        # Determine trend
        difference = secondRate - firstRate
        
        if abs(difference) < 0.1:
            trend = 'stable'
        elif difference > 0:
            trend = 'improving'
        else:
            trend = 'declining'
        
        return {
            'trend': trend,
            'direction': difference,
            'firstHalfRate': firstRate,
            'secondHalfRate': secondRate,
            'windowSize': len(recent)
        }

    #clear history and reset counters if needed
    def reset(self):
        self.history = []
        self.successCount = 0
        self.failureCount = 0
        self.totalCount = 0

    #remove oldest entry
    def removeOldest(self):
        if len(self.history) > 0:
            oldest = self.history.pop(0)  # Remove first element
            
            # Update counters
            if oldest['success']:
                self.successCount -= 1
            else:
                self.failureCount -= 1
            self.totalCount -= 1

    #filter history based on criteria (only include specific actions, or contexts)
    def filterRecords(self, window=None, actionFilter=None, contextFilter=None):
        # Start with appropriate window
        if window is None:
            records = self.history.copy()
        else:
            records = self.getRecentHistory(window)
        
        # Filter by action
        if actionFilter is not None:
            records = [r for r in records if r['action'] == actionFilter]
        
        # Filter by context
        if contextFilter is not None and self.trackContext:
            records = [r for r in records if 'context' in r and contextFilter(r['context'])]
        
        return records

#-----------------------------------------------------------#

#Define reward (trust-aware objective function)
class RewardModel:
    def __init__(self, taskSuccessWeight, trustWeight, trustTarget, interventionPenalty, decayPenalty, trustThreshold):
        self.taskSuccessWeight = taskSuccessWeight
        self.trustWeight = trustWeight
        self.trustTarget = trustTarget
        self.interventionPenalty = interventionPenalty
        self.decayPenalty = decayPenalty
        self.trustThreshold = trustThreshold

    # Compute the total reward for a given outcome
    def computeReward(self, trustLevel, wasTaskSuccessful, wasHumanIntervention, previousTrustLevel, context):
        taskReward = self.taskSuccessWeight if wasTaskSuccessful else 0.0 #task reward

        trustDeviation = abs(trustLevel - self.trustTarget) #penalize deviation from target trust
        trustReward = self.trustWeight * (1.0 - trustDeviation) #update task reward
        interventionReward = self.interventionPenalty if wasHumanIntervention else 0.0 #penalize if human intervenes

        trustChange = trustLevel - previousTrustLevel #penalize if trust does not increase
        if trustChange < 0: 
            decayReward = self.decayPenalty + abs(trustChange)
        else:
            decayReward = 0.0

        thresholdPenalty = 0.0 #for extreme threshold violations
        if trustLevel < self.trustThreshold:
            thresholdPenalty = -1.0 #large penalty for dropping below minimum trust level
        
        totalReward = taskReward + trustReward + interventionReward + decayReward + thresholdPenalty #total objective function
        return totalReward

    # Calculate the expected reward for a future state (same as computeReward but now a probability of future state)
    def getExpectedReward(self, trustLevel, successProbability, interventionProbability, previousTrustLevel):
        expectedTaskReward = self.taskSuccessWeight * successProbability #expected value of task reward
        trustDeviation = abs(trustLevel - self.trustTarget) # deviation from target trust
        trustReward = self.trustWeight * (1.0 - trustDeviation) # trust component is still deterministc (given expected taskReward and deviation)

        expectedInterventionReward = self.interventionPenalty * interventionProbability #expected intervention penalty
        trustChange = trustLevel - previousTrustLevel #difference in trust level (prior to now)
        if trustChange < 0: #if there no change
            decayReward = self.decayPenalty * abs(trustChange) #ensure trust does not stick
        else: #if there is a change
            decayReward = 0.0 #value is not sticking so no decay penalty
        
        thresholdPenalty = 0.0  #set a threshold on the penalty (dont lose 100% for one mistake)
        if trustLevel < self.trustThreshold: #current trust level must be within range of threshold otherwise its invalid
            thresholdPenalty = -1.0 #just penalize -1
        
        expectedReward = (expectedTaskReward + trustReward + 
                         expectedInterventionReward + decayReward + thresholdPenalty) #sum of expected reward
        
        return expectedReward
        
    # Evaluate the outcome of a specific action (wrapper for computeReward, my own convenience)
    def evaluateOutcome(self, action, outcome):
        return self.computeReward(trustLevel=outcome.get('trustLevel',0.5),wasTaskSuccessful=outcome.get('success',False),wasHumanIntervention=outcome.get('intervention',False),previousTrustLevel=outcome.get('previousTrust',0.5),context=outcome.get('context',None))

    # Compare the expected rewards of different potential actions (compare many robot inputs to expected rewards. Pick best action)
    def compareActions(self, actionOutcomes):
        actionRewards = [] #create empty list of actions to be done
        for action, outcome in actionOutcomes.items(): #for actions that go in that list...
            reward = self.evaluateOutcome(action,outcome) #calculate reward for action and observation/outcome pair specifically
            actionRewards.append((action,reward)) #append to list
        actionRewards.sort(key=lambda x: x[1], reverse = True) #sort function for sorting rewards (reverse so that highest reward is first)

        return actionRewards #best action

    # Set new weights for the reward components 
    def setWeights(self, taskWeight, trustWeight, interventionPenalty):
        if taskWeight is not None: #check if the task weight is provided
            self.taskSuccessWeight = taskWeight #updates weight if provided prior
        if trustWeight is not None: #check if the trust weight is provided
            self.trustWeight = trustWeight #updates weight if provided prior
        if interventionPenalty is not None: #check if the interventionpenalty is provided prior
            self.interventionPenalty = interventionPenalty #updates interventionpenalty if provided prior

    # Return the current reward model parameters in a dictionary
    def getParameters(self):
        return {'taskSuccessWeight': self.taskSuccessWeight, 'trustWeight': self.trustWeight, 'trustTarget': self.trustTarget, 'interventionPenalty': self.interventionPenalty, 'decayPenalty': self.decayPenalty, 'trustThreshold': self.trustThreshold }
    
#-----------------------------------------------------------#

#Predict future state
class TrustPOMDP:
    def __init__(self, transitionModel, observationModel, rewardModel, discountFactor = 0.95, performanceTracker=None):
        self.transitionModel = transitionModel
        self.observationModel = observationModel
        self.rewardModel = rewardModel
        self.discountFactor = discountFactor
        self.performanceTracker = performanceTracker

    # Predict the next belief state given the current belief and an action
    def predictBelief(self, currentBelief, action, context):
        successProb = self.estimateSuccess(action,context)
        predictedBelief = self.transitionModel.predictState(currentBelief,action,successProb)
        return predictedBelief

    # Update the current belief state with a new observation
    def updateBelief(self, priorBelief, action, observation, bayesianInference):
        updatedBelief = bayesianInference.updateBelief(priorBelief,observation,self.observationModel,self.transitionModel,action)
        return updatedBelief

    # Run the forward algorithm to predict beliefs over a sequence of actions
    def runForwardAlgorithm(self, initialBelief, actionSequence, horizon, context):
        beliefs = [initialBelief]
        currentBelief = initialBelief

        for i in range(min(horizon, len(actionSequence))):
            action = actionSequence[i]
            nextBelief = self.predictBelief(currentBelief,action,context)
            beliefs.append(nextBelief)
            currentBelief = nextBelief
        return beliefs

    # Decide the optimal action to take from the current belief state
    def decideAction(self, currentBelief, availableActions, horizon, context):
        bestAction = None
        bestValue = float('-inf')

        for action in availableActions:
            value = self.computeValue(currentBelief,action,horizon,context)

            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction

    # Compute the value (expected future reward) of a belief-action pair
    def computeValue(self, belief, action, horizon=1,context=None):
        successProb = self.estimateSuccess(action,context)
        nextBelief = self.transitionModel.predictState(belief,action,successProb)

        interventionProb = self.observationModel.getInterventionProbability(belief.mean, action)
        immediateReward = self.rewardModel.getExpectedReward(trustLevel=nextBelief.mean, successProbability=successProb, interventionProbability=interventionProb,previousTrustLevel=belief.mean)

        if horizon <= 1:
            return immediateReward
        return immediateReward

    # Estimate the probability of success for a given action and context
    def estimateSuccess(self, action, context=None, performanceTracker=None):
        # Use historical data if available
        if performanceTracker is not None:
            
            # Get success rate for this action
            actionSuccessRate = self.performanceTracker.getSuccessRate(
                window=None,
                actionFilter=action,
                contextFilter=None
            )
            
            # Check if we have data for this action
            actionHistory = self.performanceTracker.getHistory(window=None, actionFilter=action)
            
            if len(actionHistory) >= 3:  # Need at least 3 samples
                baseRate = actionSuccessRate
            else:
                # Use overall success rate
                baseRate = self.performanceTracker.getSuccessRate(window=None, actionFilter=None)
                if baseRate == 0.0:  # No data at all
                    baseRate = 0.75  # Default
        else:
            # Fallback defaults
            defaultRates = {
                'take_control': 0.85,
                'suggest_only': 0.70,
                'assist': 0.80,
                'urgent': 0.5
            }
            baseRate = defaultRates.get(action, 0.75)
        
        # Adjust based on context
        if context is not None:
            ttc = context.get('ttc', 10.0)
            speed = context.get('speed', 0.0)
            
            # TTC adjustment
            if ttc < 1.5:
                baseRate *= 0.7
            elif ttc < 3.0:
                baseRate *= 0.85
            elif ttc < 5.0:
                baseRate *= 0.95
            
            # Speed adjustment
            if speed > 90:
                baseRate *= 0.8
            elif speed > 75:
                baseRate *= 0.9
            
            # Context-specific historical rate (if enough data)
            if performanceTracker is not None:
                def similarContext(ctx):
                    similarTTC = abs(ctx.get('ttc', 10) - ttc) < 2.0
                    similarSpeed = abs(ctx.get('speed', 0) - speed) < 15
                    return similarTTC and similarSpeed
                
                contextualRate = self.performanceTracker.getSuccessRate(
                    window=50,
                    actionFilter=action,
                    contextFilter=similarContext
                )
                
                contextualHistory = self.performanceTracker.filterRecords(
                    window=50,
                    actionFilter=action,
                    contextFilter=similarContext
                )
                
                if len(contextualHistory) >= 5:
                    # Blend contextual and base rate
                    baseRate = 0.7 * contextualRate + 0.3 * baseRate
        
        return np.clip(baseRate, 0.0, 1.0)

    # Calculate the expected reward for a belief-action pair given an expected observation
    def getExpectedReward(self, belief, action, expectedObservation):
        successProb = self.estimateSuccess(action, None)
        nextBelief = self.transitionModel.predictState(belief, action, successProb)
        interventionProb = self.observationModel.getInterventionProbability(
            belief.mean, action
        )
        
        return self.rewardModel.getExpectedReward(
            trustLevel=nextBelief.mean,
            successProbability=successProb,
            interventionProbability=interventionProb,
            previousTrustLevel=belief.mean
        )
    
#-----------------------------------------------------------#

#Puts model all together
class TrustModel:
    def __init__(self, trustPrior, observationModelParams, transitionModelParams, inferenceParams, rewardParams, performanceTrackerParams):
        
        #Get inital state and models info
        self.trustState = TrustState(alpha=trustPrior.get('alpha', 2.0), beta=trustPrior.get('beta',2.0))
        self.transitionModel = TransitionModel(**transitionModelParams)
        self.observationModel = ObservationModel(**observationModelParams)
        self.rewardModel = RewardModel(**rewardParams)     
        self.bayesianInference = BayesianInference(**(inferenceParams or {}))
        self.performanceTracker = PerformanceTracker(**performanceTrackerParams)

        #Get POMDP Model
        self.pomdp = TrustPOMDP(self.transitionModel, self.observationModel, self.rewardModel, performanceTracker=self.performanceTracker)

        #Get history
        self.observationHistory = []
        self.actionHistory = []
        self.timeStep = 0

    # Update trust based on a complete action-outcome-response cycle
    def updateTrust(self, robotAction, robotPerformance, humanResponse, context=None):
        previousTrust = self.trustState.mean
        
        # Extract speed and ttc from context if available
        speed = context.get('speed') if context else None
        ttc = context.get('ttc') if context else None
        
        # Update trust based on performance
        self.transitionModel.transition(
            self.trustState,
            robotAction,
            performance=int(robotPerformance),
            timeStep=self.timeStep,
            speed=speed,
            ttc=ttc
        )
        
        # Update trust based on human response
        observation = 1 if humanResponse else 0
        self.bayesianInference.updateBelief(
            self.trustState,
            observation,
            self.observationModel,
            self.transitionModel,
            robotAction
        )
        
        # Record performance
        self.performanceTracker.recordPerformance(
            action=robotAction,
            success=robotPerformance,
            timestep=self.timeStep,
            context = context
        )
        
        # Update histories
        self.observationHistory.append(observation)
        self.actionHistory.append(robotAction)
        self.timeStep += 1
        
        return {
            'currentTrust': self.trustState.mean,
            'previousTrust': previousTrust,
            'trustChange': self.trustState.mean - previousTrust,
            'alpha': self.trustState.alphaVar,
            'beta': self.trustState.betaVar
        }

    def predictBehavior(self, robotAction, context=None):
        interventionProb = self.observationModel.getInterventionProbability(
            self.trustState.mean,
            robotAction
        )
        
        successProb = self.pomdp.estimateSuccess(robotAction, context)
        predictedTrust = self.transitionModel.predictState(
            self.trustState,
            robotAction,
            successProb
        )
        
        return {
            'interventionProbability': interventionProb,
            'acceptanceProbability': 1.0 - interventionProb,
            'predictedTrust': predictedTrust.mean,
            'successProbability': successProb
        }
    
    #Decision Maker!
    def recommendAction(self, availableActions, planningHorizon=1, context=None):
        actionValues = {}
        
        for action in availableActions:
            value = self.pomdp.computeValue(
                self.trustState,
                action,
                planningHorizon,
                context
            )
            actionValues[action] = value
        
        bestAction = max(actionValues, key=actionValues.get)
        predictions = self.predictBehavior(bestAction, context)
        
        return {
            'recommendedAction': bestAction,
            'expectedValue': actionValues[bestAction],
            'allActionValues': actionValues,
            'predictions': predictions
        }

    def getTrust(self):
        return self.trustState.mean

    def getTrustDistribution(self):
        return {
            'alpha': self.trustState.alphaVar,
            'beta': self.trustState.betaVar,
            'mean': self.trustState.mean,
            'variance': self.trustState.variance
        }

    def getPerformance(self):
        return self.performanceTracker.getSummary()

    def reset(self, newPrior=None):
        if newPrior is None:
            newPrior = {'alpha': 2.0, 'beta': 2.0}
        
        self.trustState.resetParameters(
            alpha=newPrior.get('alpha', 2.0),
            beta=newPrior.get('beta', 2.0)
        )
        
        self.performanceTracker.reset()
        self.transitionModel.resetTimestep()
        
        self.observationHistory = []
        self.actionHistory = []
        self.timeStep = 0