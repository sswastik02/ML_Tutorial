import tensorflow as tf
import tensorflow_probability as tfp

# -------------------------------------------- Markov Weather Model -------------------------------------------

# Weather Model  :
# 1. Cold days are encoded 0 while hot days are encoded 1
# 2. First day in our sequence has has 80% chance of being cold
# 3. A cold day has a 30% chance of being followed by a hot day
# 4. A hot day has a 20 % chance of being followed by a cold day
# 5. On each day the temperature is normally distributed with mean and standard deviation
#       i. 0 and 5 on a cold day
#       ii.15 and 10 on a hot day

# 15 mean and 10 s.d. means range of tempereature is 5 - 25

tfpd = tfp.distributions
initial_distribution = tfpd.Categorical(probs=[0.8,0.2]) # from point 2
transition_distribution = tfpd.Categorical(probs=[[0.7,0.3],[0.2,0.8]]) # from point 3
# cold to cold = 0.7; cold to hot = 0.3
# hot to cold = 0.2; hot to hot = 0.8

observation_distribution = tfpd.Normal(loc = [0.,15.],scale=[5.,10.]) # point 5

model = tfpd.HiddenMarkovModel(
    initial_distribution = initial_distribution,
    transition_distribution = transition_distribution,
    observation_distribution = observation_distribution,
    num_steps = 7
)
# num_steps is how many times you want to predict, in this case upto 7 days

mean = model.mean() # It is a tensor
# A session allocates resources to compute tensors
with tf.compat.v1.Session() as session:
    print(mean.numpy())

# Above lines are used to get predictions