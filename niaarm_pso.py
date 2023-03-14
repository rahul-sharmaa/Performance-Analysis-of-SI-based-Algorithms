from niaarm import NiaARM
from niaarm.dataset import Dataset
from niapy.algorithms.basic import ParticleSwarmAlgorithm
from niapy.task import Task, OptimizationType

# load and preprocess the dataset from csv
data = Dataset("datasets/Abalone.csv")

# Create a problem:::
# dimension represents the dimension of the problem;
# features represent the list of features, while transactions depicts the list of transactions
problem = NiaARM(data.dimension, data.features, data.transactions, metrics=('support', 'confidence'), logging=True)

# build niapy task
task = Task(problem=problem, max_iters=30, optimization_type=OptimizationType.MAXIMIZATION)

# see full list of available algorithms: https://github.com/NiaOrg/NiaPy/blob/master/Algorithms.md
algo = ParticleSwarmAlgorithm(population_size=100, min_velocity=-4.0, max_velocity=4.0)

# run algorithm
best = algo.run(task=task)

# sort rules
problem.rules.sort()

# export all rules to csv
problem.rules.to_csv('output.csv')
