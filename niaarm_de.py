from niaarm import NiaARM
from niaarm.dataset import Dataset
from niapy.algorithms.basic import DifferentialEvolution
from niapy.task import Task, OptimizationType
import time

# load and preprocess the dataset from csv
data = Dataset("datasets/Abalone.csv")

# Create a problem:::
# dimension represents the dimension of the problem;
# features represent the list of features, while transactions depicts the list of transactions
problem = NiaARM(data.dimension, data.features, data.transactions, metrics=('support', 'confidence'), logging=True)

# build niapy task
task = Task(problem=problem, max_iters=30, optimization_type=OptimizationType.MAXIMIZATION)

# see full list of available algorithms: https://github.com/NiaOrg/NiaPy/blob/master/Algorithms.md
algo = DifferentialEvolution(population_size=50, differential_weight=0.5, crossover_probability=0.9)

# run algorithm
start = time.time()
best = algo.run(task=task)
print("Program execution --- %s seconds ---" % (time.time() - start))

# sort rules
problem.rules.sort()

# export all rules to csv
problem.rules.to_csv('output.csv')
