from helpers.data_init import data_init, DataLocation, DataAttributes
from helpers.data_queries import antecedent_consequent_mopar
from helpers.helper import translate_rule
from pso.particle_search import ParticleSearchOptimization

if __name__ == '__main__':
    location = DataLocation.BASKETBALL.value
    attributes = DataAttributes.BASKETBALL.value

    pso = ParticleSearchOptimization(data=data_init(location, attributes),
                                     population_size=50,
                                     max_iterations=200,
                                     external_repository_size=50,
                                     c1=2, c2=2,
                                     inertia_weight=0.63,
                                     velocity_limit=3.83,
                                     x_rank=13.33)

    rules = pso.multi_objective_particle_search_optimization_algorithm()

    for rule in rules:
        translated_rule = translate_rule(rule.rule, attributes, antecedent_consequent_mopar)
        print("Rule: " + translated_rule + "Objectives: " + str(rule) + "\n")
