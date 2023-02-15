from aco.ant_colony import AntColonyOptimization
from helpers.data_init import data_init, DataLocation, DataAttributes
from helpers.data_queries import antecedent_consequent_aco_r
from helpers.helper import translate_rule

if __name__ == '__main__':
    location = DataLocation.BASKETBALL.value
    attributes = DataAttributes.BASKETBALL.value

    aco = AntColonyOptimization(data=data_init(location, attributes),
                                archive_size=50,
                                ant_colony_size=50,
                                max_iterations=200,
                                alpha1=4,
                                alpha2=4,
                                alpha3=1,
                                alpha4=0.001,
                                alpha5=1,
                                q=0.1,
                                e=0.85)

    rules = aco.ant_colony_optimization_for_continuous_domains()

    for rule in rules:
        translated_rule = translate_rule(rule.rule, attributes, antecedent_consequent_aco_r)
        print("Rule: " + translated_rule + "Objectives: " + str(rule) + "\n")
