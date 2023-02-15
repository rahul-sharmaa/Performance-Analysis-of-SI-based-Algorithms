from ba.bat_algorithm import BatAlgorithm
from helpers.data_init import data_init, DataLocation, DataAttributes
from helpers.data_queries import antecedent_consequent_mob_arm
from helpers.helper import translate_rule

if __name__ == '__main__':
    location = DataLocation.BASKETBALL.value
    attributes = DataAttributes.BASKETBALL.value

    bat = BatAlgorithm(data=data_init(location, attributes),
                       population_size=50,
                       iterations=40,
                       pareto_points=5,
                       alpha=0.4, beta=0.3, gamma=0.2, delta=0.1,
                       min_support=0.2, min_confidence=0.5)

    rules = bat.multi_objective_bat_algorithm()

    for rule in rules:
        translated_rule = translate_rule(rule.rule, attributes, antecedent_consequent_mob_arm)
        print("Rule: " + translated_rule + "Objectives: " + str(rule) + "\n")
