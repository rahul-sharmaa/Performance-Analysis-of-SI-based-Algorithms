from cso.cuckoo_search import CuckooSearchOptimization
from helpers.data_init import data_init, DataLocation, DataAttributes
from helpers.data_queries import antecedent_consequent_mocanar
from helpers.helper import translate_rule

if __name__ == '__main__':
    location = DataLocation.BASKETBALL.value
    attributes = DataAttributes.BASKETBALL.value

    cso = CuckooSearchOptimization(data=data_init(location, attributes),
                                   population_size=50,
                                   pa=0.3,
                                   pmut=0.05,
                                   num_of_tourn=30,
                                   max_generation=200,
                                   num_of_increment=1,
                                   num_of_rnd_cuckoo=1,
                                   w1=0.2, w2=0.5, w3=0.3)

    rules = cso.multi_objective_cuckoo_search_algorithm()

    for rule in rules:
        translated_rule = translate_rule(rule.rule, attributes, antecedent_consequent_mocanar)
        print("Rule: " + translated_rule + "Objectives: " + str(rule) + "\n")
