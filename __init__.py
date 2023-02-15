import time

from cso.cuckoo_search import CuckooSearchOptimization
from helpers.data_init import data_init, DataLocation, DataAttributes
from helpers.data_queries import antecedent_consequent_mocanar
from helpers.helper import translate_rule

if __name__ == '__main__':
    d = data_init(DataLocation.BODY_FAT.value, DataAttributes.BODY_FAT.value)

    mocanar = CuckooSearchOptimization(data=d,
                                       population_size=500,
                                       pa=0.3,
                                       pmut=0.1,
                                       num_of_tourn=15,
                                       max_generation=5,
                                       num_of_increment=1,
                                       num_of_rnd_cuckoo=1,
                                       w1=0.2, w2=0.5, w3=0.3,
                                       min_support=0,
                                       min_confidence=0)

    start = time.time()

    rules = mocanar.multi_objective_cuckoo_search_algorithm()
    o, c, s, i = 0, 0, 0, 0

    for r in rules:
        print(r.__str__())
        print(translate_rule(r.rule, DataAttributes.BODY_FAT.value, antecedent_consequent_mocanar))
        o += r.comp
        c += r.conf
        s += r.supp
        i += r.inter
    l = len(rules)
    print("means comp, conf, supp, inter", o / l, c / l, s / l, i / l)

    end = time.time()
    print("TIME", end-start)
    print("RULES", l)
