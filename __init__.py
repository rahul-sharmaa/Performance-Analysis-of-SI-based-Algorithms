import time

from aco.ant_colony import AntColonyOptimization
from helpers.data_init import data_init, DataLocation, DataAttributes
from helpers.data_queries import antecedent_consequent_aco_r
from helpers.helper import translate_rule

if __name__ == '__main__':

    d = data_init(DataLocation.BODY_FAT.value, DataAttributes.BODY_FAT.value)

    aco = AntColonyOptimization(data=d,
                                archive_size=150,
                                ant_colony_size=15,
                                max_iterations=250,
                                alpha1=4, alpha2=4, alpha3=1, alpha4=1, alpha5=1, q=0.1, e=0.85)
    o, c, s, i, co = 0, 0, 0, 0, 0

    start = time.time()

    rules = aco.ant_colony_optimization_for_continuous_domains()

    end = time.time()

    for r in rules:
        print(r.__str__())
        print(translate_rule(r.rule, DataAttributes.BODY_FAT.value, antecedent_consequent_aco_r))
        o += r.obj
        c += r.conf
        s += r.supp
        i += r.inter
        co += r.comp
    r_len = len(rules)
    print("rules", r_len)
    print("means obj, conf, supp, inter, comp", o / r_len, c / r_len, s / r_len, i / r_len, co/r_len)

    print("time", end-start)
