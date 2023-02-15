import time

from helpers.data_init import data_init, DataLocation, DataAttributes
from helpers.data_queries import antecedent_consequent_mopar
from helpers.helper import translate_rule
from pso.particle_search import ParticleSearchOptimization

if __name__ == '__main__':
    d = data_init(DataLocation.BODY_FAT.value, DataAttributes.BODY_FAT.value)

    pso = ParticleSearchOptimization(data=d,
                                     population_size=100,
                                     max_iterations=150,
                                     external_repository_size=100,
                                     c1=2, c2=2,
                                     inertia_weight=0.63,
                                     velocity_limit=3.83,
                                     x_rank=13.33)

    start = time.time()

    rules = pso.multi_objective_particle_search_optimization_algorithm()

    end = time.time()

    s, c, co, i = 0, 0, 0, 0
    for r in rules:
        print(r)
        print(translate_rule(r.rule, DataAttributes.BODY_FAT.value, antecedent_consequent_mopar))
        s += r.supp
        c += r.conf
        co += r.comp
        i += r.inter
    r_len = len(rules)
    print("rules", r_len)
    print("supp, conf, comp, inter", s/r_len, c / r_len, co / r_len, i / r_len)

    print("time", end-start)
