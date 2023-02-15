import time

from ba.bat_algorithm import BatAlgorithm
from helpers.data_init import data_init


def translate_rule(rule, attributes, data):
    antec_txt = "If "
    conseq_txt = " Then "
    for c, r in enumerate(rule):
        if c > 0 and c < rule[0] and r != 0:
            antec_txt += attributes[c - 1] + " is "
            interval = data[attributes[c - 1]].query("id == " + str(r))["interval"].describe()[2]
            antec_txt += str(interval) + ", "
        elif c > 0 and c >= rule[0] and r != 0:
            conseq_txt += attributes[c - 1] + " is "
            interval = data[attributes[c - 1]].query("id == " + str(r))["interval"].describe()[2]
            conseq_txt += str(interval) + ", "
    return antec_txt + conseq_txt


if __name__ == '__main__':
    basketball_attributes = ["assists_per_minute", "height", "time_played", "age", "points_per_minute"]
    basketball_data_location = "../data/BK.dat"
    basketball_data = data_init(basketball_data_location, basketball_attributes)

    quake_attributes = ["focal_depth", "latitude", "longitude", "richter"]
    quake_data_location = "../data/QU.dat"
    quake_data = data_init(quake_data_location, quake_attributes)

    bat = BatAlgorithm(data=basketball_data,
                       population_size=50,
                       iterations=5,
                       pareto_points=5,
                       alpha=0.4, beta=0.3, gamma=0.2, delta=0.1,
                       min_support=0.2, min_confidence=0.5)
    o, c, s, i, j = 0, 0, 0, 0, 0

    start = time.time()

    rules = bat.mob_arm()

    end = time.time()

    for r in rules:
        print(r.rule, translate_rule(r.rule, bat.attributes, bat.data), r.obj, r.supp, r.conf, r.comp, r.inter, r.rule)
        o += r.obj
        c += r.conf
        s += r.supp
        i += r.inter
        j += r.comp

    r_len = len(rules)
    if r_len == 0:
        r_len = 1
    print("rules", r_len)
    print("means obj, conf, supp, inter, comp", o / r_len, c / r_len, s / r_len, i / r_len, j/r_len)

    print("time", end - start)
    print(bat.evolution_counter)
