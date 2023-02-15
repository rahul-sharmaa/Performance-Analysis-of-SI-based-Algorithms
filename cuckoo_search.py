import copy
import random
import numpy as np
import math

from helpers.data_checker import check_locations_with_concrete_values
from helpers.data_init import define_lb_and_ub
from helpers.data_queries import query_records, antecedent_consequent_mocanar, find_antecedent_and_consequent_parts
from helpers.helper import delete_duplicated_rules, find_non_dominated, first_rule_dominates_second_mocanar
from helpers.objective_functions import calculate_objectives


class Cuckoo:

    def __init__(self, rule, support, confidence, comprehensibility, interestingness):
        self.rule = rule
        self.supp = support
        self.conf = confidence
        self.comp = comprehensibility
        self.inter = interestingness

    def __str__(self):
        txt = "Support:{supp:.2f}, Confidence:{conf:.2f}, Comprehensibility:{comp:.2f}, Interestingness:{inter:.2f}"
        return txt.format(supp=self.supp, conf=self.conf, comp=self.comp, inter=self.inter, rule=self.rule)

    def set_objectives(self, support, confidence, comprehensibility, interestingness):
        self.supp = support
        self.conf = confidence
        self.comp = comprehensibility
        self.inter = interestingness


class CuckooSearchOptimization:

    def __init__(self,
                 data,
                 population_size,
                 pa, pmut,
                 num_of_tourn,
                 max_generation,
                 num_of_increment,
                 num_of_rnd_cuckoo,
                 w1, w2, w3,
                 min_support, min_confidence):
        self.population_size = population_size
        self.pa = pa
        self.pmut = pmut
        self.num_of_tourn = num_of_tourn
        self.max_generation = max_generation
        self.num_of_increment = num_of_increment
        self.num_of_rnd_cuckoo = num_of_rnd_cuckoo
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.data = data
        self.attributes = list(data.columns)
        self.min_support = min_support
        self.min_confidence = min_confidence

        self.population = []
        self.cuckoo_eggs = []
        self.num_of_attributes = len(self.attributes)
        self.location_list = [0, 1, 2]
        self.distribution = np.random.dirichlet(np.ones(3), size=1)[0]
        self.lb_list, self.ub_list = define_lb_and_ub(self.attributes, self.data)

        self.intervals = []
        for i in range(len(self.attributes)):
            self.intervals.append((self.ub_list[i] - self.lb_list[i]) * 0.5)

    def multi_objective_cuckoo_search_algorithm(self):
        end_non_dominated = []

        for runs in range(0, self.num_of_increment):
            non_dominated = []
            print("initializing")
            self.population = self.initialize_population()
            self.cuckoo_eggs = []

            for generation in range(0, self.max_generation):
                print("increment " + str(runs))
                print("generation " + str(generation))

                for _ in range(0, self.num_of_rnd_cuckoo):
                    self.population.sort(key=lambda x: x.supp + x.conf + x.comp + x.inter, reverse=True)
                    new_cuckoo = self.get_new_cuckoo_by_levy_flights(source_cuckoo=self.generate_random_cuckoo(),
                                                                     target_cuckoo=self.population[0])
                    self.population[-1] = new_cuckoo
                for cuckoo in self.population:
                    new_egg = self.get_new_cuckoo_by_levy_flights(source_cuckoo=copy.deepcopy(cuckoo),
                                                                  target_cuckoo=self.get_best_cuckoo_with_tournament())
                    self.cuckoo_eggs.append(new_egg)

                self.population = self.do_choosing()

                temp_rules = merge(non_dominated, self.population)
                temp_rules = delete_duplicated_rules(temp_rules)
                non_dominated = copy.deepcopy(find_non_dominated(temp_rules, first_rule_dominates_second_mocanar))

            end_non_dominated.extend(non_dominated)
            self.distribution = np.random.dirichlet(np.ones(3), size=1)[0]
            self.population = []

        end_non_dominated = delete_duplicated_rules(end_non_dominated)
        end_non_dominated = find_non_dominated(end_non_dominated, first_rule_dominates_second_mocanar)
        return end_non_dominated

    def initialize_population(self):
        population = []
        for p in range(self.population_size):
            cuckoo = self.generate_random_cuckoo()
            population.append(cuckoo)
        return population

    def generate_random_cuckoo(self):
        rule = []
        locations = []
        for i in range(0, self.num_of_attributes):
            location = random.choices(self.location_list, self.distribution)[0]
            lower_bound = random.choices(self.data.iloc[:, i])[0]
            upper_bound = random.uniform(lower_bound, self.ub_list[i])

            locations.append(location)
            lower_bound, upper_bound = self.check_attribute_bounds(lower_bound, upper_bound, i)
            rule.append([location, lower_bound, upper_bound])
        r = check_locations_with_concrete_values(rule, locations)
        supp, conf, comp, inter = self.evaluate_objectives(r)
        if supp > 0:
            print(supp, conf, inter, r)
        cuckoo = Cuckoo(r, supp, conf, comp, inter)
        return cuckoo

    def check_condition_and_fix(self, cuckoo):
        locations = []
        for i in range(0, self.num_of_attributes):
            locations.append(cuckoo.rule[i][0])
            if cuckoo.rule[i][1] < self.lb_list[i]:
                cuckoo.rule[i][1] = self.lb_list[i]
            if cuckoo.rule[i][2] > self.ub_list[i]:
                cuckoo.rule[i][2] = self.ub_list[i]
            cuckoo.rule[i][1], cuckoo.rule[i][2] = self.check_attribute_bounds(cuckoo.rule[i][1], cuckoo.rule[i][2], i)
        cuckoo.rule = check_locations_with_concrete_values(cuckoo.rule, locations)

    def check_attribute_bounds(self, lower_bound, upper_bound, i):
        if lower_bound >= upper_bound:
            lower_bound = random.uniform(self.lb_list[i], upper_bound)
        if upper_bound - lower_bound > self.intervals[i]:
            upper_bound = lower_bound + self.intervals[i]/2
        return lower_bound, upper_bound

    def get_new_cuckoo_by_levy_flights(self, source_cuckoo, target_cuckoo):
        beta = 3 / 2
        sigma1 = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (
                math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))))
        sigma2 = sigma1 ** (1 - beta)

        for i in range(0, self.num_of_attributes):
            u = random.random() * sigma2
            v = random.random()
            step = u / (abs(v) ** (1 / beta))
            step_size1 = self.w1 * step * (target_cuckoo.rule[i][0] - source_cuckoo.rule[i][0] * random.random())
            step_size2 = self.w2 * step * (target_cuckoo.rule[i][1] - source_cuckoo.rule[i][1] * random.random())
            step_size3 = self.w3 * step * (target_cuckoo.rule[i][2] - source_cuckoo.rule[i][2] * random.random())
            source_cuckoo.rule[i][0] = round(source_cuckoo.rule[i][0] + step_size1 * random.random())
            source_cuckoo.rule[i][1] = source_cuckoo.rule[i][1] + step_size2 * random.random()
            source_cuckoo.rule[i][2] = source_cuckoo.rule[i][2] + step_size3 * random.random()

        cuckoo_egg = self.do_mutation(source_cuckoo)
        self.check_condition_and_fix(cuckoo_egg)
        supp, conf, comp, inter = self.evaluate_objectives(cuckoo_egg.rule)
        cuckoo_egg.set_objectives(supp, conf, comp, inter)
        return cuckoo_egg

    def do_mutation(self, source_cuckoo):
        mutation_list = [0, 1]
        distribution = [1 - self.pmut, self.pmut]
        for f in source_cuckoo.rule:
            if random.choices(mutation_list, distribution)[0] == 1:
                f[0] = random.choices(self.location_list, self.distribution)[0]
        return source_cuckoo

    def get_best_cuckoo_with_tournament(self):
        tournament = random.choices(self.population, k=self.num_of_tourn)
        non_dom = find_non_dominated(tournament, first_rule_dominates_second_mocanar)
        return random.choices(non_dom, k=1)[0]

    def evaluate_objectives(self, rule):
        antecedent_list, consequent_list = find_antecedent_and_consequent_parts(rule,
                                                                                self.attributes,
                                                                                antecedent_consequent_mocanar)
        antecedent, consequent, both_records, all_records = query_records(antecedent_list, consequent_list, self.data)
        consequent_attributes = len(consequent_list)
        all_attributes = len(antecedent_list) + consequent_attributes
        supp, conf, comp, inter = calculate_objectives(antecedent, consequent, both_records, all_records, consequent_attributes, all_attributes)
        if supp < self.min_support or conf < self.min_confidence:
            return 0, 0, 0, 0
        else:
            return supp, conf, comp, inter

    def do_choosing(self):
        obj_share = round(self.population_size / 4)

        self.cuckoo_eggs.sort(key=lambda x: x.supp, reverse=True)
        sup_share = round(len(self.cuckoo_eggs) - self.pa * len(self.cuckoo_eggs))
        self.cuckoo_eggs = self.cuckoo_eggs[:sup_share]

        temp_rules = merge(self.cuckoo_eggs, self.population)
        new_population = []

        temp_rules.sort(key=lambda x: x.supp, reverse=True)
        new_population.extend(temp_rules[:obj_share])
        temp_rules = temp_rules[obj_share:]

        temp_rules.sort(key=lambda x: x.conf, reverse=True)
        new_population.extend(temp_rules[:obj_share])
        temp_rules = temp_rules[obj_share:]

        temp_rules.sort(key=lambda x: x.inter, reverse=True)
        new_population.extend(temp_rules[:obj_share])
        temp_rules = temp_rules[obj_share:]

        temp_rules.sort(key=lambda x: x.comp, reverse=True)
        new_population.extend(temp_rules[:obj_share])

        return new_population


def merge(non_dominated, population):
    temp_rules = copy.deepcopy(population)
    temp_rules.extend(copy.deepcopy(non_dominated))
    return temp_rules
