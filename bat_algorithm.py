import copy
import math
import random
import numpy as np
import pandas as pd
from helpers.data_checker import fix_locations, is_move_location_indicator, fix_location_indicator, \
    is_fix_antecedent, fix_attribute_value, is_fix_consequent
from helpers.data_discretization import discretize
from helpers.data_queries import find_antecedent_and_consequent_parts, antecedent_consequent_mob_arm, query_records
from helpers.helper import delete_duplicated_rules
from helpers.objective_functions import calculate_objectives


class Bat:

    def __init__(self, frequency, velocity, rate, loudness, rule,
                 objective, support, confidence, comprehensibility, interestingness):
        self.frequency = frequency
        self.velocity = velocity
        self.rate = rate
        self.initial_rate = rate
        self.loudness = loudness
        self.rule = rule
        self.obj = objective
        self.supp = support
        self.conf = confidence
        self.comp = comprehensibility
        self.inter = interestingness

    def __str__(self):
        txt = "Support:{supp:.2f}, Confidence:{conf:.2f}, Comprehensibility:{comp:.2f}, Interestingness:{inter:.2f}"
        return txt.format(supp=self.supp, conf=self.conf, comp=self.comp, inter=self.inter)

    def set_objectives(self, objective, support, confidence, comprehensibility, interestingness):
        self.obj = objective
        self.supp = support
        self.conf = confidence
        self.comp = comprehensibility
        self.inter = interestingness


class BatAlgorithm:

    def __init__(self,
                 data,
                 population_size,
                 iterations,
                 pareto_points,
                 alpha, beta, gamma, delta,
                 min_support,
                 min_confidence):
        self.attributes = list(data.columns)
        self.data = discretize(data, self.attributes)
        self.population_size = population_size
        self.iterations = iterations
        self.pareto_points = pareto_points
        self.population = []
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.average_loudness = 0
        self.evolution_counter = 0
        self.min_support = min_support
        self.min_confidence = min_confidence

        self.gbest = None
        self.lb, self.ub = self.find_lower_and_upper_bounds()

    def mob_arm(self):
        self.initialize_population()
        self.population.sort(key=lambda x: x.obj, reverse=True)
        self.gbest = copy.deepcopy(self.population[0])
        non_dominate_solutions = [copy.deepcopy(self.gbest)]

        for p in range(0, self.pareto_points):
            weights = np.random.dirichlet(np.ones(2), size=1)[0]

            for j in range(self.iterations):
                print("parento ", p)
                print("iteration", j)

                for bat in self.population:
                    bat.frequency = 1 + len(self.attributes) * self.beta
                    bat.velocity = len(self.attributes) - bat.frequency - bat.velocity
                    new_rule = self.generate_new_solution(bat)

                    if random.random() > bat.rate:
                        random_index = random.randrange(1, len(self.attributes))
                        new_rule[random_index] = self.gbest.rule[random_index]
                    new_rule = self.check_and_fix(new_rule)

                    obj, supp, conf, comp, inter = self.evaluate_fitness(new_rule, weights)

                    if obj > bat.obj:
                        bat.rule = new_rule
                        bat.rate = bat.initial_rate * (1 - math.exp(-self.gamma * j))
                        bat.loudness = self.alpha * bat.loudness
                        self.evolution_counter += 1
                        bat.set_objectives(obj, supp, conf, comp, inter)
                        non_dominate_solutions.append(copy.deepcopy(bat))

                self.population.sort(key=lambda x: x.obj, reverse=True)
                if self.population[0].obj > self.gbest.obj:
                    self.gbest = copy.deepcopy(self.population[0])

            print("len", len(non_dominate_solutions))

        return delete_duplicated_rules(non_dominate_solutions)

    def generate_random_rule(self):
        rule = [random.randrange(2, len(self.attributes) - 1)]
        for r in self.attributes:
            if random.choices([0, 1], [0.5, 0.5])[0] == 1:
                rule.append(random.choices(self.data[(r, "id")])[0])
            else:
                rule.append(0)
        return self.check_and_fix(rule)

    def check_and_fix(self, rule):
        antecedent, consequent = find_antecedent_and_consequent_parts(rule, self.attributes,
                                                                      antecedent_consequent_mob_arm)
        if len(antecedent) == 0 and len(consequent) > 1:
            rule = fix_locations(rule, is_move_location_indicator, fix_location_indicator, self.data, self.attributes)
        elif len(antecedent) == 0:
            rule = fix_locations(rule, is_fix_antecedent, fix_attribute_value, self.data, self.attributes)
        if len(consequent) == 0 and len(antecedent) > 1:
            rule = fix_locations(rule, is_move_location_indicator, fix_location_indicator, self.data, self.attributes)
        elif len(consequent) == 0:
            rule = fix_locations(rule, is_fix_consequent, fix_attribute_value, self.data, self.attributes)
        return rule

    def calculate_objectives(self, bat, weights):
        rule = bat.rule
        bat.obj, bat.supp, bat.conf, bat.comp, bat.inter = self.evaluate_fitness(rule, weights)

    def evaluate_fitness(self, rule, weights):
        antecedent_part, consequent_part = find_antecedent_and_consequent_parts(rule, self.attributes,
                                                                                antecedent_consequent_mob_arm)
        query_antecedent = ' and '.join(antecedent_part)
        query_consequent = ' and '.join(consequent_part)

        antecedent = self.data.query(query_antecedent).describe().iloc[0, 0]
        consequent = self.data.query(query_consequent).describe().iloc[0, 0]
        contain_both = self.data.query(query_antecedent + ' and ' + query_consequent).describe().iloc[0, 0]
        all_records = self.data.describe().iloc[0, 0]

        consequent_attributes = len(consequent_part)
        all_attributes = len(antecedent_part) + consequent_attributes

        supp, conf, comp, inter = calculate_objectives(antecedent, consequent, contain_both, all_records, consequent_attributes, all_attributes)

        if supp < self.min_support or conf < self.min_confidence:
            return 0, 0, 0, 0, 0

        objective1 = self.alpha * conf + self.beta * supp / self.alpha + self.beta
        objective2 = self.gamma * comp + self.delta * inter / self.gamma + self.delta
        objective = weights[0] * objective1 + weights[1] * objective2
        return objective, supp, conf, comp, inter

    def generate_new_solution(self, bat):
        rule = copy.deepcopy(bat.rule)
        velocity = copy.deepcopy(bat.velocity)
        frequency = copy.deepcopy(bat.frequency)
        while velocity < frequency:
            if random.random() > bat.loudness:
                velocity += 1
            else:
                velocity -= 1

            if velocity <= 0 or velocity > len(self.attributes):
                velocity = 0

            rule = self.modify_rule(velocity, frequency, rule)
            velocity += 1
        for r in range(len(rule)):
            rule[r] = round(rule[r])
        return rule

    def modify_rule(self, velocity, frequency, rule):
        counter = frequency
        for r in range(1, len(rule)):
            if counter == 0:
                break
            if r >= velocity and rule[r] != 0:
                new_rule = rule[r] + velocity
                if self.ub[r] < new_rule or self.lb[r] > new_rule:
                    new_rule = random.choices(self.data[(self.attributes[r - 1], "id")])[0]
                rule[r] = new_rule
                counter -= 1
        return rule

    def initialize_population(self):
        for i in range(0, self.population_size):
            print("ba ", i)
            bat = self.initialize_bat()
            self.population.append(bat)
            print(bat.rule, bat.supp, bat.conf)

    def initialize_bat(self):
        rule = self.generate_random_rule()
        frequency = random.randrange(0, len(self.attributes))
        velocity = random.randrange(0, len(self.attributes) + 1)
        rate = random.random()
        loudness = random.random()
        obj, supp, conf, comp, inter = self.evaluate_fitness(rule, [0.5, 0.5])
        return Bat(frequency, velocity, rate, loudness, rule, obj, supp, conf, comp, inter)

    def calculate_average_loudness(self):
        loudness = 0
        for l in self.population:
            loudness += l.loudness
        self.average_loudness = loudness / len(self.population)

    def find_lower_and_upper_bounds(self):
        lb = [1]
        ub = [len(self.attributes)]
        for a in self.attributes:
            lower, upper = self.find_attribute_value_range(a)
            lb.append(lower)
            ub.append(upper)
        return lb, ub

    def find_attribute_value_range(self, attribute):
        print(self.data[(attribute, "id")].describe())
        return self.data[(attribute, "id")].describe().iloc[3], self.data[(attribute, "id")].describe().iloc[-1]
