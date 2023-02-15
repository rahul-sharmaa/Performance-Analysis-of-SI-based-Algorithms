import copy
import math
import random
import numpy as np
from helpers.data_checker import find_locations_with_value_range, check_locations_with_value_range
from helpers.data_init import define_lb_and_ub
from helpers.data_queries import query_records, antecedent_consequent_aco_r, find_antecedent_and_consequent_parts
from helpers.helper import find_non_dominated, first_rule_dominates_second_aco_r
from helpers.objective_functions import calculate_objectives


class Solution:

    def __init__(self, rule, support, confidence, interestingness, int_objective, objective, comprehensibility):
        self.rule = rule
        self.conf = confidence
        self.supp = support
        self.inter = interestingness
        self.int_obj = int_objective
        self.obj = objective
        self.comp = comprehensibility

    def __str__(self):
        txt = "Obj:{obj:.2f}, Support:{supp:.2f}, Confidence:{conf:.2f}, Interestingness:{inter:.2f}"
        return txt.format(obj=self.obj, supp=self.supp, conf=self.conf, inter=self.inter)

    def set_objectives(self, support, confidence, interestingness, int_obj, objective, comp):
        self.supp = support
        self.conf = confidence
        self.inter = interestingness
        self.int_obj = int_obj
        self.obj = objective
        self.comp = comp


class AntColonyOptimization:

    def __init__(self,
                 data,
                 archive_size,
                 ant_colony_size,
                 max_iterations,
                 alpha1, alpha2, alpha3, alpha4, alpha5,
                 q, e):
        self.data = data
        self.attributes = list(data.columns)
        self.archive_size = archive_size
        self.ant_colony_size = ant_colony_size
        self.max_iterations = max_iterations
        self.archive, self.weights, self.probabilities = [], [], []
        self.alpha1, self.alpha2, self.alpha3, self.alpha4, self.alpha5 = alpha1, alpha2, alpha3, alpha4, alpha5
        self.q = q
        self.e = e
        self.lb_list, self.ub_list = define_lb_and_ub(self.attributes, self.data)
        self.all_records = self.data.describe().iloc[0, 0]

    def ant_colony_optimization_for_continuous_domains(self):
        self.init_archive()
        self.archive.sort(key=lambda l: l.obj, reverse=True)

        for i in range(self.max_iterations):
            print(i)
            self.weights = self.init_weights()
            self.probabilities = self.calculate_probabilities()
            ants = []

            for _ in range(self.ant_colony_size):
                solution = copy.deepcopy(random.choices(self.archive, self.probabilities)[0])

                for count, s in enumerate(solution.rule):
                    s[1], s[2] = self.sample_gaussian_function(count, s[1])

                supp, conf, inter, int_obj, obj, comp = self.evaluate_objectives(solution.rule)
                solution.set_objectives(supp, conf, inter, int_obj, obj, comp)
                ants.append(solution)

            self.archive.extend(ants)
            self.archive.sort(key=lambda k: k.obj, reverse=True)
            self.archive = self.archive[:self.archive_size]

        return find_non_dominated(self.archive, first_rule_dominates_second_aco_r)

    def init_archive(self):
        for i in range(self.archive_size):
            row = []
            for k in range(len(self.attributes)):
                row.append(self.generate_solution(k))
            locations = find_locations_with_value_range(row)
            row = check_locations_with_value_range(row, locations)
            supp, conf, inter, int_obj, obj, comp = self.evaluate_objectives(row)
            print(supp, conf, inter)
            solution = Solution(row, supp, conf, inter, int_obj, obj, comp)
            self.archive.append(solution)

    def generate_solution(self, dimension):
        ac = round(random.uniform(0, 1), 2)

        lb = random.uniform(self.lb_list[dimension], self.ub_list[dimension])
        ub = random.uniform(lb, self.ub_list[dimension])

        std = (ub - lb) / 2
        value = lb + std

        return [ac, value, std]

    def init_weights(self):
        weights = []
        for counter, solution in enumerate(self.archive):
            k = self.archive_size
            weight = math.e ** ((-counter ** 2) / (2 * self.q ** 2 * k ** 2)) / (self.q * k * math.sqrt(2 * math.pi))
            weights.append(weight)
        return weights

    def calculate_probabilities(self):
        probabilities = []
        weight_sum = 0
        for w in self.weights:
            weight_sum += w
        for weight in self.weights:
            probability = weight / weight_sum
            probabilities.append(probability)
        return probabilities

    def calculate_standard_deviation(self, index, value):
        standard_deviation = 0
        for solution in self.archive:
            standard_deviation += abs(solution.rule[index][1] - value) / (self.archive_size - 1)
        standard_deviation *= self.e
        return standard_deviation

    def evaluate_objectives(self, rule):
        antecedent_part, consequent_part = find_antecedent_and_consequent_parts(rule, self.attributes,
                                                                                antecedent_consequent_aco_r)
        antecedent, consequent, both_records, all_records = query_records(antecedent_part, consequent_part, self.data)
        consequent_attributes = len(consequent_part)
        all_attributes = len(antecedent_part) + consequent_attributes
        supp, conf, comp, inter = calculate_objectives(antecedent, consequent, both_records, all_records,
                                                       consequent_attributes, all_attributes)
        int_obj = self.calculate_int_obj(rule)
        obj = self.calculate_obj(supp, conf, inter, int_obj)
        return supp, conf, inter, int_obj, obj, comp

    def calculate_int_obj(self, rule):
        int_obj = 0
        for c, r in enumerate(rule):
            ub, lb = self.calculate_upper_bound(r[1], r[2]), self.calculate_lower_bound(r[1], r[2])
            int_obj += (ub - lb) / (self.ub_list[c] - self.lb_list[c])
        return int_obj

    def calculate_obj(self, supp, conf, inter, int_obj):
        return self.alpha1 * supp + self.alpha2 * conf + self.alpha3 * inter - self.alpha4 * int_obj

    def calculate_upper_bound(self, value, sd):
        return value + self.alpha5 * sd

    def calculate_lower_bound(self, value, sd):
        return value - self.alpha5 * sd

    def sample_gaussian_function(self, dimension, value):
        sd = self.calculate_standard_deviation(dimension, value)
        x_list = np.linspace(self.lb_list[dimension], self.ub_list[dimension], int(self.all_records))
        pdf = np.exp((-0.5 * (x_list - value) ** 2) / sd ** 2) / (sd * np.sqrt(2 * np.pi))
        new_value = random.choices(x_list, pdf)[0]
        new_sd = sd
        return new_value, new_sd
