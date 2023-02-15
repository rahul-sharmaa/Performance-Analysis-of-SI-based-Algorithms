import copy
import random
import time

from helpers.data_checker import find_locations_with_value_range, check_locations_with_value_range
from helpers.data_init import define_lb_and_ub
from helpers.data_queries import find_antecedent_and_consequent_parts, antecedent_consequent_mopar, query_records
from helpers.helper import find_non_dominated, first_rule_dominates_second_mopar, dominates_mopar, \
    delete_duplicated_rules
from helpers.objective_functions import calculate_objectives


class Particle:

    def __init__(self, rule, velocities, support, confidence, comprehensibility, interestingness):
        self.rule = rule
        self.velocities = velocities
        self.local_non_dominated = [rule]
        self.local_best = rule
        self.local_best_objectives = [(confidence, comprehensibility, interestingness)]
        self.supp = support
        self.conf = confidence
        self.comp = comprehensibility
        self.inter = interestingness
        self.rank = 0
        self.probability = 0
        self.dominates_count = 0

    def __str__(self):
        txt = "Support:{supp:.2f}, Confidence:{conf:.2f}, Comprehensibility:{comp:.2f}, Interestingness:{inter:.2f}"
        return txt.format(supp=self.supp, conf=self.conf, comp=self.comp, inter=self.inter)

    def update_velocity(self, inertia_weight, c1, c2, global_best, velocity_limit):
        for counter, velocity in enumerate(self.velocities):
            v1 = []
            for c, v in enumerate(velocity):
                r1, r2 = random.random(), random.random()
                position = self.rule[counter][c]
                local_best_pos = self.local_best[counter][c]
                global_best_pos = global_best[counter][c]
                new_velocity = inertia_weight * velocity[c] + c1 * r1 * (local_best_pos - position) + c2 * r2 * (
                        global_best_pos - position)
                if new_velocity > velocity_limit:
                    new_velocity = new_velocity / velocity_limit
                v1.append(new_velocity)
            self.velocities[counter] = v1

    def update_position(self):
        new_rule = []
        for c, r in enumerate(self.rule):
            new_r = [round(r[0] + self.velocities[c][0], 2), r[1] + self.velocities[c][1], r[2] + self.velocities[c][2]]
            new_rule.append(new_r)
        self.rule = new_rule

    def set_objectives(self, support, confidence, comprehensibility, interestingness):
        self.supp = support
        self.conf = confidence
        self.comp = comprehensibility
        self.inter = interestingness

    def update_non_dominated_local(self):
        l_conf, l_comp, l_inter = self.local_best_objectives[-1]
        rule = copy.deepcopy(self.rule)
        conf, comp, inter = copy.deepcopy(self.conf), copy.deepcopy(self.comp), copy.deepcopy(self.inter)
        if dominates_mopar(conf, comp, inter, l_conf, l_comp, l_inter):
            self.local_best = rule
            self.local_non_dominated.append(rule)
            self.local_best_objectives.append((conf, comp, inter))
        elif not dominates_mopar(l_conf, l_comp, l_inter, conf, comp, inter):
            chosen = random.choices([self.rule, self.local_best])[0]
            if chosen == self.rule:
                self.local_best = rule
                self.local_non_dominated.append(rule)
                self.local_best_objectives.append((conf, comp, inter))

    def find_dominated_count(self):
        counter = 0
        p_conf, p_comp, p_inter = self.conf, self.comp, self.inter
        for counter, rule in enumerate(self.local_non_dominated):
            conf, comp, inter = self.local_best_objectives[counter]
            counter += 1 if dominates_mopar(p_conf, p_comp, p_inter, conf, comp, inter) else 0
        return counter


class ParticleSearchOptimization:

    def __init__(self,
                 data,
                 population_size,
                 max_iterations,
                 external_repository_size,
                 c1, c2,
                 inertia_weight,
                 velocity_limit,
                 x_rank):
        self.data = data
        self.attributes = list(data.columns)
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.external_repository_size = external_repository_size
        self.c1 = c1
        self.c2 = c2
        self.inertia_weight = inertia_weight
        self.velocity_limit = velocity_limit
        self.x_rank = x_rank

        self.lb_list, self.ub_list = define_lb_and_ub(self.attributes, self.data)

    def multi_objective_particle_search_optimization_algorithm(self):
        population = self.init_population()
        external_repository = copy.deepcopy(find_non_dominated(population, first_rule_dominates_second_mopar))
        global_best = copy.deepcopy(self.roulette_wheel(population))

        for i in range(0, self.max_iterations):
            print("iter", i)

            for particle in population:
                self.update_particle(particle, global_best)
                particle.update_non_dominated_local()

            external_repository = self.update_external_repository(external_repository, population)
            global_best = copy.deepcopy(self.update_global_best(external_repository, population))

        return external_repository

    def init_population(self):
        population = []
        for i in range(0, self.population_size):
            rule = self.generate_rule()
            rule = self.check_and_fix_rule(rule)
            particle_velocities = self.generate_velocities()
            supp, conf, comp, inter = self.evaluate_fitness(rule)
            print(supp, conf, comp, inter)
            particle = Particle(rule, particle_velocities, supp, conf, comp, inter)
            population.append(particle)
        return population

    def generate_rule(self):
        rule = []
        for i in range(0, len(self.attributes)):
            lower_bound = random.uniform(self.lb_list[i], self.ub_list[i])
            upper_bound = random.uniform(lower_bound, self.ub_list[i])
            acn = round(random.uniform(0, 1), 2)
            rule.append([acn, lower_bound, upper_bound])
        return self.check_and_fix_rule(rule)

    def generate_velocities(self):
        velocities = []
        for i in range(0, len(self.attributes)):
            velocity = []
            for v in range(0, 3):
                velocity.append(random.uniform(0, self.velocity_limit))
            velocities.append(velocity)
        return velocities

    def update_particle(self, particle, global_best):
        particle.update_velocity(self.inertia_weight, self.c1, self.c2, global_best, self.velocity_limit)
        particle.update_position()
        particle.rule = self.check_and_fix_rule(particle.rule)
        supp, conf, comp, inter = self.evaluate_fitness(particle.rule)
        particle.set_objectives(supp, conf, comp, inter)

    def update_external_repository(self, external_repository, population):
        external_repository.extend(population)
        external_repository = delete_duplicated_rules(external_repository)
        external_repository = copy.deepcopy(find_non_dominated(external_repository, first_rule_dominates_second_mopar))

        if len(external_repository) > self.external_repository_size:
            external_repository.sort(key=lambda x: x.dominates_count)
            return external_repository[:self.external_repository_size]
        return external_repository

    def update_global_best(self, external_repository, population):
        if len(external_repository) == 0:
            return sorted(population, key=lambda x: x.conf + x.comp + x.inter, reverse=True)[0].rule
        return self.roulette_wheel(external_repository)

    def evaluate_fitness(self, rule):
        antecedent_part, consequent_part = find_antecedent_and_consequent_parts(rule, self.attributes,
                                                                                antecedent_consequent_mopar)
        antecedent, consequent, both_records, all_records = query_records(antecedent_part, consequent_part, self.data)
        consequent_attributes = len(consequent_part)
        all_attributes = len(antecedent_part) + consequent_attributes
        supp, conf, comp, inter = calculate_objectives(antecedent, consequent, both_records, all_records, consequent_attributes, all_attributes)
        print(supp, conf, inter)
        return supp, conf, comp, inter

    def roulette_wheel(self, population):
        sum_of_ranks = 0
        probabilities = []
        for p in population:
            dom_count = p.find_dominated_count()
            p.rank = self.x_rank if dom_count == 0 else self.x_rank / dom_count
            sum_of_ranks += p.rank
        for p in population:
            p.probability = p.rank / sum_of_ranks
            probabilities.append(p.probability)
        return random.choices(population, probabilities)[0].rule

    def check_and_fix_rule(self, rule):
        rule = self.check_rule_limits(rule)
        locations = find_locations_with_value_range(rule)
        rule = check_locations_with_value_range(rule, locations)
        return rule

    def check_rule_limits(self, rule):
        new_rule = []
        for c, r in enumerate(rule):
            acn, lb, ub = r[0], r[1], r[2]
            if acn < 0 or acn > 1:
                acn = round(random.uniform(0, 1), 2)
            if lb < self.lb_list[c] or lb >= self.ub_list[c]:
                lb = random.uniform(self.lb_list[c], self.ub_list[c])
            if ub > self.ub_list[c] or ub < self.lb_list[c]:
                ub = random.uniform(lb, self.ub_list[c])
            new_rule.append([acn, lb, ub])
        return new_rule
