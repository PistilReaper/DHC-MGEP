# coding=utf-8

'''
    .. moduleauthor:: Tianyi Chen
This :mod:`DHC_MGEP` module provides the addtional algrithms of DHC-MGEP to DHC-GEP, 
including functions of tensor operators,
'''
import numpy as np
import itertools    
import random
from fractions import Fraction
import os
import sympy as sp
import re
import deap
import warnings
import pickle
import datetime
import geppy as gep
import time

"""
The original author of the following functions is Shuhua Gao and Wenjun Ma. The geppy source code written by Shuhua Gao can be here https://github.com/ShuhuaGao/geppy/blob/master/geppy/algorithms/basic.py.
Wenjun Ma modified the function 'gep_simple', adding the following features:
    * Output the real-time resultant mathematical expressions to a .dat file.
    * Write the populations to a .pkl file every 20 generations for ease of subsequent restarting if necessary.
    * Terminate evolution with given threshold. When the error of the best individual is smaller than the threshold, the evolution is terminated. 

Following module provides fundamental boilerplate GEP algorithm implementations. After registering proper
operations into a :class:`deap.base.Toolbox` object, the GEP evolution can be simply launched using the present
algorithms. Of course, for complicated problems, you may want to define your own algorithms, and the implementation here
can be used as a reference.
"""
# from builtins import str
def plasmid_generate(func, host_pop):
    '''
    :param func: The plasmid individual function
    :param host_pop: The host population
    '''
    plasmid_pop = []
    p_times_lis = count_p_functions(host_pop)
    for i in range(len(p_times_lis)):
        plasmid_pop.append([])
        for j in range(p_times_lis[i]):
            plasmid_pop[i].append(func())
    return plasmid_pop

def count_p_functions(host_pop):
    p_counts = []
    for ind in host_pop:
        matches = re.findall('p_\(', str(ind))
        p_counts.append(len(matches))
    return p_counts



def _validate_basic_toolbox(tb):
    """
    Validate the operators in the toolbox *tb* according to our conventions.
    """
    assert hasattr(tb, 'select'), "The toolbox must have a 'select' operator."
    # whether the ops in .pbs are all registered
    for op in tb.pbs:
        assert op.startswith('mut') or op.startswith('cx'), "Operators must start with 'mut' or 'cx' except selection."
        assert hasattr(tb, op), "Probability for a operator called '{}' is specified, but this operator is not " \
                                "registered in the toolbox.".format(op)
    # whether all the mut_ and cx_ operators have their probabilities assigned in .pbs
    for op in [attr for attr in dir(tb) if attr.startswith('mut') or attr.startswith('cx')]:
        if op not in tb.pbs:
            warnings.warn('{0} is registered, but its probability is NOT assigned in Toolbox.pbs. '
                          'By default, the probability is ZERO and the operator {0} will NOT be applied.'.format(op),
                          category=UserWarning)


def _apply_modification(population, operator, pb):
    """
    Apply the modification given by *operator* to each individual in *population* with probability *pb* in place.
    """
    for i in range(len(population)):
        if random.random() < pb:
            population[i], = operator(population[i])
            del population[i].fitness.values
            
            population[i].plasmid = []
    return population

def _apply_modification_plasmid(population, operator, pb):
    for i in range(len(population)):
        if len(population[i].plasmid) > 0:
            for j in len(population[i].plasmid):
                if random.random() < pb:
                    population[i].plasmid[j], = operator(population[i].plasmid[j])
                    del population[i].fitness.values
    return population

def _apply_crossover(population, operator, pb):
    """
    Mate the *population* in place using *operator* with probability *pb*.
    """
    for i in range(1, len(population), 2):
        if random.random() < pb:
            population[i - 1], population[i] = operator(population[i - 1], population[i])
            del population[i - 1].fitness.values
            del population[i].fitness.values

            population[i-1].plasmid = []
            population[i].plasmid = []
    return population


def gep_simple(host_population, plasmid_population, toolbox, n_generations=100, n_elites=1,
               stats=None, hall_of_fame=None, verbose=__debug__,tolerance = 1e-10,GEP_type = ''):
    """
    This algorithm performs the simplest and standard gene expression programming.
    The flowchart of this algorithm can be found
    `here <https://www.gepsoft.com/gxpt4kb/Chapter06/Section1/SS1.htm>`_.
    Refer to Chapter 3 of [FC2006]_ to learn more about this basic algorithm.

    .. note::
        The algorithm framework also supports the GEP-RNC algorithm, which evolves genes with an additional Dc domain for
        random numerical constant manipulation. To adopt :func:`gep_simple` for GEP-RNC evolution, use the
        :class:`~geppy.core.entity.GeneDc` objects as the genes and register Dc-specific operators.
        A detailed example of GEP-RNC can be found at `numerical expression inference with GEP-RNC
        <https://github.com/ShuhuaGao/geppy/blob/master/examples/sr/numerical_expression_inference-RNC.ipynb>`_.
        Users can refer to Chapter 5 of [FC2006]_ to get familiar with the GEP-RNC theory.

    :param host_population: a list of host individuals
    :param plasmid_population: a list of plasmid individuals
    :param toolbox: :class:`~geppy.tools.toolbox.Toolbox`, a container of operators. Regarding the conventions of
        operator design and registration, please refer to :ref:`convention`.
    :param n_generations: max number of generations to be evolved
    :param n_elites: number of elites to be cloned to next generation
    :param stats: a :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param hall_of_fame: a :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: whether or not to print the statistics.
    :returns: The final population
    :returns: A :class:`~deap.tools.Logbook` recording the statistics of the
              evolution process

    .. note:
        To implement the GEP-RNC algorithm for numerical constant evolution, the :class:`geppy.core.entity.GeneDc` genes
        should be used. Specific operators are used to evolve the Dc domain of :class:`~geppy.core.entity.GeneDc` genes
        including Dc-specific mutation/inversion/transposition and direct mutation of the RNC array associated with
        each gene. These operators should be registered into the *toolbox*.
    """
    _validate_basic_toolbox(toolbox)
    logbook = deap.tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    start_time = time.time()

    is_exists = os.path.exists('pkl')
    if not is_exists:
        os.mkdir('pkl')
    
    is_exists = os.path.exists('output')
    if not is_exists:
        os.mkdir('output')
    
    simplified_best_list = []
    for gen in range(n_generations + 1):
        # %% First, generation for hosts
        
        # evaluate: only evaluate the invalid ones, i.e., no need to reevaluate the unchanged ones
        invalid_individuals = [ind for ind in host_population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_individuals)

        for ind, fit in zip(invalid_individuals, fitnesses):
            ind.fitness.values = fit


        # record statistics and log
        if hall_of_fame is not None:
            hall_of_fame.update(host_population)
        record = stats.compile(host_population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_individuals), **record)

        if verbose:
            print(logbook.stream)

        if gen == n_generations:
            break

        # selection with elitism
        elites = deap.tools.selBest(host_population, k=n_elites)
        offspring = toolbox.select(host_population, len(host_population) - n_elites)

        # output the real-time result
        # Every 20 generations, the current optimal individual is checked, and if a new optimal individual appears, it is output to file.
        if gen > 0 and gen % 20 == 0:
            elites_IR = elites[0]
            simplified_best = toolbox.compile(elites_IR)
            if str(simplified_best) not in simplified_best_list:
                simplified_best_list.append(str(simplified_best))
                elapsed = time.time() - start_time
                time_str = '%.2f' % (elapsed)   
                if elites_IR.fitness.values[0] != -1.:      
                    key= f'In generation {gen}, with CPU running {time_str}s, \nOur No.1 best prediction is:'
                    with open(f'output/{GEP_type}_equation.dat', "a") as f:
                        f.write('\n'+ key+ str(simplified_best)+ '\n'+f'with fitness = {elites_IR.fitness.values[0]}'+'\n')
                else:
                    key= f'In generation {gen}, with CPU running {time_str}s, \nOur No.1 best prediction 1 is:'
                    with open(f'output/{GEP_type}_equation.dat', "a") as f:
                        f.write('\n'+ key+ str(simplified_best)+ '\n'+f'which is invalid!'+'\n' )

        # Termination criterion of error tolerance
        if gen > 0 and gen % 100 == 0:
            error_min = 1 - elites[0].fitness.values[0]
            if error_min < tolerance:
                time_now = str(datetime.datetime.now())
                pklFileName = time_now[:16].replace(':', '_').replace(' ', '_')
                output_hal = open(f'pkl/{GEP_type}.pkl', 'wb')
                str_class = pickle.dumps(host_population)
                output_hal.write(str_class)
                output_hal.close()
                break

        # # Termination criterion of error reducing velocity
        # # Compute the best individual every 500 generations, if the best 
        # # one is the same with that of the 500 generations before, 
        # # terminate the evolution immediately.
        # if gen % 300 == 0:
        #     elites_IR = elites[0]
        #     middle_simplified_best = gep.simplify(elites_IR)
        #     if gen == 0:
        #         realtime_middle_simplified_best = middle_simplified_best
        #     else:
        #         if realtime_middle_simplified_best == middle_simplified_best:
        #             output_hal = open(f'pkl/{GEP_type}.pkl', 'wb')
        #             str_class = pickle.dumps(host_population)
        #             output_hal.write(str_class)
        #             output_hal.close()
        #             break
        #         else:
        #             realtime_middle_simplified_best = middle_simplified_best

        # replication 
        # toolbox.clone() is copy.deepcopy()
        offspring = [toolbox.clone(ind) for ind in offspring]
        
        # mutation
        for op in toolbox.pbs:
            if op.startswith('mut_'):
                offspring = _apply_modification(offspring, getattr(toolbox, op), toolbox.pbs[op])

        
        # crossover
        for op in toolbox.pbs:
            if op.startswith('cx_'):
                offspring = _apply_crossover(offspring, getattr(toolbox, op), toolbox.pbs[op])

        
        # %% Then, generation for plasmids (only apply operators to plasmids whose host is not changed)
        # mutation for plasmid
        for op in toolbox.pbs:
            if op.startswith('mut'):#including Dc-specific operators
                offspring = _apply_modification_plasmid(offspring, getattr(toolbox, op), toolbox.pbs[op])
        
        # generate new plasmids for mutated offsprings
        for ind in offspring:
            if len(ind.plasmid) == 0:
                matches = re.findall('p_\(', str(ind))
                for j in range(len(matches)):
                    ind.plasmid.append(toolbox.plasmid_individual())
                    # if 'mul(D_ii, miu)' in str(ind.plasmid[-1]):
                    #     print(str(ind.plasmid[-1]))
        
        # replace the current host population with the offsprings and update plasmid population
        host_population = elites + offspring
        for ind_host, ind_plasmid in zip(host_population, plasmid_population):
            ind_plasmid = ind_host.plasmid
    
    time_now = str(datetime.datetime.now())
    pklFileName = time_now[:16].replace(':', '_').replace(' ', '_')
    output_hal = open(f'pkl/{GEP_type}.pkl', 'wb')
    str_class = pickle.dumps(host_population)
    output_hal.write(str_class)
    output_hal.close()

    return host_population, logbook


__all__ = ['gep_simple', 'plasmid_generate', 'count_p_functions', 'evaluate', 'dimensional_verification']