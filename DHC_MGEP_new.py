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
# new genetic operators designed for M-GEP
_DEBUG = False

def _choose_subsequence(seq, min_length=1, max_length=-1):
    if max_length <= 0:
        max_length = len(seq)
    length = random.randint(min_length, max_length)
    start = random.randint(0, len(seq) - length)
    return start, start + length
    
def _choose_function(pset):
    return random.choice(pset.functions)

def _choose_a_terminal(terminals):
    """
    Choose a terminal from the given list *terminals* randomly.

    :param terminals: iterable, terminal set
    :return: a terminal
    """
    terminal = random.choice(terminals)
    if isinstance(terminal, gep.EphemeralTerminal):  # an Ephemeral
        terminal = copy.deepcopy(terminal)  # generate a new one
        terminal.update_value()
    return terminal

def _choose_terminal(pset):
    return _choose_a_terminal(pset.terminals) 

def mutate_uniform(host_individual, host_pset, func, ind_pb='2p'):
    plasmid_lis_lis = host_individual.plasmid
    if isinstance(ind_pb, str):
        assert ind_pb.endswith('p'), "ind_pb must end with 'p' if given in a string form"
        length = host_individual[0].head_length + host_individual[0].tail_length
        ind_pb = float(ind_pb.rstrip('p')) / (len(host_individual) * length)
    for gene, plasmid_lis in zip(host_individual, plasmid_lis_lis):
        # mutate the gene with the associated pset
        # head: any symbol can be changed into a function or a terminal
        for i in range(gene.head_length):
            if random.random() < ind_pb:
                if gene[i].name == 'p_' : # p mutate
                    if random.random() < 0.5:  # to a function
                        new_function = _choose_function(host_pset)
                        gene[i] = new_function
                        if new_function.name != 'p_': # p to else function
                            plasmid_lis.pop(0)
                    else:                      # to a terminal
                        gene[i] = _choose_terminal(host_pset)
                        plasmid_lis.pop(0) # p to terminal
                else: # not p mutate
                    if random.random() < 0.5:  # to a function
                        new_function = _choose_function(host_pset)
                        gene[i] = new_function
                        if new_function.name == 'p_': # else to p
                            plasmid_lis.insert(0, func())
                    else:                      # to a terminal
                        gene[i] = _choose_terminal(host_pset)
        # tail: only change to another terminal
        for i in range(gene.head_length, gene.head_length + gene.tail_length):
            if random.random() < ind_pb:
                gene[i] = _choose_terminal(host_pset)
    return host_individual,

def invert(individual):
    """
    A gene is randomly chosen, and afterwards a subsequence within this gene's head domain is randomly selected
    and inverted.

    :param individual: :class:`~geppy.core.entity.Chromosome`, a chromosome for host
    :return: a tuple of one individual

    Typically, a small inversion rate of 0.1 is used.
    """
    if individual.head_length < 2:
        return individual,
    location = random.choice(range(len(individual)))
    gene = individual[location]
    plasmid_lis = individual.plasmid[location]
    start, end = _choose_subsequence(gene.head, 2, gene.head_length)
    subsequence = gene[start: end]
    n_i = 0
    n_j = 0
    for i in range(end+1):
        if gene[i].name == 'p_':
            n_i += 1
    for j in range(len(subsequence)):
        if subsequence[j].name == 'p_':
            n_j += 1
    gene[start: end] = reversed(gene[start: end])
    plasmid_lis[(n_i - n_j):n_i] = reversed(plasmid_lis[(n_i - n_j):n_i])
    if _DEBUG:
        print('invert [{}: {}]'.format(start, end))
    return individual,

def _choose_donor_donee(individual):
    i1, i2 = random.choices(range(len(individual)), k=2)  # with replacement
    return individual[i1], individual[i2], i1, i2

def _choose_subsequence_indices(i, j, min_length=1, max_length=-1):
    """
    Choose a subsequence from [i, j] (both included) and return the subsequence boundaries [a, b] (both included).
    Additionally, the min_length and max length of [a, b], i.e., b - a + 1 can be specified.
    """
    if max_length <= 0:
        max_length = j - i + 1
    length = random.randint(min_length, max_length)
    start = random.randint(i, j - length + 1)
    return start, start + length - 1

def is_transpose(individual):
    """
    Perform IS transposition in place

    :param individual: :class:`~geppy.core.entity.Chromosome`, a chromosome for host
    :return: a tuple of one individual

    An Insertion Sequence (IS) is a random chosen segment across the chromosome, and then the
    IS is copied to be inserted at another position in the head of gene, except the start position. Note that an IS from
    one gene may be copied to a different gene.
    Typically, the IS transposition rate is set around 0.1.
    """
    # Donor is the gene who give out a segment, while donee is the gene who take in a segment. Donor may be the same as donee.
    donor, donee, i1, i2 = _choose_donor_donee(individual) 
    donor_plasmid_lis = individual.plasmid[i1]
    donee_plasmid_lis = individual.plasmid[i2]
    a, b = _choose_subsequence_indices(0, donor.head_length + donor.tail_length - 1, max_length=donee.head_length - 1)
    is_start, is_end = a, b + 1
    is_ = donor[is_start: is_end]
    n_i = 0
    n_j = 0
    for i in donor[: is_end]:
        if i.name == 'p_':
            n_i += 1
    for j in is_:
        if j.name == 'p_':
            n_j += 1
    is_plasmid = donor_plasmid_lis[(n_i - n_j):n_i]
    insertion_pos = random.randint(1, donee.head_length - len(is_))
    n_k = 0
    for k in donee[:insertion_pos]:
        if k.name == 'p_':
            n_k += 1
    n_l = 0
    for l in donee[insertion_pos: insertion_pos + donee.head_length - insertion_pos - len(is_)]:
        if l.name == 'p_':
            n_l += 1
    donee_plasmid_lis[:] = donee_plasmid_lis[:n_k] + is_plasmid + donee_plasmid_lis[n_k : n_k+n_l]
    donee[:] = donee[:insertion_pos] + is_ + \
               donee[insertion_pos: insertion_pos + donee.head_length - insertion_pos - len(is_)] + \
               donee[donee.head_length:]
    if _DEBUG:
        print('IS transpose: g{}[{}:{}] -> g{}[{}:]'.format(i1, is_start, is_end, i2, insertion_pos))
    return individual,

def ris_transpose(individual):
    """
    Perform RIS transposition in place

    :param individual: :class:`~geppy.core.entity.Chromosome`, a chromosome for host
    :return: a tuple of one individual

    A Root Insertion Sequence (RIS) is a segment of consecutive elements that starts with a
    function. Once an RIS is randomly chosen, it is copied and inserted at the root (first position) of a gene. Note
    that an RIS from one gene may be copied to a different gene.
    Typically, the RIS transposition rate is set around 0.1.
    """
    n_trial = 0
    while n_trial <= 2 * len(individual):
        donor, donee, i1, i2 = _choose_donor_donee(individual)
        # choose a function node randomly to start RIS
        function_indices = [i for i, p in enumerate(donor.head) if isinstance(p, gep.Function)]
        if not function_indices:  # no functions in this donor, try another
            n_trial += 1
            continue
        donor_plasmid_lis = individual.plasmid[i1]
        donee_plasmid_lis = individual.plasmid[i2]
        ris_start = random.choice(function_indices)
        # determine the length randomly
        length = random.randint(2, min(donee.head_length, donor.head_length + donor.tail_length - ris_start))
        # insert ris at the root of donee
        ris = donor[ris_start: ris_start + length]
        n_i = 0
        n_j = 0
        for i in donor[: ris_start + length]:
            if i.name == 'p_':
                n_i += 1
        for j in ris:
            if j.name == 'p_':
                n_j += 1
        ris_plasmid = donor_plasmid_lis[(n_i - n_j):n_i]
        n_k = 0
        for k in donee[0: donee.head_length - length]:
            if k.name == 'p_':
                n_k += 1
        donee_plasmid_lis[:] = ris_plasmid + donee_plasmid_lis[: n_k]
        donee[:] = ris + donee[0: donee.head_length - length] + donee[donee.head_length:]
        if _DEBUG:
            print('RIS transpose: g{}[{}:{}] -> g{}[0:]'.format(i1, ris_start, ris_start + length, i2))
        return individual,
    return individual,

def gene_transpose(individual):
    """
    Perform gene transposition in place

    :param individual: :class:`~geppy.core.entity.Chromosome`, a chromosome for host
    :return: a tuple of one individual

    An entire gene is selected randomly and exchanged with the first gene in the chromosome.
    Obviously, this operation only makes sense if the chromosome is a multigenic one. Typically, the gene transposition
    rate is set around 0.1.

    """
    if len(individual) <= 1:
        return individual,
    source = random.randint(1, len(individual) - 1)
    individual[0], individual[source] = individual[source], individual[0]
    individual.plasmid[0], individual.plasmid[source] = individual.plasmid[source], individual.plasmid[0]
    if _DEBUG:
        print('Gene transpose: g0 <-> g{}'.format(source))
    return individual,

def crossover_one_point(ind1, ind2):
    """
    Execute one-point recombination of two host individuals. The two individuals are modified in place, and the two children
    are returned.

    :param ind1: The first individual (chromosome) participating in the crossover.
    :param ind2: The second individual (chromosome) participating in the crossover.
    :return: A tuple of two children individuals.

    Note the crossover can happen at any point across the whole chromosome and thus entire genes may be also exchanged
    between the two parents if they are multigenic chromosomes.
    """
    assert len(ind1) == len(ind2)
    # the gene containing the recombination point, and the point index in the gene
    which_gene = random.randint(0, len(ind1) - 1)
    which_point = random.randint(0, len(ind1[which_gene]) - 1)
    plasmid_lis1 = ind1.plasmid
    plasmid_lis2 = ind2.plasmid
    # exchange the upstream materials
    plasmid_lis1[:which_gene], plasmid_lis2[:which_gene] = plasmid_lis2[:which_gene], plasmid_lis1[:which_gene]
    n_i = 0
    n_j = 0
    for i in ind1[which_gene][:which_point + 1]:
        if i.name == 'p_':
            n_i += 1
    for j in ind2[which_gene][:which_point + 1]:
        if j.name == 'p_':
            n_j += 1
    plasmid_lis1[which_gene][:n_i], plasmid_lis2[which_gene][:n_j] = plasmid_lis2[which_gene][:n_j], plasmid_lis1[which_gene][:n_i]

    ind1[:which_gene], ind2[:which_gene] = ind2[:which_gene], ind1[:which_gene]
    ind1[which_gene][:which_point + 1], ind2[which_gene][:which_point + 1] = \
        ind2[which_gene][:which_point + 1], ind1[which_gene][:which_point + 1]
    if _DEBUG:
        print('cxOnePoint: g{}[{}]'.format(which_gene, which_point))
    return ind1, ind2


def crossover_two_point(ind1, ind2):
    """
    Execute two-point recombination of two individuals. The two individuals are modified in place, and the two children
    are returned. The materials between two randomly chosen points are swapped to generate two children.

    :param ind1: The first individual (chromosome) participating in the crossover.
    :param ind2: The second individual (chromosome) participating in the crossover.
    :return: A tuple of two individuals.

    Note the crossover can happen at any point across the whole chromosome and thus entire genes may be also exchanged
    between the two parents if they are multigenic chromosomes.
    """
    assert len(ind1) == len(ind2)
    plasmid_lis1 = ind1.plasmid
    plasmid_lis2 = ind2.plasmid
    # the two genes containing the two recombination points
    g1, g2 = random.choices(range(len(ind1)), k=2)  # with replacement, thus g1 may be equal to g2
    if g2 < g1:
        g1, g2 = g2, g1
    # the two points in g1 and g2
    p1 = random.randint(0, len(ind1[g1]) - 1)
    p2 = random.randint(0, len(ind1[g2]) - 1)
    
    # change the materials between g1->p1 and g2->p2: first exchange entire genes, then change partial genes at g1, g2
    if g1 == g2:
        if p1 > p2:
            p1, p2 = p2, p1
        n_i ,n_j, n_k, n_l = 0, 0, 0, 0
        for i in ind1[g1][: p2+1]:
            if i.name == 'p_':
                n_i += 1
        for j in ind1[g1][p1: p2+1]:
            if j.name == 'p_':
                n_j += 1
        for k in ind2[g2][: p2+1]:
            if k.name == 'p_':
                n_k += 1
        for l in ind2[g2][p1: p2+1]:
            if l.name == 'p_':
                n_l += 1
        plasmid_lis1[g1][n_i-n_j:n_i], plasmid_lis2[g2][n_k-n_l:n_k] = plasmid_lis2[g2][n_k-n_l:n_k], plasmid_lis1[g1][n_i-n_j:n_i]
        ind1[g1][p1: p2+1], ind2[g2][p1: p2+1] = ind2[g2][p1: p2+1], ind1[g1][p1: p2+1]
    else:
        n_i ,n_j, n_k, n_l = 0, 0, 0, 0
        for i in ind1[g1][:p1]:
            if i.name == 'p_':
                n_i += 1
        for j in ind2[g1][:p1]:
            if j.name == 'p_':
                n_j += 1
        for k in ind1[g2][: p2+1]:
            if k.name == 'p_':
                n_k += 1
        for l in ind2[g2][: p2+1]:
            if l.name == 'p_':
                n_l += 1
        plasmid_lis1[g1 + 1: g2], plasmid_lis2[g1 + 1: g2] = plasmid_lis2[g1 + 1: g2], plasmid_lis1[g1 + 1: g2]
        plasmid_lis1[g1][n_i:], plasmid_lis2[g1][n_j:] = plasmid_lis2[g1][n_j:], plasmid_lis1[g1][n_i:]
        plasmid_lis1[g2][:n_k], plasmid_lis2[g2][:n_l] = plasmid_lis2[g2][:n_l], plasmid_lis1[g2][:n_k]
        ind1[g1 + 1: g2], ind2[g1 + 1: g2] = ind2[g1 + 1: g2], ind1[g1 + 1: g2]
        ind1[g1][p1:], ind2[g1][p1:] = ind2[g1][p1:], ind1[g1][p1:]
        ind1[g2][:p2 + 1], ind2[g2][:p2 + 1] = ind2[g2][:p2 + 1], ind1[g2][:p2 + 1]
    if _DEBUG:
        print('cxTwoPoint: g{}[{}], g{}[{}]'.format(g1, p1, g2, p2))
    return ind1, ind2

def crossover_gene(ind1, ind2):
    """
    Entire genes are exchanged between two parent chromosomes. The two individuals are modified in place, and the two
    children are returned.

    :param ind1: The first individual (chromosome) participating in the crossover.
    :param ind2: The second individual (chromosome) participating in the crossover.
    :return: a tuple of two children individuals

    This operation has no effect if the chromosome has only one gene. Typically, a gene recombination rate
    around 0.2 is used.
    """
    assert len(ind1) == len(ind2)
    pos1, pos2 = random.choices(range(len(ind1)), k=2)
    ind1[pos1], ind2[pos2] = ind2[pos2], ind1[pos1]
    ind1.plasmid[pos1], ind2.plasmid[pos2] = ind2.plasmid[pos2], ind1.plasmid[pos1]
    if _DEBUG:
        print('cxGene: ind1[{}] <--> ind2[{}]'.format(pos1, pos2))
    return ind1, ind2

class ExpressionTree:
    """
    Class representing an expression tree (ET) in GEP, which may be obtained by translating a K-expression, a
    gene, or a chromosome, i.e., genotype-phenotype mapping.
    """
    def __init__(self, root):
        """
        Initialize a tree with the given *root* node.

        :param root: :class:`ExpressionTree.Node`, the root node
        """
        self._root = root

    @property
    def root(self):
        """
        Get the root node of this expression tree.
        """
        return self._root

    class Node:
        """
        Class representing a node in the expression tree. Each node has a variable number of children, depending on
        the arity of the primitive at this node.
        """
        def __init__(self, name, index = None):
            self._children = []
            self._name = name
            self._index = index

        @property
        def children(self):
            """
            Get the children of this node.
            """
            return self._children

        @property
        def name(self):
            """
            Get the name (label) of this node.
            """
            return self._name

        @property
        def index(self):
            """
            Get the index (in K-expression) of this node.
            """
            return self._index

    @classmethod
    def from_genotype(cls, genome):
        """
        Create an expression tree by translating *genome*, which may be a K-expression, a gene, or a chromosome.

        :param genome: :class:`KExpression`, :class:`Gene`, or :class:`Chromosome`, the genotype of an individual
        :return: :class:`ExpressionTree`, an expression tree
        """
        if isinstance(genome, gep.Gene):
            return cls._from_kexpression(genome.kexpression)
        elif isinstance(genome, gep.KExpression):
            return cls._from_kexpression(genome)
        elif isinstance(genome, gep.Chromosome):
            if len(genome) == 1:
                return cls._from_kexpression(genome[0].kexpression)
            sub_trees = [cls._from_kexpression(gene.kexpression, i) for i, gene in enumerate(genome)]
            # combine the sub_trees with the linking function
            root = cls.Node(genome.linker.__name__)
            root.children[:] = sub_trees
            return cls(root)
        raise TypeError('Only an argument of type KExpression, Gene, and Chromosome is acceptable. The provided '
                        'genome type is {}.'.format(type(genome)))

    @classmethod
    def _from_kexpression(cls, expr, index = None):
        """
        Create an expression tree from a K-expression.

        :param expr: a K-expression
        :return: :class:`ExpressionTree`, an expression tree
        """
        if len(expr) == 0:
            return None
        # first build a node for each primitive
        if index == None:
            nodes = [cls.Node(p.name, '['+str(i)+']') for i, p in enumerate(expr)]
        else:
            nodes = [cls.Node(p.name, '['+str(index)+']['+str(i)+']') for i, p in enumerate(expr)]
        # connect each node to its children if any
        i = 0
        j = 0
        while i < len(nodes):
            for _ in range(expr[i].arity):
                j += 1
                nodes[i].children.append(nodes[j])
            i += 1
        return cls(nodes[0])


def plasmid_generate(func, host_pop):
    '''
    :param func: The plasmid individual function
    :param host_pop: The host population
    '''
    plasmid_pop = []
    p_times_lis = _count_p_functions(host_pop)
    for i in range(len(p_times_lis)):
        plasmid_pop.append([])
        for j in range(len(p_times_lis[i])):
            plasmid_pop[i].append([])
            for k in range(p_times_lis[i][j]):
                plasmid_pop[i][j].append(func())
    return plasmid_pop

def _count_p_functions(host_pop):
    p_counts_pop = []
    for ind in host_pop:
        p_counts_ind =[]
        for gene in ind:
            p_num = [symbol.name for symbol in gene].count('p_')
            p_counts_ind.append(p_num)
        p_counts_pop.append(p_counts_ind)
    return p_counts_pop



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


def _apply_modification_host(population, operator, pb):
    """
    Apply the modification given by *operator* to each individual in *population* with probability *pb* in place.
    """
    for i in range(len(population)):
        if random.random() < pb:
            population[i], = operator(population[i])
            # test_pop = [population[i]]
            # if [[len(j) for j in i.plasmid] for i in test_pop] != _count_p_functions(test_pop):
            #     print(op)
            #     raise TypeError('Fuck!')
            del population[i].fitness.values
    return population

def _apply_modification_plasmid(population, operator, pb):
    for i in range(len(population)):
        if len(population[i].plasmid) > 0:
            for plasmid_lis in population[i].plasmid:
                if len(plasmid_lis) > 0:
                    for plasmid_ind in plasmid_lis:
                        if random.random() < pb:
                            plasmid_ind, = operator(plasmid_ind)
                            del population[i].fitness.values
    return population

def _apply_crossover_host(op, population, operator, pb):
    """
    Mate the *population* in place using *operator* with probability *pb*.
    """
    for i in range(1, len(population), 2):
        if random.random() < pb:
            population[i - 1], population[i] = operator(population[i - 1], population[i])
            del population[i - 1].fitness.values
            del population[i].fitness.values
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
        
        # mutation for host
        for op in toolbox.pbs:
            if op.startswith('mut') and (not op.endswith('plasmid')):
                offspring = _apply_modification_host(offspring, getattr(toolbox, op), toolbox.pbs[op])
        
        # crossover for host
        for op in toolbox.pbs:
            if op.startswith('cx'):
                offspring = _apply_crossover_host(op, offspring, getattr(toolbox, op), toolbox.pbs[op])
        
        # %% Then, generation for plasmids
        # mutation for plasmid
        for op in toolbox.pbs:
            if op.startswith('mut') and op.endswith('plasmid'):
                offspring = _apply_modification_plasmid(offspring, getattr(toolbox, op), toolbox.pbs[op])

        # %% end of a total generation
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