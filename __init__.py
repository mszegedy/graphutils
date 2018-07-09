# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

'''Utilities for working with networkx graphs (or similar), mainly for aligning
two graphs.'''

import itertools
from functools import reduce
import networkx as nx
import consts

def partitions(ns, m, non_empty_p=True):
    '''Algorithm U from Donald Knuth's The Art of Computer Programming, Vol. 4,
    Fasc. 3B for finding all m-partitions of the iterable ns, where the minimum
    size of a partition is min_size. Copied, updated, and extended from:
    https://codereview.stackexchange.com/a/1944/173336'''

    if not non_empty_p:
        return itertools.chain(([[]]*(m-n_non_empty) + partition \
                                for partition \
                                in partitions(ns, n_non_empty,
                                              non_empty_p = True)) \
                               for n_non_empty in range(m))

    def visit(n, a):
        ps = [[] for i in range(m)]
        for j in range(n):
            ps[a[j + 1]].append(ns[j])
        return ps

    def f(mu, nu, sigma, n, a):
        if mu == 2:
            yield visit(n, a)
        else:
            for v in f(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v
        if nu == mu + 1:
            a[mu] = mu - 1
            yield visit(n, a)
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                yield visit(n, a)
        elif nu > mu + 1:
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = mu - 1
            else:
                a[mu] = mu - 1
            if (a[nu] + sigma) % 2 == 1:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v

    def b(mu, nu, sigma, n, a):
        if nu == mu + 1:
            while a[nu] < mu - 1:
                yield visit(n, a)
                a[nu] = a[nu] + 1
            yield visit(n, a)
            a[mu] = 0
        elif nu > mu + 1:
            if (a[nu] + sigma) % 2 == 1:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] < mu - 1:
                a[nu] = a[nu] + 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = 0
            else:
                a[mu] = 0
        if mu == 2:
            yield visit(n, a)
        else:
            for v in b(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v

    n = len(ns)
    a = [0] * (n + 1)
    for j in range(1, m + 1):
        a[n - m + j] = j - 1
    return f(m, n, 0, n, a)

def exact_partitions(ns, m_list):
    '''Yields all possible partitions of set ns when partitioned into sets of
    exactly the sizes given in m_list.'''
    assert sum(m_list) = len(ns)
    for permutation in itertools.permutations(ns):
        iter_permutation = iter(permutation)
        yield [list(next(iter_permutation) for _ in range(m)) \
               for m in m_list]

def default_scorefxn(node, lhs_g, rhs_g, caching=True):
    '''Give a score for an alignment of two nodes. This scorefxn gives 1 for a
    match, 0 for a non-match, and -1 for gaps.'''
    if caching and node in default_scorefxn.cache:
        return default_scorefxn.cache[node]
    ret = None
    if consts.GAP in node:
        ret = -1
    elif lhs_g[node[0]] == rhs_g[node[1]]:
        ret = 1
    else:
        ret = 0
    if caching:
        default_scorefxn.cache[node] = ret
    return ret
default_scorefxn.cache = {}

class AlignmentSet:
    '''An immutable, unordered collection of aligned nodes, where an aligned
    node is a 2-tuple of two unique node names, or one unique node name on
    either side and a consts.GAP on the other.'''
    cache = {}
    def __init__(self, parent_node, alignments):
        self.__parent_node = parent_node # name of an aligned node
        # alignments should be a collection of names of aligned nodes
        self.__alignments = frozenset(alignments)
    def __len__(self):
        return len(self.__alignments)
    def __iter__(self):
        return iter(self.__alignments)
    def __contains__(self, item):
        return item in self.__alignments
    def __hash__(self):
        return hash((self.__parent_node, self.__alignments))
    @property
    def parent_node(self):
        return self.__parent_node
    @property
    def alignments(self):
        return self.__alignments
    @property
    def lhs_children(self):
        '''Return generating lhs_children.'''
        return frozenset(a[0] for a in self.__alignments if a[0] != consts.GAP)
    @property
    def rhs_children(self):
        '''Return generating rhs_children.'''
        return frozenset(a[1] for a in self.__alignments if a[1] != consts.GAP)
    @property
    def siblings(self):
        '''All other AlignmentSets with the same lhs_children and
        rhs_children.'''
        return AlignmentSet.all_from_children(self.__parent_node,
                                              self.lhs_children,
                                              self.rhs_children) \
                           .difference({self})
    def union(self, other_as):
        '''Combine this AlignmentSet with another one and return the result.
        Note that, unlike set unions, this isn't a symmetric operation, as the
        .parent_node is inherited from self and not from other_as. Therefore,
        the AlignmentSet being called should be the one whose parent node is
        further upstream.'''
        return AlignmentSet(self.__parent_node,
                            self.__alignments.union(other_as.alignments))

class AlignedNode:
    def __init__(self, lhs_node, rhs_node, *, lhs_upstream_of=None,
                 rhs_upstream_of=None):
        self.lhs = lhs_node
        self.rhs = rhs_node
        self.lhs_upstream_of = lhs_upstream_of
        self.rhs_upstream_of = rhs_upstream_of
    def __eq__(self, other):
        assert isinstance(other, AlignedNode)
        return self.lhs == other.lhs and \
               self.rhs == other.rhs and \
               self.lhs_upstream_of == other.lhs_upstream_of and \
               self.rhs_upstream_of == other.rhs_upstream_of
    def __hash__(self):
        return hash(self.lhs, self.rhs,
                    self.lhs_upstream_of, self.rhs_upstream_of)

class Branch:
    def __init__(self, node, lhs_children, rhs_children,
                 alignments=AlignmentSet()):
        self.__node = node                 # AlignedNode
        self.__lhs_children = lhs_children # frozenset of node names
        self.__rhs_children = rhs_children # frozenset of node names
        self.__alignments   = alignments   # AlignmentSet
    @property
    def node(self):
        return self.__node
    @property
    def lhs_children(self):
        return self.__lhs_children
    @property
    def rhs_children(self):
        return self.__rhs_children
    @property
    def alignments(self):
        return self.__alignments
    def __hash__(self):
        return hash((self.__node,
                     self.__lhs_children,
                     self.__rhs_children,
                     self.__alignments))

class TreeNode:
    '''Node for the alignment tree.'''
    def __init__(self, parent_branch, alignment_set, parent_tree_node):
        # branch whose expansion led to this node:
        self.__parent_branch = parent_branch
        # alignment set from the different possible alignments of the parent
        # branch that (uniquely) maps to this node:
        self.__alignment_set = alignment_set
        # union of alignments of the entire ancestry of this node:
        self.__total_alignment_set = parent_tree_node.total_alignment_set \
                                                     .union(alignment_set)
    @property
    def parent_branch(self):
        return self.__parent_branch
    @property
    def alignment_set(self):
        return self.__alignment_set
    @property
    def total_alignment_set(self):
        return self.__total_alignment_set
    def __hash__(self):
        return hash(self.__parent_branch,
                    self.__alignment_set,
                    self.__total_alignment_set)

class AlignmentTree:
    def __init__(self, lhs_g, rhs_g, start_node, scorefxn=default_scorefxn):
        '''Constructs and scores an entire alignment tree between two graphs,
        for a specified starting aligned node, and a given scorefxn.'''
        self.lhs_g = lhs_g
        self.rhs_g = rhs_g
        self.scorefxn = scorefxn
        # maps TreeNodes to a tuple of scores for each, where the tuple lists
        # the scores of the TreeNode's parents in order of descent, ending with
        # the score of the TreeNode itself
        self.tree_node_scores = {}
    def tree_nodes_from_branch(self, branch):
        '''Returns the set of daughter tree nodes from a branch, each having the
        parent that gives it the highest score.'''
        ## copied from above; needs to be rewritten completely, partially to be
        ## magic and always know the parents that give the highest score

        ## trivial cases
        key = (parent_alignment,
               frozenset(lhs_children),
               frozenset(rhs_children))
        if key in cls.cache:
            return cls.cache[key]
        if len(lhs_children) == 0 and len(rhs_children) == 0:
            cls.cache[key] = frozenset()
            return cls.cache[key]
        if len(lhs_children) == 0:
            cls.cache[key] = frozenset({
                AlignmentSet(parent_node,
                             frozenset((consts.GAP, node) \
                                       for node in rhs_children))})
            return cls.cache[key]
        if len(rhs_children) == 0:
            cls.cache[key] = frozenset({
                AlignmentSet(parent_node,
                             frozenset((node, consts.GAP) \
                                       for node in lhs_children))})
            return cls.cache[key]

        ## reorder sets into larger and smaller (can be equal size)
        if len(lhs_children) >= len(rhs_children):
            l_set  = lhs_children # larger set
            s_set  = rhs_children # smaller set
            switch = lambda x,y: (x,y) # switch back to lhs, rhs
        else:
            l_set  = rhs_children # larger set
            s_set  = lhs_children # smaller set
            switch = lambda x,y: (y,x) # switch back to lhs, rhs
        ## construct and collect every possible alignment set
        alignment_sets = {}
        # every loop variable:
        #     matched_s_set:   s_set nodes aligned to l_set nodes
        #     us_gapped_s_set: s_set nodes aligned to gaps upstream from l_set
        #         nodes
        #     ds_gapped_s_set: s_set nodes aligned to gaps downstream from
        #         parent node
        #     unmatched_s_set: s_set nodes not aligned to anything, because
        #         l_set nodes were aligned to gaps upstream from them
        # l_set loop variables correspond to the same things for l_set.
        for matched_s_set, us_gapped_s_set, ds_gapped_s_set, unmatched_s_set \
        in itertools.chain \
                    .from_iterable(itertools.permutations(partition) \
                                   for partition \
                                   in partitions(s_set, 4, non_empty_p=False)):
            for matched_l_set, us_gapped_l_set, ds_gapped_l_set,
                unmatched_l_set \
            in exact_partitions(l_set, (len(matched_s_set),
                                        len(unmatched_s_set),
                                        len(l_set)-(len(matched_s_set) + \
                                                    len(unmatched_s_set) + \
                                                    len(us_gapped_s_set)),
                                        len(us_gapped_s_set))):
                alignments = {}
                for lhs_node, rhs_node in zip(*switch(matched_l_set,
                                                      matched_s_set)):
                    alignments.add(switch(lhs_node, rhs_node))
                # dunno what to do with unaligned nodes yet
                for aligned_node, unaligned_node in zip(us_gapped_l_set,
                                                        unmatched_s_set):
                    alignments.add(switch(aligned_node, None))
                for aligned_node, unaligned_node in zip(us_gapped_s_set,
                                                        unmatched_l_set):
                    alignments.add(switch(None, aligned_node))
                ds_gapped_lhs_set, ds_gapped_rhs_set = \
                    switch(ds_gapped_l_set, ds_gapped_s_set)
                for node in ds_gapped_lhs_set:
                    alignments.add((node, None))
                for node in ds_gapped_rhs_set:
                    alignments.add((None, node))
                alignment_sets.add(AlignmentSet(parent_node, alignments))
        # key was calculated at the beginning
        cls.cache[key] = frozenset(alignment_sets)
        return cls.cache[key]
