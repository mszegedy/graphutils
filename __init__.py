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
import queue
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

def tree_from_graph(g, start_node):
    '''Make a minimum depth, tree-shaped digraph out of a normal graph, with
    start_node as the root.'''
    # apparently recursion is inefficient in Python, so we're gonna
    # take the long and boring route with a list accumulator
    tree = nx.DiGraph()
    tree.add_node(start_node, **g[start_node])
    # using queue.Queue here because I might wanna parallelize in the
    # future
    to_expand = queue.Queue()
    to_expand.put(start_node)
    while to_expand.qsize() != 0:
        node = to_expand.get()
        for child in g[node]:
            if child not in tree:
                tree.add_node(child, **g[child])
                tree.add_edge(node, child)
                to_expand.put(child)
        to_expand.task_done()
    return tree

def default_scorefxn(node, lhs_g, rhs_g, caching=True):
    '''Give a score for an alignment of two nodes. This scorefxn gives 1 for a
    match, 0 for a non-match, and -1 for gaps.'''
    if caching and node in default_scorefxn.cache:
        return default_scorefxn.cache[node]
    ret = None
    if consts.GAP in node:
        ret = -1
    elif lhs_g[node.lhs] == rhs_g[node.rhs]:
        ret = 1
    else:
        ret = 0
    if caching:
        default_scorefxn.cache[node] = ret
    return ret
default_scorefxn.cache = {}

class AlignmentSet:
    '''An immutable, unordered collection of AlignedNodes.'''
    cache = {}
    def __init__(self, alignments):
        # alignments should be a collection of names of aligned nodes
        self.__alignments = frozenset(alignments)
    def __eq__(self, other):
        return self.__alignments == other.alignments
    def __len__(self):
        return len(self.__alignments)
    def __iter__(self):
        return iter(self.__alignments)
    def __contains__(self, item):
        return item in self.__alignments
    def __hash__(self):
        return hash(self.__alignments)
    @property
    def alignments(self):
        return self.__alignments
    @property
    def lhs_children(self):
        '''Return generating lhs_children.'''
        return frozenset(a.lhs for a in self.__alignments if a.lhs != consts.GAP)
    @property
    def rhs_children(self):
        '''Return generating rhs_children.'''
        return frozenset(a.rhs for a in self.__alignments if a.rhs != consts.GAP)
    @property
    def siblings(self):
        '''All other AlignmentSets with the same lhs_children and
        rhs_children.'''
        return AlignmentSet.all_from_children(self.__parent_node,
                                              self.lhs_children,
                                              self.rhs_children) \
                           .difference({self})
    def is_sibling(self, other):
        return self.lhs_children == other.lhs_children and \
               self.rhs_children == other.rhs_children
    def union(self, other_as):
        '''Combine this AlignmentSet with another one and return the result.
        Note that, unlike set unions, this isn't a symmetric operation, as the
        .parent_node is inherited from self and not from other_as. Therefore,
        the AlignmentSet being called should be the one whose parent node is
        further upstream.'''
        return AlignmentSet(self.__parent_node,
                            self.__alignments.union(other_as.alignments))

class AlignedNode:
    '''An atomic pair of two aligned nodes: one from the left-hand-side graph,
    and one from the right-hand-side graph. One is permitted to be consts.GAP.
    Additionally, the gap is permitted to be given a node that it is upstream
    of.'''
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
    '''Node for the alignment tree, used as a key to map pieces of alignments
    to branches (which then map to more pieces of alignments, etc).'''
    def __init__(self, parent_branch, alignment_set, parent_tree_node, score):
        # branch whose expansion led to this node:
        self.__parent_branch = parent_branch
        # alignment set from the different possible alignments of the parent
        # branch that (uniquely) maps to this node:
        self.__alignment_set = alignment_set
        # union of alignments of the entire ancestry of this node:
        self.__total_alignment_set = parent_tree_node.total_alignment_set \
                                                     .union(alignment_set)
        self.__scores = parent_tree_node.scores + (score,)
    @property
    def parent_branch(self):
        return self.__parent_branch
    @property
    def alignment_set(self):
        return self.__alignment_set
    @property
    def total_alignment_set(self):
        return self.__total_alignment_set
    @property
    def scores(self):
        return self.__scores
    def __hash__(self):
        return hash((self.__parent_branch,
                     self.__alignment_set,
                     self.__total_alignment_set))

class StartingTreeNode(TreeNode):
    '''Root node for alignment tree.'''
    def __init__(self, start_node):
        self.__parent_branch = None
        self.__alignment_set = AlignmentSet({start_node})
        self.__total_alignment_set = self.__alignment_set
        self.__scores = ()

class AlignmentTree:
    def __init__(self, lhs_g, rhs_g, lhs_start, rhs_start,
                 scorefxn=default_scorefxn):
        '''Constructs and scores an entire alignment tree between two graphs,
        for a specified starting aligned node, and a given scorefxn.'''
        self.lhs_g = lhs_g
        self.rhs_g = rhs_g
        self.start_node = AlignNode(lhs_start, rhs_start)
        self.scorefxn = scorefxn
        self.root = StartingTreeNode(self.start_node)
        self.branches_to_nodes = {} # Branch -> {AlignmentSet -> TreeNode}
        self.nodes_to_branches = {} # TreeNode -> {AlignedNode -> Branch}
    def expand_node(self, node):
        '''Adds the daughter branches of a particular node to
        .nodes_to_branches, and then the daughter nodes of those branches to
        .branches_to_nodes. Returns the daughter nodes.'''
        ### construct .nodes_to_branches (both levels)
        try:
            branch_dict = self.nodes_to_branches[node]
        except KeyError:
            branch_dict = {}
            for aligned_node in node.alignments:
                lhs_successors = self.lhs_tree.successors(aligned_node.lhs)
                rhs_successors = self.rhs_tree.successors(aligned_node.rhs)
                branch_dict[aligned_node] = \
                    Branch(aligned_node,
                           lhs_successors,
                           rhs_successors,
                           AlignmentSet(
                               {alignment \
                                for alignment \
                                in node.total_alignments \
                                if (alignment.lhs in lhs_successors) or \
                                   (alignment.rhs in rhs_successors)}))
            self.nodes_to_branches[node] = branch_dict
        ### construct .branches_to_nodes (both levels)
        for branch in branch_dict.values():
            ## reorder sets into larger and smaller (can be equal size)
            lhs_children = branch.lhs_children
            rhs_children = branch.rhs_children
            if len(lhs_children) >= len(rhs_children):
                l_set  = lhs_children # larger set
                s_set  = rhs_children # smaller set
                switch = lambda x,y: (x,y) # switch back to lhs, rhs
            else:
                l_set  = rhs_children # larger set
                s_set  = lhs_children # smaller set
                switch = lambda x,y: (y,x) # switch back to lhs, rhs
            ## construct and collect every possible AlignmentSet
            alignment_sets = set()
            # every loop variable:
            #     matched_s_set:   s_set nodes aligned to l_set nodes
            #     us_gapped_s_set: s_set nodes aligned to gaps upstream from l_set
            #         nodes
            #     ds_gapped_s_set: s_set nodes aligned to gaps downstream from
            #         parent node
            #     unmatched_s_set: s_set nodes not aligned to anything, because
            #         l_set nodes were aligned to gaps upstream from them
            # l_set loop variables correspond to the same things for l_set.
            for matched_s_set, us_gapped_s_set, ds_gapped_s_set,
                unmatched_s_set \
            in itertools.chain \
                        .from_iterable(itertools.permutations(partition) \
                                       for partition \
                                       in partitions(s_set, 4,
                                                     non_empty_p=False)):
                for matched_l_set, us_gapped_l_set, ds_gapped_l_set,
                    unmatched_l_set \
                in exact_partitions(l_set, (len(matched_s_set),
                                            len(unmatched_s_set),
                                            len(l_set)-(len(matched_s_set) + \
                                                        len(unmatched_s_set) + \
                                                        len(us_gapped_s_set)),
                                            len(us_gapped_s_set))):
                    alignments = set()
                    for lhs_node, rhs_node in zip(*switch(matched_l_set,
                                                          matched_s_set)):
                        alignments.add(AlignedNode(lhs_node, rhs_node))
                    # dunno what to do with unaligned nodes yet
                    for gapped_node, unmatched_node in zip(us_gapped_l_set,
                                                           unmatched_s_set):
                        lhs_upstream_from, rhs_upstream_from = \
                            switch(None, unmatched_node)
                        alignments.add(
                            AlignedNode(*switch(gapped_node, consts.GAP),
                                        lhs_upstream_from=lhs_upstream_from,
                                        rhs_upstream_from=rhs_upstream_from))
                    for gapped_node, unmatched_node in zip(us_gapped_s_set,
                                                           unmatched_l_set):
                        lhs_upstream_from, rhs_upstream_from = \
                            switch(unmatched_node, None)
                        alignments.add(
                            AlignedNode(*switch(consts.GAP, gapped_node),
                                        lhs_upstream_from=lhs_upstream_from,
                                        rhs_upstream_from=rhs_upstream_from))
                    ds_gapped_lhs_set, ds_gapped_rhs_set = \
                        switch(ds_gapped_l_set, ds_gapped_s_set)
                    for node in ds_gapped_lhs_set:
                        alignments.add((node, consts.GAP))
                    for node in ds_gapped_rhs_set:
                        alignments.add((consts.GAP, node))
                    alignment_sets.add(AlignmentSet(alignments))
            ## for each AlignmentSet, 
    @property
    def lhs_tree(self):
        if hasattr(self, '__lhs_tree'):
            return self.__lhs_tree
        else:
            self.__lhs_tree = tree_from_graph(self.lhs_g, self.start_node.lhs)
            return self.__lhs_tree
    @property
    def rhs_tree(self):
        if hasattr(self, '__rhs_tree'):
            return self.__rhs_tree
        else:
            self.__rhs_tree = tree_from_graph(self.rhs_g, self.start_node.rhs)
            return self.__rhs_tree
    def alignment(self):
        ## 0. build tree through recursive expansion over each level of each
        ##    tree
        # because Python apparently sucks at recursion we're stuck with a good
        # old-fashioned queue again
        to_expand = queue.Queue()
        to_expand.put(self.root)
        while to_expand.qsize() != 0:
            node = to_expand.get()
            expansions = self.expand_node(node) # write this later
            for child in expansions:
                to_expand.put(child)
            to_expand.task_done()
        ## 1. get leaf with highest score
        best_leaf = max(self.leaves, key=lambda n: sum(n.scores))
        ## 2. collapse tree upwards, taking the highest-scoring expansions at
        ##    each node
        to_collapse = queue.Queue()
        to_collapse.put(best_leaf)
        current_highest_node = best_leaf
        total_alignment = set()
        while to_expand.qsize() != 0:
            node = to_collapse.get()
            for sibling in (sibling \
                            for sibling \
                            in self.branches_to_nodes[node.parent_branch] \
                            if sibling != node):
                pass
            to_collapse.task_done()
