import networkx as nx
import sys
import codon

from time import time

def graph_to_codon_repr(G):
    ref = {}
    H = []

    def node_ref(n, ref):
        x = ref.get(n, -1)
        if x == -1:
            x = len(ref)
            ref[n] = x
        return x

    for e in G.edges():
        k = node_ref(e[0], ref)
        v = node_ref(e[1], ref)

        while len(H) <= max(k, v):
            H.append([])

        H[k].append(v)
        H[v].append(k)

    return H

@codon.jit
def betweenness_centrality(G):
    def _single_source_shortest_path_basic(G, s):
        S = []
        P = {}
        for v in range(len(G)): #G:
            P[v] = []
        sigma = dict.fromkeys(range(len(G)), 0.0) #dict.fromkeys(G, 0.0)  # sigma[v]=0 for v in G
        D = {}
        sigma[s] = 1.0
        D[s] = 0
        Q = deque([s])
        while Q:  # use BFS to find shortest paths
            v = Q.popleft()
            S.append(v)
            Dv = D[v]
            sigmav = sigma[v]
            for w in G[v]:
                if w not in D:
                    Q.append(w)
                    D[w] = Dv + 1
                if D[w] == Dv + 1:  # this is a shortest path, count paths
                    sigma[w] += sigmav
                    P[w].append(v)  # predecessors
        return S, P, sigma, D

    def _accumulate_basic(betweenness, S, P, sigma, s):
        delta = dict.fromkeys(S, 0.)
        while S:
            w = S.pop()
            coeff = (1 + delta[w]) / sigma[w]
            for v in P[w]:
                delta[v] += sigma[v] * coeff
            if w != s:
                betweenness[w] += delta[w]
        return betweenness, delta

    def _rescale(betweenness, n, normalized, directed=False, k: Optional[float] = None, endpoints=False):
        if normalized:
            if endpoints:
                if n < 2:
                    scale = None  # no normalization
                else:
                    # Scale factor should include endpoint nodes
                    scale = 1 / (n * (n - 1))
            elif n <= 2:
                scale = None  # no normalization b=0 for all nodes
            else:
                scale = 1 / ((n - 1) * (n - 2))
        else:  # rescale by 2 for undirected graphs
            if not directed:
                scale = 0.5
            else:
                scale = None
        if scale is not None:
            if k is not None:
                scale = scale * n / k
            for v in betweenness:
                betweenness[v] *= scale
        return betweenness

    betweenness = dict.fromkeys(range(len(G)), 0.0)  # b[v]=0 for v in G
    for s in range(len(G)):
        S, P, sigma, _ = _single_source_shortest_path_basic(G, s)
        betweenness, _ = _accumulate_basic(betweenness, S, P, sigma, s)
    betweenness = _rescale(betweenness, len(G), normalized=True)
    return betweenness

t0 = time()
g = nx.read_edgelist(sys.argv[1], delimiter="\t", nodetype=str, create_using=nx.Graph)
t1 = time()
print('read took', t1 - t0)

t0 = time()
h = graph_to_codon_repr(g)
t1 = time()
print('conv took', t1 - t0)

t0 = time()
bc = betweenness_centrality(h)
t1 = time()
print('bc took', t1 - t0)
