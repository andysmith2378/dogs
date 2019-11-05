import networkx

class RepresentedByName(object):
    __repr__ = lambda x: x.name

    @staticmethod
    def getES(block, graphFromWhichToDrawReturnedEdges):  # returns a set of edges
        if isinstance(block, set):
            result = set()
            [result.update(graphFromWhichToDrawReturnedEdges.edges(location)) for location in block]
            return result
        return graphFromWhichToDrawReturnedEdges.edges(block)

class NameEqualsClassName(object):
    haste       = 1.02                # bias in favour of moving own dogs towards the middle instead of other dogs away
    movePenalty = (0, 0.3, 0.4, 0.4)  # extra bias against 0, 1, 2 or 3 dogs moving
    midPenalty  = (0, 0, 0.2, 0.6)    # extra bias against 0, 1, 2 or 3 dogs occupying middle spaces
    goalSpaces  = ('a1', 'a2', 'a3')  # locations of the middle spaces
    wKey, bW    = 'weight', 3         # bW sets the edge weight of all the edges that connect to an occupied space
    bigNumber   = 1000000000
    smallNumber = -bigNumber
    __init__    = lambda x: setattr(x, 'name', x.__class__.__name__)

    @staticmethod
    def adjustWeight(graphWhoseEdgesWeightsToAdjust, oldWeight=1, newWeight=0):
        for left, right in graphWhoseEdgesWeightsToAdjust.edges:
            if graphWhoseEdgesWeightsToAdjust.edges[left, right][NameEqualsClassName.wKey] == oldWeight:
                graphWhoseEdgesWeightsToAdjust.edges[left, right][NameEqualsClassName.wKey] = newWeight
        return graphWhoseEdgesWeightsToAdjust

    @staticmethod
    def getEmptyGraph(setOfNodes, openEdges):
        result = networkx.Graph()
        result.add_nodes_from(setOfNodes)
        result.add_weighted_edges_from(openEdges)
        return result