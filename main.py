import copy, itertools, multiprocessing as mp, networkx as netx
from board import edges, nodes
from graphhelpers import RepresentedByName, NameEqualsClassName

class Dog(NameEqualsClassName, RepresentedByName):
    basePenalty = 0.3  # bias against dogs of this class moving or occupying the middle spaces
    __call__    = lambda x, y, z: set([loc for loc in y[x.loc].keys() if loc < x.loc]).difference(x.getBlockers(z, y))
    filterMoves = lambda x, y, z: y
    block       = lambda x, y, z: x.loc
    setEdgeW    = lambda x, y, z: [z.edges[l, r].__setitem__(Dog.wKey, Dog.bW) for l, r in Dog.getES(x.block(z, y), z)]
    retreats    = lambda w, x, y, z: z.retreatFrom(x, y, w)
    retreatFrom = lambda w, x, y, z: set(x[z.loc].keys()).difference(z.getBlockers(y, x))
    myG         = lambda v, w, x, y, z: w
    _setLoc     = lambda x, y: setattr(x, '_locations', {mp.current_process().pid: y})

    def __init__(self, initialLocation):
        NameEqualsClassName.__init__(self)
        self._setLoc(initialLocation)
        self._defaultLocation = initialLocation

    def addDogsICanBarkAt(self, target, othI, barkGraph, myI):
        [target.append((myI, othI), ) for othI, r in enumerate(othI) if self is not r and self.loc in barkGraph[r.loc]]

    def getBlockers(self, listOfAllDogs, blockGraph):
        result = Kennel()
        [result(dog.block(blockGraph, listOfAllDogs)) for dog in listOfAllDogs]
        return result

    def _findShortest(self, listOfAllDogs, graphToUse):
        if self.loc in NameEqualsClassName.goalSpaces:
            return 0
        return self.__class__._updateShortest(listOfAllDogs, graphToUse, self)

    def _countGoalsAndMoves(self, score, goalCount, moveCount):
        if self.loc in NameEqualsClassName.goalSpaces:
            score     += self.__class__.basePenalty
            goalCount += 1
        elif self.__class__.basePenalty > 0 and self(graph, allD):
            score     += self.__class__.basePenalty
            moveCount += 1
        return score, goalCount, moveCount

    def _getLocation(self):     # this may amount to paranoia
        processID = mp.current_process().pid
        if processID in self._locations:
            return self._locations[processID]
        return self._defaultLocation
    loc = property(_getLocation, _setLoc)

    @staticmethod
    def getGraph(listOfAllTheDogs, setOfNodes, openEdges):
        result = NameEqualsClassName.getEmptyGraph(setOfNodes, openEdges)
        [dog.setEdgeW(listOfAllTheDogs, result) for dog in listOfAllTheDogs]
        return result

    @staticmethod
    def getScore(allDgsList, myDgsList, otherDgsList, nodeSet, edgeSet):
        bestSSc, sGraph = Dog.contribMine(allDgsList, edgeSet, myDgsList, nodeSet, otherDgsList)
        return Dog._addPenalties(Dog.haste * Dog.contribOther(bestSSc, sGraph, otherDgsList, allDgsList), otherDgsList)

    @staticmethod
    def getMoves(listOfAllTheDogs):
        result = list(itertools.permutations(range(len(listOfAllTheDogs)), 3))
        for dog in listOfAllTheDogs:
            result = dog.filterMoves(result, listOfAllTheDogs)
        return result

    @staticmethod
    def contribOther(bestScore, contributionGraph, listOfOtherDogs, listOfAllDogs):
        for otherDog in listOfOtherDogs:
            bestScore -= otherDog._findShortest(listOfAllDogs, contributionGraph)
        return bestScore

    @staticmethod
    def contribMine(allDogs, eSet, myD, nSet, othD):
        return Dog._scoreL(eSet, [dog.loc for dog in myD], myD, nSet, othD), Dog.getGraph(allDogs, nSet, eSet)

    @staticmethod
    def fewestMoves(bestScore, eSet, moveOrder, nSet, otherDogList, targetList, targetMap):
        score, originalLocations = 0, {}
        for indx in moveOrder:
            score = Dog._checkPath(eSet, indx, nSet, originalLocations, otherDogList, score, targetList, targetMap)
        return Dog._sendDogsBackWhereTheyCameFrom(bestScore, moveOrder, originalLocations, score, targetMap)

    @staticmethod
    def assessBark(bark):       # also uses allD
        leftInd, rightInd, retreat = bark
        left, right                = allD[leftInd], allD[rightInd]
        originalL, originalR       = left.loc, right.loc
        left.loc                   = right.loc
        right.loc                  = retreat
        score, left.loc, right.loc = Dog.getScore(allD, myDogs, otherDogs, nodes, edges), originalL, originalR
        return score, (leftInd, rightInd, retreat)

    @staticmethod
    def assessMove(moveIndices):    # also uses allD and graph
        ind1, ind2, ind3 = moveIndices
        dog1, dog2, dog3 = allD[ind1], allD[ind2], allD[ind3]
        ori1, ori2, ori3 = dog1.loc, dog2.loc, dog3.loc
        bestS            = None
        bestSSc          = NameEqualsClassName.bigNumber
        for d1 in dog1(graph, allD):
            bestS, bestSSc = Dog._firstDog(bestS, bestSSc, d1, dog1, dog2, dog3, ind1, ind2, ind3, ori1, ori2, ori3)
        return bestSSc, bestS

    @staticmethod
    def getBarkOptions(allDg, grh):
        result, adjacentPairs = [], []
        [left.addDogsICanBarkAt(adjacentPairs, allDg, grh, leftIndex) for leftIndex, left in enumerate(allDg)]
        [result.extend([(lI, rI, rt) for rt in allDg[rI].retreats(grh, allDg, allDg[lI])]) for lI, rI in adjacentPairs]
        return result

    @staticmethod
    def _sendDogsBackWhereTheyCameFrom(bestScore, moveOrder, originalLocations, score, targetMap):
        if score < bestScore:
            bestScore = score
        [setattr(targetMap[indx], 'loc', originalLocations[indx]) for indx in moveOrder]
        return bestScore

    @staticmethod
    def _updateBestS(bestS, bestSSc, destination1, destination2, destination3, dog1Index, dog2Index, dog3Index):
        score = Dog.getScore(allD, myDogs, otherDogs, nodes, edges)
        if score < bestSSc:
            bestSSc = score
            bestS   = ((dog1Index, destination1), (dog2Index, destination2), (dog3Index, destination3),)
        return bestS, bestSSc

    @staticmethod
    def _findBestScore(setOfEdges, listOfMyDogs, setOfNodes, listOfOtherDogs):
        result = NameEqualsClassName.bigNumber
        for targetMap in itertools.permutations(listOfMyDogs):
            result = Dog._shortCut(setOfEdges, setOfNodes, listOfOtherDogs, result, targetMap)
        return result

    @staticmethod
    def _checkPath(es, indx, ns, originalLocations, otherDogs, score, targetList, targetMap):
        myDog, tLoc                   = targetMap[indx], ''.join(['a', str(indx + 1)])
        originalLocations[indx], notD = myDog.loc, otherDogs + targetList[:indx] + targetList[indx + 1:]
        if myDog.loc != tLoc:
            score += 1 + netx.astar_path_length(myDog.myG(Dog.getGraph(notD, ns, es), notD, ns, es), myDog.loc, tLoc)
        myDog.loc = tLoc
        return score

    @staticmethod
    def _addPenalties(bestSSc, listOfOtherDogs):
        goals, moves = 0, 0
        for dog in listOfOtherDogs:
            bestSSc, goals, moves  = dog._countGoalsAndMoves(bestSSc, goals, moves)
            bestSSc               += NameEqualsClassName.midPenalty[goals] + NameEqualsClassName.movePenalty[moves]
        return bestSSc

    @staticmethod
    def _shortCut(edgeSet, nodeSet, otherDogList, result, targetMap):
        targetList = list(targetMap)
        for moveOrder in itertools.permutations((0, 1, 2), ):
            result = Dog.fewestMoves(result, edgeSet, moveOrder, nodeSet, otherDogList, targetList, targetMap)
        return result

    @staticmethod
    def _firstDog(bestS, bestSSc, d1, dog1, dog2, dog3, ind1, ind2, ind3, ori1, ori2, ori3):
        dog1.loc = d1
        for d2 in dog2(graph, allD):
            bestS, bestSSc = Dog._middleDog(bestS, bestSSc, d1, d2, dog2, dog3, ind1, ind2, ind3, ori2, ori3)
        dog1.loc = ori1
        return bestS, bestSSc

    @staticmethod
    def _middleDog(bestS, bestSSc, d1, d2, dog2, dog3, ind1, ind2, ind3, ori2, ori3):
        dog2.loc = d2
        for d3 in dog3(graph, allD):
            bestS, bestSSc = Dog._lastDog(bestS, bestSSc, d1, d2, d3, ind1, ind2, dog3, ind3, ori3)
        dog2.loc = ori2
        return bestS, bestSSc

    @staticmethod
    def _lastDog(bestS, bestSSc, dest1, dest2, dest3, dog1Ind, dog2Ind, dog3, dog3Ind, original3):
        dog3.loc = dest3
        bestS, bestSSc = Dog._updateBestS(bestS, bestSSc, dest1, dest2, dest3, dog1Ind, dog2Ind, dog3Ind)
        dog3.loc = original3
        return bestS, bestSSc

    @staticmethod
    def _scoreL(edgeSet, locations, myDogList, nodeSet, otherDogList):
        if 'a1' in locations and 'a2' in locations and 'a3' in locations:
            return NameEqualsClassName.smallNumber
        return Dog._findBestScore(edgeSet, myDogList, nodeSet, otherDogList)

    @staticmethod
    def _updateShortest(allDogs, grph, otherDog):
        result = NameEqualsClassName.bigNumber
        for target in NameEqualsClassName.goalSpaces:
            score = 1 + netx.astar_path_length(otherDog.myG(grph, allDogs, nodes, edges), otherDog.loc, target)
            if score < result:
                result = score
        return result

class Kennel(set):
    logfile   = 'gamelog.txt'
    statefile = 'currentstate.txt'
    delim     = ', '

    def __call__(self, other):
        if isinstance(other, set):
            return self.update(other)
        return self.add(other)

    @staticmethod
    def survey(assess, possible, poolSize, sortWith=lambda x: x[0]):
        if possible:
            with mp.Pool(poolSize) as pool:
                return sorted(pool.map_async(assess, possible).get(), key=sortWith)[0]
        return NameEqualsClassName.bigNumber, None

    @staticmethod
    def moveAll(move):
        allD[move[0]].loc = move[1]
        return f'{allD[move[0]]} to {move[1]}'

    @staticmethod
    def reap():
        with open(Kennel.statefile, 'r') as gamestate:
            return [Kennel._readline(fileline) for fileline in gamestate.readlines()]

    @staticmethod
    def sow(listOfTextLines, mD, oD, victoryText='computer declares victory'):
        with open(Kennel.statefile, 'w+') as state:
            [state.writelines([Kennel.delim.join([d.__class__.__name__, d.loc, "\n"]) for d in ds]) for ds in (mD, oD)]
        if mD[0].loc in Dog.goalSpaces and mD[1].loc in Dog.goalSpaces and mD[2].loc in Dog.goalSpaces:
            listOfTextLines.append(victoryText)
        with open(Kennel.logfile, 'a+') as gamelog:
            gamelog.writelines(["".join([line, "\n"]) for line in listOfTextLines])
            gamelog.writelines(['\n'],)
        print("\n".join(listOfTextLines))

    @staticmethod
    def _readline(line):
        dogclass, doglocation, _ = line.split(Kennel.delim)
        return globals()[dogclass](doglocation)

class Frank(Dog):
    basePenalty = 0.35

    def addDogsICanBarkAt(self, target, othI, barkGraph, myI):
        pass        # Frank can't bark

class Pippa(Dog):
    basePenalty = 0

    def __call__(self, grph, listOfAllDogs):
        return set(grph).difference(self.getBlockers(listOfAllDogs, grph))  # Pippa can move to any empty space

    def myG(self, originalGraph, listOfAllDogs, nodeSet, edgeSet):
        return NameEqualsClassName.adjustWeight(copy.deepcopy(originalGraph))

class Pepe(Dog):
    basePenalty = 0.29

    basket = 'c5'

    def retreats(self, grph, listOfAllDogs, barkingDog):
        for dog in listOfAllDogs:
            if dog.loc == Pepe.basket:
                return Dog.retreats(self, grph, listOfAllDogs, barkingDog)
        return {Pepe.basket}      # Pepe retreats to his basket if unoccupied

class Muffin(Dog):
    basePenalty = 0.4

    def filterMoves(self, indexList, dogList):
        for indices in indexList:
            dogClasses = [dogList[indx].__class__ for indx in indices]
            while dogClasses.count(self.__class__) > 1:
                indexList.remove(indices)
        return indexList    # Muffin can only move once each turn

    def myG(self, graph, listOfAllDogs, nodeSet, edgeSet):
        return NameEqualsClassName.adjustWeight(copy.deepcopy(graph), 1, 2)

class Lucy(Dog):
    basePenalty = 0.28

    def __call__(self, grph, listOfAllDogs):
        return set(grph[self.loc].keys()).difference(self.getBlockers(listOfAllDogs, grph))

class Chico(Dog):
    basePenalty = 0.33

    def addDogsICanBarkAt(self, t, oI, bG, mI):
        [t.append((mI, i), ) for i, r in enumerate(oI) if self is not r and self.loc in bG[r.loc] and self.loc > r.loc]

class Banjo(Dog):
    basePenalty = 0.32

    def retreatFrom(self, retreatGraph, listOfAllDogs, target):
        for dog in listOfAllDogs:
            if dog.loc == Pepe.basket:
                return Dog.retreatFrom(self, retreatGraph, listOfAllDogs, target)
        return {Pepe.basket}

class Coco(Dog):
    basePenalty = 0.31

    def block(self, blockGraph, listOfAllDogs):
        result = {self.loc}
        for location in blockGraph[self.loc].keys():
            if location < self.loc:
                result.add(location)
        return result   # Dogs can't move to spaces next to Coco higher than the one she occupies

class Darcy(Dog):
    basePenalty = 0.34

    smallDogs = (Coco, Lucy, Pepe, Banjo)

    def retreats(self, retreatGraph, listOfAllDogs, barkingDog):
        if barkingDog.__class__ in Darcy.smallDogs:
            return {self.loc}  # Darcy doesn't retreat from small dogs
        return Dog.retreats(self, retreatGraph, listOfAllDogs, barkingDog)

try:
    readstate         = Kennel.reap()
    myDogs, otherDogs = readstate[:3], readstate[3:]
except (EOFError, FileNotFoundError):
    myDogs, otherDogs = [Muffin('e1'), Frank('e4'), Darcy('e5')], [Pippa('e2'), Pepe('e3'), Coco('e6')]
my1, my2, my3 = myDogs
ot4, ot5, ot6 = otherDogs
allD, graph   = (my1, my2, my3, ot4, ot5, ot6,), NameEqualsClassName.getEmptyGraph(nodes, edges)
possibleBarks = Dog.getBarkOptions(allD, graph)
possibleMoves = Dog.getMoves(allD)

if __name__ == '__main__':
    cpuCount       = mp.cpu_count()
    bestBSc, bestB = Kennel.survey(Dog.assessBark, possibleBarks, cpuCount)
    bestMSc, bestM = Kennel.survey(Dog.assessMove, possibleMoves, cpuCount)
    if bestMSc < bestBSc:
        Kennel.sow([Kennel.moveAll(mv) for mv in bestM], myDogs, otherDogs)
    else:
        allD[bestB[0]].loc = allD[bestB[1]].loc
        allD[bestB[1]].loc = bestB[2]
        Kennel.sow([f'{allD[bestB[0]]} barks at {allD[bestB[1]]}',
                    f'{allD[bestB[1]]} retreats to {bestB[2]}'], myDogs, otherDogs)