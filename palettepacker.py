# palettepacker.py
# Cooper Harasyn (cooprocks123e) Sept. 2022
# public domain

# References:
# https://en.wikipedia.org/wiki/Simulated_annealing
# https://www.mathworks.com/help/gads/how-simulated-annealing-works.html

from collections import defaultdict
from functools import reduce, lru_cache
from math import exp
from random import Random

MAX_COLOURS = 15
MAX_PALETTES = 6
class _AnnealingParameters:
    NUM_ITERS = 400
    NUM_ANNEALING_RUNS = 32
    E_SCALE = 1 / 40
    @staticmethod
    def T(x):
        return x ** 2
    @staticmethod
    def P(E, En, T):
        if T == 0:
            return 0
        expP = (En-E) * _AnnealingParameters.E_SCALE / T
        if expP > 162:
            return 0
        return 1 / (1 + exp(expP))

class UnsolvableException(Exception):
    pass

def _approximatePalettes(paletteData):
    # Set of palettes which we will modify during processing.
    pals = set(paletteData)
    # Set of palettes that have reached the max. num of colours and don't
    # need to be considered during processing.
    donepals = set()
    while True:
        merged = False
        # Remove simple subsets and full ("done") palettes
        # This should reduce the amount of processing we have to do later on.
        toBeIgnored = set()
        for p1 in pals:
            assert len(p1) <= MAX_COLOURS
            if len(p1) == MAX_COLOURS:
                # Remove palettes that have the max # of entries
                donepals.add(p1)
                toBeIgnored.add(p1)
                continue
            for p2 in pals:
                if p1 == p2:
                    continue
                if p1.issubset(p2):
                    # Ignore palettes that are a subset of another
                    toBeIgnored.add(p1)
                    break
        pals -= toBeIgnored

        # For each colour, find the number of uses
        colourcount = defaultdict(lambda: 0)
        for p in pals:
            for c in p:
                colourcount[c] += 1

        # Find all palettes sharing a particular colour
        for col, freq in sorted(colourcount.items(), key=lambda s: -s[1]):
            if freq <= 1:
                # If we're down to colours showing up in only one palette,
                # we can't do any merges here.
                # Try something else instead.
                break
            # Join all of the palettes having this colour into a single palette
            palettesToCombine = set(p for p in pals if col in p)
            combined = reduce(lambda s,f: s.union(f), palettesToCombine, set())
            if len(combined) > MAX_COLOURS:
                # If the resulting palette has too many colours, don't use it
                continue
            # Remove all of the palettes we just combined, and add 
            pals -= palettesToCombine
            pals.add(frozenset(combined))
            merged = True
            break
        if merged:
            # Restart from step 1
            continue
        # Step through, looking for palettes that have only differenceThreshold
        # colours differing from each other.
        for differenceThreshold in range(1, MAX_COLOURS - 1):
            for p1 in pals:
                for p2 in pals:
                    # If we can't combine these, ignore the pairing.
                    if p1 == p2 or len(p1 | p2) > 15:
                        continue
                    # If these two have few enough colours differing,
                    # merge them.
                    if len(p1 ^ p2) <= differenceThreshold:
                        merged = True
                        newpal = p1.union(p2)
                        pals.remove(p1)
                        pals.remove(p2)
                        pals.add(newpal)
                        break
                if merged:
                    break
        if merged:
            # Restart from step 1
            continue
        # We can't do any further optimizations with this method. Move on to
        # simulated annealing.
        break
    return frozenset(donepals | pals)

def _simulatedAnnealing(paletteData, approximation):
    # Get the parameters class from earlier on in the file for easier reference.
    params = _AnnealingParameters

    @lru_cache
    def palettesFromState(s):
        # Given the sets of all tile palettes in the different subpalettes,
        # create the set of colours in the subpalettes.
        def mergePaletteIndices(pidxs):
            # Merge all the colours from all the tile palettes based on the
            # indices given into one single set of colours (subpalette).
            out = set()
            for colours in (paletteData[i] for i in pidxs):
                out |= colours
            return frozenset(out)
        return frozenset(mergePaletteIndices(pidxs) for pidxs in s)

    @lru_cache
    def EFromPalettes(sp):
        e = 0
        # For each subpalette, calculate the energy resulting from that
        # subpalette. (Lower is better, annealing algo. minimizes this energy)
        for p in sp:
            if not p:
                continue
            l = len(p)
            # Fee for each colour used - incentivizes using fewer colours overall
            e += 1 * l
            # Fee for using a palette - incentivizes leaving palettes empty
            e += 10
            # Fee for each colour over max - incentivizes staying within colour limit
            if l > MAX_COLOURS:
                e += 50 * (l - MAX_COLOURS)
        return e

    def E(s):
        sp = palettesFromState(s)
        return EFromPalettes(sp)

    def neighbour(s, rng):
        # Move one tile palette from one subpalette to another, to create
        # a neighbouring state snew from s.
        subList = list(s)
        subIdxSrc, subIdxDst = tuple(rng.sample(range(len(subList)), 2))
        if len(subList[subIdxSrc]) == 0:
            # We can't move from a subpalette having no tile palettes.
            # Swap the source/destination.
            subIdxSrc, subIdxDst = subIdxDst, subIdxSrc
        if len(subList[subIdxSrc]) == 0:
            # Our source still doesn't have any tile palettes in it.
            # This means that neither one of them did.
            # Choose another one that does have tile palettes.
            subIdxSrc = rng.choice([idx for idx, p in enumerate(subList) if len(p) > 0])
            # Impossible to choose dst if we're here, since it must have
            # len == 0, and we're picking from lists with len != 0
        # Ensure that the logic above is holding...
        assert len(subList[subIdxSrc]) > 0
        # Get the sets of tile palette indices in the two subpalettes we chose
        subSrc = set(subList[subIdxSrc])
        subDst = set(subList[subIdxDst])
        # Remove indices in highest -> lowest order, so that the ordering doesn't
        # change for the other index due to the prior one being removed.
        subList.pop(max(subIdxSrc, subIdxDst))
        subList.pop(min(subIdxSrc, subIdxDst))
        # Move one tile palette index from the source set to the destination set.
        tileIdxToMove = rng.choice(list(subSrc))
        assert tileIdxToMove not in subDst
        subSrc.remove(tileIdxToMove)
        subDst.add(tileIdxToMove)
        # Return the new modified subpalettes in the correct state format.
        return frozenset(subList + [frozenset(subSrc), frozenset(subDst)])

    def generateS0(rng):
        '''
        Generate the initial state for the simulated annealing, based on the
        approximate results passed in.
        '''
        bins = [[] for _ in range(MAX_PALETTES)]
        # Get the MAX_PALETTES largest palettes from the approximation results.
        approx = sorted(approximation, key=lambda s: -len(s))[:MAX_PALETTES]
        done = set()
        # For each original palette, determine which of the approximated
        # palettes fits it, if any.
        for pi, p in enumerate(paletteData):
            for api, ap in enumerate(approx):
                if ap.issuperset(p):
                    bins[api].append(pi)
                    done.add(pi)
                    break
        # If there are any original palettes that didn't fit into the first N
        # approx'd palettes, throw them in randomly.
        for pi in range(len(paletteData)):
            if pi not in done:
                bins[rng.randrange(0, MAX_PALETTES)].append(pi)
        return frozenset(frozenset(p) for p in bins)

    rng = Random()
    # Create our s0 result based on the approximation, and find the energy
    # as a starting point.
    s0 = generateS0(rng)
    Es0 = E(s0)
    # Record the best result we've seen so far.
    bestE = Es0
    bestS = s0
    # Run a number of simulated annealing runs.
    for _ in range(params.NUM_ANNEALING_RUNS):
        # Start with the previous best. Even if we're in a local minimum, the
        # simulated annealing algorithm will be able to escape from it, and
        # hopefully through the power of trying many times, will find the
        # global minimum.
        s = bestS
        Es = bestE
        for k in range(params.NUM_ITERS):
            # Calculate current temperature.
            t = params.T(1 - (k+1) / params.NUM_ITERS)
            # Find a neighbouring state and calculate the energy.
            snew = neighbour(s, rng)
            Esnew = E(snew)
            # If the new state is better than our old best, record it.
            if Esnew < bestE:
                bestS = snew
                bestE = Esnew
            # If it's at a lower energy or if we pass our
            # "acceptance probability" check, then save it as our current
            # position.
            if Esnew < Es or params.P(Es, Esnew, t) > rng.random():
                s = snew
                Es = Esnew
    return sorted(palettesFromState(bestS), key=lambda p: -len(p))

def tilePalettesToSubpalettes(paletteData):
    '''
    Given a list of tile palettes, this function will fit them into subpalettes
    and return both the subpalettes and the mapping from the index into the
    tile palette list to the index into the subpalette list.

    Example with 5 colours instead of 15 per subpalette, and using symbolic
        values Cx... instead of integers:
    Given the inputs:
    [[C1, C2, C3, C4],
     [C1, C2, C5],
     [C5, C6, C7],
     [C1, C4, C5],
     [C8, C9]]
    This function could return:
    (
        [[C1, C2, C3, C4, C5],
         [C5, C6, C7, C8, C9]],
        {
            0: 0,
            1: 0,
            2: 1,
            3: 0,
            4: 1
        }
    )
    '''
    # Perform initial palette deduplication
    paletteDataDedup = set(frozenset(tilePal) for tilePal in paletteData)
    duplicatesToRemove = set()
    for pal1 in paletteDataDedup:
        for pal2 in paletteDataDedup:
            if pal1 == pal2:
                continue
            if pal1.issubset(pal2):
                duplicatesToRemove.add(pal1)
                break
    paletteDataDedup -= duplicatesToRemove
    paletteDataInternal = list(paletteDataDedup)
    # Perform our packing algorithm in two steps
    approx = _approximatePalettes(paletteDataInternal)
    finalSubpalettesSet = _simulatedAnnealing(paletteDataInternal, approx)
    if any(len(subPal) > MAX_COLOURS for subPal in finalSubpalettesSet):
        raise UnsolvableException("Unable to find a solution")
    # Calculate the subpalette index for each input palette
    tileToSubpaletteMap = {}
    for iTile, tilePal in enumerate(paletteData):
        for iSub, subPal in enumerate(finalSubpalettesSet):
            if frozenset(tilePal).issubset(subPal):
                tileToSubpaletteMap[iTile] = iSub
                break
        else:
            assert False, "This should never happen"
    return ([sorted(pal) for pal in finalSubpalettesSet], tileToSubpaletteMap)

def _verifyTestData(inputs, expectedSubpalettes, expectedMapping):
    resultSubpalettes, resultMapping = tilePalettesToSubpalettes(inputs)
    print('resultSubpalettes =', resultSubpalettes)
    print('resultMapping =', resultMapping)
    assert expectedSubpalettes and expectedMapping
    for subpal in expectedSubpalettes:
        assert subpal in resultSubpalettes
    assert len(resultMapping) == len(inputs)
    assert all(i in resultMapping for i in expectedMapping)
    for mapTile, mapSub in expectedMapping.items():
        expectedSubpal = expectedSubpalettes[mapSub]
        resultSubpal = resultSubpalettes[resultMapping[mapTile]]
        assert expectedSubpal == resultSubpal

def _verifyFailure(inputs):
    try:
        tilePalettesToSubpalettes(inputs)
    except UnsolvableException:
        pass
    else:
        assert False, "Solver should have failed"

def test_packing1():
    global MAX_COLOURS
    MAX_COLOURS = 5
    inputs = [
        [1,2,3,4],
        [1,2,5],
        [5,6,7],
        [1,4,5],
        [8,9]
    ]
    expectedSubpalettes = [
        [1,2,3,4,5],
        [5,6,7,8,9]
    ]
    expectedMapping = {
        0:0,
        1:0,
        2:1,
        3:0,
        4:1
    }
    _verifyTestData(inputs, expectedSubpalettes, expectedMapping)

def test_packing2():
    global MAX_COLOURS
    MAX_COLOURS = 15
    inputs = [[x] for x in range(100,115)] + [
        list(range(100,115)),
        list(range(200,215)),
    ]
    expectedSubpalettes = [
        list(range(100,115)),
        list(range(200,215)),
    ]
    expectedMapping = {i:0 for i in range(15)}
    expectedMapping.update({
        15:0,
        16:1,
    })
    _verifyTestData(inputs, expectedSubpalettes, expectedMapping)

def test_packing3():
    global MAX_COLOURS
    MAX_COLOURS = 9
    inputs = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 5, 6, 7],
        [1, 8, 9, 10, 11, 12, 13],
        [11, 12, 13, 14, 15],
        [1, 14, 15],
        [21, 22, 23, 24],
        [1, 2, 3, 21, 22, 23],
        [1, 23, 24, 25],
        [1, 23, 24, 26],
        [1, 2, 3, 5, 6, 7]
    ]
    expectedSubpalettes = [
        [1, 2, 3, 4, 5, 6, 7],
        [1, 8, 9, 10, 11, 12, 13, 14, 15],
        [1, 2, 3, 21, 22, 23, 24, 25, 26]
    ]
    expectedMapping = {
        0:0,
        1:0,
        2:1,
        3:1,
        4:1,
        5:2,
        6:2,
        7:2,
        8:2,
        9:0
    }
    _verifyTestData(inputs, expectedSubpalettes, expectedMapping)

def test_negative1():
    global MAX_COLOURS
    MAX_COLOURS = 1
    inputs = [
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7]
    ]
    _verifyFailure(inputs)
