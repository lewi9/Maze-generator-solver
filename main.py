import numpy as np
import matplotlib.pyplot as plt
import random
import sys

from anytree import Node, RenderTree
from operator import itemgetter
	
def genMaze(x: int, y : int) -> np.array:
    l: np.array = np.empty([y*2-1, x*2-1])
    
    for i in range(2*y-1):
        for j in range(2*x-1):
            if not i % 2:
                if not (i+j)%2:
                    l[i][j] = 1
                    continue
            l[i][j] = 0

    def genRoad(a: int, b:int) -> None:
        l[a][b] = 2
        direction: int = random.randrange(4)
        increament: int = 0
        
        while increament < 4:
            direction = (direction+increament)%4
            if direction == 0:
                if a+2 < y*2-1:
                    if l[a+2][b] == 1:
                        l[a+1][b] = 1
                        genRoad(a+2,b)
            elif direction == 1:
                if b+2 < x*2-1:
                    if l[a][b+2] == 1:
                        l[a][b+1] = 1
                        genRoad(a,b+2)
            elif direction == 2:
                if a-2 >= 0:
                    if l[a-2][b] == 1:
                        l[a-1][b] = 1
                        genRoad(a-2,b)
            else:
                if b-2 >= 0:
                    if l[a][b-2] == 1:
                        l[a][b-1] = 1
                        genRoad(a,b-2)
            increament += 1

    genRoad(random.randrange(y)*2, random.randrange(x)*2)
    l[l==2] = 1
    l[0][0] = 2
    l[2*y-2][2*x-2] = 2
    return l

def randMaze(maze: np.array, x: int, y:int, p:float) -> np.array:
    l: np.array = np.copy(maze)
    count: int = np.count_nonzero(l==0)
    wallToDestroy: int = int(random.randrange(count)*p)
    while wallToDestroy > 0:
        tx: int = random.randrange(x*2-1)
        ty: int = random.randrange(y*2-1)
        if l[ty][tx] == 0:
            l[ty][tx] = 1
            wallToDestroy-= 1
    return l

def bushyMaze(maze: np.array, x: int, y:int, weightCenter: int, weightOutsideCenter: int) -> np.array:
    l: np.array = np.copy(maze)
    centerFactor: int = 4
    outsideCenterFactor: int = 8
    centerBorderY: tuple = (int((2*y-1)/centerFactor),int((2*y-1)/centerFactor*(centerFactor-1)))
    centerBorderX: tuple = (int((2*x-1)/centerFactor),int((2*x-1)/centerFactor*(centerFactor-1)))
    outsideCenterBorderY: tuple = (int((2*y-1)/outsideCenterFactor),int((2*y-1)/outsideCenterFactor*(outsideCenterFactor-1)))
    outsideCenterBorderX: tuple = (int((2*x-1)/outsideCenterFactor),int((2*x-1)/outsideCenterFactor*(outsideCenterFactor-1)))
    
    for i in range(outsideCenterBorderY[0], outsideCenterBorderY[1]):
        for j in range(outsideCenterBorderX[0], outsideCenterBorderX[1]):
            if l[i][j] > 0:
                l[i][j] = weightOutsideCenter
                
    for i in range(centerBorderY[0], centerBorderY[1]):
        for j in range(centerBorderX[0], centerBorderX[1]):
            if l[i][j] > 0:
                l[i][j] = weightCenter
                
    l[0][0] = (weightCenter + weightOutsideCenter) * 2
    l[2*y-2][2*x-2] = (weightCenter + weightOutsideCenter) * 2
    return l
            
def waterSplash(maze: np.array, x: int, y: int) -> np.array:
    l: np.array = np.copy(maze)
    l[l>1] = 1
    l[2*y-2][2*x-2] = 2
    def water(w: int, z: int, p: int) -> None:
        if w+1 < 2*y-1:
            if l[w+1][z] == 1 or l[w+1][z] > l[w][z]+1:
                l[w+1][z] = 1+p
                water(w+1, z, p+1)
        if w-1 >= 0:
            if l[w-1][z] == 1 or l[w-1][z] > l[w][z]+1:
                l[w-1][z] = 1+p
                water(w-1, z, p+1)
        if z+1 < 2*x-1:
            if l[w][z+1] == 1 or l[w][z+1] > l[w][z]+1:
                l[w][z+1] = 1+p
                water(w, z+1, p+1)
        if z-1 >= 0:
            if l[w][z-1] == 1 or l[w][z-1] > l[w][z]+1:
                l[w][z-1] = 1+p
                water(w,z-1,p+1)

    water(2*y-2,2*x-2,1)
    l[2*y-2][2*x-2] = 0
    return l
        
def solveMaze(maze: np.array, waterSplashed: np.array, x:int, y:int, weightSum: int = 1) -> tuple:
    l: np.array = np.copy(maze)
    l[0][0] = -2
    l[y*2-2][x*2-2] = 1
    tree: list = [Node([0, 0, waterSplashed[0][0], 0]),]
    
    def solve() -> tuple:
        def findMinimumNodeMove() -> list:
            leaves: list = list(tree[0].leaves)
            leavesName: list = [node.name for node in leaves]

            minValue: int = min(leavesName, key=itemgetter(2))
            minIndex: int = leavesName.index(minValue)
            
            node: Node = leaves[minIndex]

            parents: list = [parent for parent in node.iter_path_reverse()]
            parentsName: list = [parent.name for parent in parents]

            minValue = min(parentsName, key=itemgetter(2))
            minIndex = parentsName.index(minValue)

            node = parents[minIndex]
            
            w: int = node.name[0]
            z: int = node.name[1]
            value: int = node.name[2]
            cost: int = node.name[3]

            if w == y*2-2 and z == x*2-2:
                return [[], w, z, value, cost, node]
            
            r = []
            if w+1 < 2*y-1:
                if l[w+1][z] > 0:
                    r.append(("u", waterSplashed[w+1][z] + cost + l[w+1][z]))
            if w-1 >= 0:
                if l[w-1][z] > 0:
                    r.append(("d", waterSplashed[w-1][z] + cost + l[w-1][z]))
            if z+1 < 2*x-1:
                if l[w][z+1] > 0:
                    r.append(("r", waterSplashed[w][z+1] + cost + l[w][z+1]))
            if z-1 >= 0:
                if l[w][z-1] > 0:   
                    r.append(("l", waterSplashed[w][z-1] + cost + l[w][z-1]))

            if len(r) == 0:
                if node.is_leaf:
                    tree.remove(node)
                    node.parent = None
                else:
                    tree[tree.index(node)].name[2] = np.Inf
                return findMinimumNodeMove()

            return [r, w, z, value, cost, node]

        r, w, z, value, cost, node = findMinimumNodeMove()
        
        if w == y*2-2 and z == x*2-2:
                l[w][z] = -2
                return (True, value)
            
        r = sorted(r, key=itemgetter(1))
        index: int = 0
        while index < len(r):
            if r[index][0] == "u":
                tree.append(Node([w+1, z, r[index][1], cost + l[w+1][z]], parent = node))
                l[w+1][z] = -1
                effect = solve()
                if effect[0]:
                    return (True, effect[1])
            elif r[index][0] == "d":
                tree.append(Node([w-1, z, r[index][1], cost + l[w-1][z]], parent = node))
                l[w-1][z] = -1
                effect = solve()
                if effect[0]:
                    return (True, effect[1])
            elif r[index][0] == "r":
                tree.append(Node([w, z+1, r[index][1], cost + l[w][z+1]], parent = node))
                l[w][z+1] = -1
                effect = solve()
                if effect[0]:
                    return (True, effect[1])    
            else:
                tree.append(Node([w, z-1, r[index][1], cost + l[w][z-1]], parent = node))
                l[w][z-1] = -1
                effect = solve()
                if effect[0]:
                    return (True, effect[1])
            index += 1
        return (False, None)
    
    n: int = solve()[1]

    def markRoad(node: Node) -> None:
        l[node.name[0]][node.name[1]] = -2
        if node.parent:
            markRoad(node.parent)

    markRoad(tree[-1])
    
    for i in range(2*y-1):
        for j in range(2*x-1):
            if l[i][j] == -1:
                l[i][j] = maze[i][j]
                
    l[l==-2] = weightSum*2
    
    return (l, n)

def showMaze(ax : plt.axes, title: str, maze: np.array, x: int, y: int, weightCenter: int = 1, weightOutsideCenter: int = 1, flagHeuristic: bool = False) -> None:
    xv: list = np.linspace(0,2*x,2*x+1)
    yh: list = np.linspace(0,2*y,2*y+1)
    xp: list = []
    yp: list = []

    for i in range(2*y-1):
        for j in range(2*x-1):
            if maze[i][j] > 0:
                if not flagHeuristic:
                    
                    if maze[i][j] == 1:
                        xp.append(j)
                        yp.append(i)
                    elif maze[i][j] == 2:
                        for k in range(2):
                            xp.append(j)
                            yp.append(i)
                    elif maze[i][j] == min(weightCenter,weightOutsideCenter):
                        for k in range(2):
                            xp.append(j)
                            yp.append(i)
                    elif maze[i][j] == max(weightCenter,weightOutsideCenter):
                        for k in range(3):
                            xp.append(j)
                            yp.append(i)
                    elif maze[i][j] == (weightCenter+weightOutsideCenter)*2:
                        for k in range(4):
                            xp.append(j)
                            yp.append(i)
                    
                else:
                    for k in range(int(maze[i][j])):
                        xp.append(j)
                        yp.append(i)
                            
    H, X, Y = np.histogram2d(x = xp, y = yp, bins=(xv,yh))
    ax.pcolormesh(X,Y,H.T, cmap="viridis")
    ax.axis("square")
    ax.set_title(title, fontweight='bold')
    ax.set_xlim([0,2*x-1])
    ax.set_ylim([0,2*y-1])
    ax.set_axis_off()

def main():

    # Note that maze will be 2*x wide and 2*y height
    # IMPORTANT x*y should be not too much, because can be problem with shell and program execution (recursion)
    x: int = 50
    y: int = 50

    x = max(x,5)
    y = max(y,5)

    weightCenter: int = 3
    weightOutsideCenter: int = 2

    weightSum: int = weightCenter+weightOutsideCenter
    
    sys.setrecursionlimit(4*x*y)

    n: int = 0
    nR: int = 0
    nRB: int = 0
    maze            = genMaze(x, y)
    mazeR           = randMaze(maze, x, y, 0.2)
    mazeRB          = bushyMaze(mazeR, x, y, weightCenter, weightOutsideCenter)
    waterSplashed   = waterSplash(maze, x, y)
    waterSplashedR  = waterSplash(mazeR, x, y)
    waterSplashedRB = waterSplash(mazeRB, x, y)
    solution,  n    = solveMaze(maze, waterSplashed, x, y)
    solutionR, nR   = solveMaze(mazeR, waterSplashedR, x, y)
    solutionRB, nRB = solveMaze(mazeRB, waterSplashedRB, x, y, weightSum)

    fig, ax = plt.subplots(3,3,figsize=(50,50))

    showMaze(ax[0,0], "Maze.", maze, x, y)
    showMaze(ax[0,1], "Randomized maze.", mazeR, x, y)
    showMaze(ax[0,2], "Randomized bushy maze. Walking by bush costs more.", mazeRB, x, y, weightCenter, weightOutsideCenter )
    showMaze(ax[1,0], "Hueristic function (water splash).", waterSplashed, x, y, flagHeuristic = True)
    showMaze(ax[1,1], "Hueristic function (water splash)\nof randomized maze.", waterSplashedR, x, y, flagHeuristic = True)
    showMaze(ax[1,2], "Hueristic function (water splash)\nof randomized bushy maze.", waterSplashedRB, x, y, flagHeuristic = True)
    showMaze(ax[2,0], "Solution of maze. Cost: " + str(n), solution, x, y)
    showMaze(ax[2,1], "Solution of randomized maze. Cost: " + str(nR), solutionR, x, y)
    showMaze(ax[2,2], "Solution of randomized bushy maze. Cost: " + str(nRB), solutionRB, x, y, weightCenter, weightOutsideCenter)

    fig.suptitle("Maze generator and solver. Start is in bottom-left corner, end is in top-right.", fontsize=18, fontweight='bold')
    
    fig.show()

if __name__ == "__main__":
    main()
