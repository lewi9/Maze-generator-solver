import numpy as np
import matplotlib.pyplot as plt
import random
import sys

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
        t: int = random.randrange(4)
        increament: int = 0
        
        while increament < 4:
            t = (t+increament)%4
            if t == 0:
                if a+2 < y*2-1:
                    if l[a+2][b] == 1:
                        l[a+1][b] = 1
                        genRoad(a+2,b)
            elif t == 1:
                if b+2 < x*2-1:
                    if l[a][b+2] == 1:
                        l[a][b+1] = 1
                        genRoad(a,b+2)
            elif t == 2:
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
    t: int = int(random.randrange(count)*p)
    while t > 0:
        tx = random.randrange(x*2-1)
        ty = random.randrange(y*2-1)
        if l[ty][tx] == 0:
            l[ty][tx] = 1
            t -= 1
    return l
                 
def waterSplash(maze: np.array, x: int, y:int) -> np.array:
    l: np.array = np.copy(maze)
    l[0][0] = 1
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
    l[2*y-2][2*x-2,] = 0
    return l
        
def solveMaze(maze: np.array, waterSplashed: np.array, x:int, y:int) -> tuple:
    l: np.array = np.copy(maze)
    l[l>1] = 1

    def solver(w: int, z: int) -> bool:
        if w == y*2-2 and z == x*2-2:
            l[w][z] = 3
            return True
        r = []
        if w+1 < 2*y-1:
            r.append(("u", waterSplashed[w+1][z]))
        if w-1 >= 0:
            r.append(("d", waterSplashed[w-1][z]))
        if z+1 < 2*x-1:
            r.append(("r", waterSplashed[w][z+1]))
        if z-1 >= 0:
            r.append(("l", waterSplashed[w][z-1]))

        r: list = sorted(r, key=itemgetter(1))
        index: int = 0
        while index < len(r):
            if r[index][0] == "u":
                if l[w+1][z] == 1:
                    l[w+1][z] = 2
                    if solver(w+1,z):
                        l[w][z] = 3
                        return True
            elif r[index][0] == "d":
                if l[w-1][z] == 1:
                    l[w-1][z] = 2
                    if solver(w-1,z):
                        l[w][z] = 3
                        return True
            elif r[index][0] == "r":
                if l[w][z+1] == 1:
                    l[w][z+1] = 2
                    if solver(w,z+1):
                        l[w][z] = 3
                        return True
            else:
                if l[w][z-1] == 1:
                    l[w][z-1] = 2
                    if solver(w,z-1):
                        l[w][z] = 3
                        return True
            index += 1
        return False
    
    l[0][0] = 2
    solver(0,0)
    
    l[l==2] = 1
    l[l==3] = 2
    return (l, np.count_nonzero(l==2))

def showMaze(ax : plt.axes, title: str, maze: np.array, x: int, y:int):
    xv: list = np.linspace(0,2*x,2*x+1)
    yh: list = np.linspace(0,2*y,2*y+1)
    xp: list = []
    yp: list = []

    for i in range(2*y-1):
        for j in range(2*x-1):
            if maze[i][j] > 0:
                for k in range(int(maze[i][j])):
                    xp.append(j)
                    yp.append(i)
                    
    H,X,Y = np.histogram2d(x = xp, y = yp, bins=(xv,yh))
    ax.pcolormesh(X,Y,H.T, cmap="viridis")
    ax.axis("square")
    ax.set_title(title, fontweight='bold')
    ax.set_xlim([0,2*x-1])
    ax.set_ylim([0,2*y-1])
    ax.set_axis_off()

def main():

    # Note that maze will be 2*x wide and 2*y height
    # IMPORTANT x*y should be not too much, because can be problem with shell and program execution (recursion)
    x: int = 80
    y: int = 30

    if x<3:
        x=3
    if y<3:
        y=3
    
    sys.setrecursionlimit(4*x*y)

    n: int = 0
    nR: int = 0
    maze            = genMaze(x, y)
    mazeR           = randMaze(maze, x, y, 0.1)
    waterSplashed   = waterSplash(maze, x, y)
    waterSplashedR  = waterSplash(mazeR, x, y)
    solution,  n    = solveMaze(maze, waterSplashed, x, y)
    solutionR, nR   = solveMaze(mazeR, waterSplashedR, x, y)

    fig,ax=plt.subplots(3,2,figsize=(50,50))

    showMaze(ax[0,0], "Maze.", maze, x, y)
    showMaze(ax[0,1], "Randomized maze.", mazeR, x, y)
    showMaze(ax[1,0], "Water splash of maze.", waterSplashed, x, y)
    showMaze(ax[1,1], "Water splash of randomized maze.", waterSplashedR, x, y)
    showMaze(ax[2,0], "Solution of maze. Steps: " + str(n), solution, x, y)
    showMaze(ax[2,1], "Solution of randomized maze. Steps: " + str(nR), solutionR, x, y)

    fig.suptitle("Maze generator and solver. Start is in bottom-left corner, end is in top-right.", fontsize=18, fontweight='bold')
    
    fig.show()

if __name__ == "__main__":
    main()
