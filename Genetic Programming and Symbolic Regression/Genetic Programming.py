import csv
import random
import copy
import math
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from queue import PriorityQueue

data = []
data2 = []
operators = ["+","-","*","/"]
terminals = ["x"]

with open('dataset1.csv') as csv_file:
    readCSV = csv.reader(csv_file)
    for row in readCSV:
        data.append((row[0], row[1]))
    numDataset = 1
data.pop(0)



# with open('dataset2.csv') as csv_file:
#     readCSV = csv.reader(csv_file)
#     for row in readCSV:
#          #x,y,z, and f(x,y,z)
#         data.append((row[0], row[1], row[2], row[3]))
#     numDataset = 2
# data.pop(0)

train,test = train_test_split(data, test_size=0.2)

class Tree:

    #constructor to make tree
    def __init__(self, value):
        self.left = None
        self.right = None
        self.value = value
        self.fitness = None
        self.parent = None

    #counts the number of nodes in the tree in order to make the random node function work
    def nodeCount(self):
        if self is None: 
            return 0 
        if(self.left is None and self.right is None): 
            return 1
        else: 
            return 1 + self.left.nodeCount() + self.right.nodeCount() 
        
    
    #this may be working maybe not that well but it should give you a random node in a
    #tree after moving through the tree in pre order
    
    def randomNode(self, node_index):
        copy = node_index
        #if the tree is empty 
        if self.value is None:
            return None
        #if the tree is just the root
        if self.left is None and self.right is None:
            return self
        if node_index == 1: 
            return self
        #while not at the specific node
        #traverse the nodes and return that specific node
        if node_index % 2 == 0 and copy % 2 != 0 or node_index % 2 != 0 and copy % 2 == 0:
            return self.left.randomNode(node_index - 1)
        if node_index % 2 == 0 and copy % 2 == 0 or node_index % 2 != 0 and copy % 2 != 0:
            return self.right.randomNode(node_index - 1)


    def grow(self, max_depth, curDepth, consa, consb):
        #creates a tree from scratch
        workingSet = []
    
        num = random.randint(0,1)
        num2 = random.randint(0,1)
        if num == 0:
            workingSet = operators
            leftVal = random.choice(workingSet)
        else:
            workingSet = terminals
            num3 = random.randint(0,2)
            if num3 == 2:
                leftVal = random.choice(workingSet)
            else:
                #randint for dataset1 and uniform for dataset2
                if numDataset == 1:
                    leftVal = random.randint(consa, consb)
                else:
                    leftVal = random.uniform(consa, consb)
    
        if num2 == 0:
            workingSet = operators
            rightVal = random.choice(workingSet)
        else:
            workingSet = terminals
            num3 = random.randint(0,2)
            if num3 == 2:
                rightVal = random.choice(workingSet)
            else:
                #randint for dataset1 and uniform for dataset2
                if numDataset == 1:
                    rightVal = random.randint(consa, consb)
                else:
                    rightVal = random.uniform(consa, consb)
         
        if curDepth >= max_depth - 1:
            workingSet = terminals
            if numDataset == 1:
                workingSet = terminals + [random.randint(consa, consb)]
            else:
                workingSet = terminals + [random.uniform(consa, consb)]
            leftVal = random.choice(workingSet)

            if numDataset == 1:
                workingSet = terminals + [random.randint(consa, consb)]
            else:
                workingSet = terminals + [random.uniform(consa, consb)]

            rightVal = random.choice(workingSet)

        if self.value in operators:
            self.left = Tree(leftVal)
            self.right = Tree(rightVal)
            self.left.parent = self
            self.right.parent = self
            curDepth += 1
        if leftVal in operators:
            self.left.grow(max_depth, curDepth, consa, consb)
        if rightVal in operators:
            self.right.grow(max_depth, curDepth, consa, consb)

    def mutate(self):
        #gets the random node
        clone = self.clone()
        num_nodes = clone.nodeCount()
        random_num = random.randint(1, num_nodes)
        random_node = clone.randomNode(random_num)
        if random_node.value in operators:
            new = random.choice(operators)
            random_node.value = new
        if random_node.value in terminals:
            new = random.choice(terminals)
            random_node.value = new
        clone.findFitness(data)
        return clone

    def clone(self):
        #returns a copy of the tree that is passed in
        copy = Tree(self.value)
        copy.fitness = self.fitness
        copy.parent = self.parent
        if self.left != None:
            copy.left = Tree.clone(self.left)
        if self.right != None:
            copy.right = Tree.clone(self.right)
        return copy
    
    #geeks for geeks has one of these 
    def maxDepth(self): 
        if self is None: 
            return 0
        if self.left is None and self.right is None:
            return 1
        else: 
            leftD = self.left.maxDepth() 
            rightD = self.right.maxDepth() 
  
            if (leftD>rightD): 
                return leftD+1
            else: 
                return rightD+1
      
    def crossover(self,tree2):
    
        tree1Copy = self.clone()
        tree2Copy = tree2.clone()

        numNodes1 = tree1Copy.nodeCount()
        numNodes2 = tree2Copy.nodeCount()

        randomNum1 = random.randint(1, numNodes1)
        randomNum2 = random.randint(1, numNodes2)

        swapNode1 = tree1Copy.randomNode(randomNum1)
        swapNode2 = tree2Copy.randomNode(randomNum2)

        tempNode = swapNode2

        if swapNode1.parent is not None:
            swapNode2.parent = swapNode1.parent
            if swapNode1.parent.left == swapNode1:
                swapNode1.parent.left = swapNode2
            else:
                swapNode1.parent.right = swapNode2
        else:
            swapNode2.parent = None

        swapNode2.left = swapNode1.left
        swapNode2.right = swapNode1.right
        swapNode2.value = swapNode1.value

        if tempNode.parent is not None:
            swapNode1.parent = tempNode.parent
            if tempNode.parent.left == tempNode:
                tempNode.parent.left = swapNode1
            else:
                tempNode.parent.right = swapNode1
        else:
            swapNode1.parent = None

        swapNode1.left = tempNode.left
        swapNode1.right = tempNode.right
        swapNode1.value = tempNode.value

        tree1Copy.findFitness(data)
        tree2Copy.findFitness(data)

        if tree1Copy.fitness < tree2Copy.fitness:
            return tree1Copy
        if tree1Copy.fitness > tree2Copy.fitness:
            return tree2Copy

        return tree1Copy
     
    
    def evaluate(self, x,y,z):
        if self is None:
            return None
        if self.left is None and self.right is None:
            if self.value == "x":
                return float(x)
            elif self.value == "y":
                return float(y)
            elif self.value == "z":
                return float(z)
            else:
                return float(self.value)

        if self.value == "+":
            return float(self.left.evaluate(x,y,z)) + float(self.right.evaluate(x,y,z))
        if self.value == "-":
            return float(self.left.evaluate(x,y,z)) - float(self.right.evaluate(x,y,z))
        if self.value == "*":
            return float(self.left.evaluate(x,y,z)) * float(self.right.evaluate(x,y,z))
        if self.value == "/":
            if float(self.right.evaluate(x,y,z)) == 0.0:
                #chooses ops not including / and %
                return np.inf
            else:
                return float(self.left.evaluate(x,y,z)) / float(self.right.evaluate(x,y,z))

        # if self.parent == "log":
        #     return math.exp(float(self.evaluate(x,y,z), 10))
        # if self.parent == "ln":
        #     return float(self.evaluate(math.log(x,y,z)))
        # if self.parent == "sin":
        #     return float(self.evaluate(math.sin(x,y,z)))
        # if self.parent == "cos":
        #     return float(self.evaluate(math.cos(x,y,z)))
        # if self.parent == "tan":
        #     return float(self.evaluate(math.tan(x,y,z)))
        # if self.parent == "abs":
        #     return float(self.evaluate(abs(x,y,z)))
        # if self.parent == "sqrt":
        #     return float(self.evaluate(math.sqrt(x,y,z)))
    
    def __lt__(self, other):
        return self.fitness < other.fitness          

    def findFitness(self, data):
        #Root Mean Squared Error
        squaredSum = 0
        for i in range(len(data)):
            if numDataset == 2:
                result = self.evaluate(data[i][0],data[i][1], data[i][2])
                desired = float(data[i][3])
            else:
                result = self.evaluate(data[i][0],0.0,0.0)
                desired = float(data[i][1])

            squaredSum += (desired- float(result))**2
        squaredSum /= len(data)
        fitScore = math.sqrt(squaredSum)
        fitScore += self.nodeCount()*(0.05)
        #lower is better
        self.fitness = fitScore

    def printTree(self):
        #prints value, then left, then right
        print(self.value)
        if self.left:
            self.left.printTree()
        if self.right:
            self.right.printTree()
    

def createPopulation(population_size, depth, consa, consb):
    population = []
    for i in range(population_size):
        tree = Tree(random.choice(operators))
        depth = random.randint(1,10)
        tree.grow(depth, 1, consa, consb)
        population.append(tree)
    return population

    #we want to select a program that has high fitness but we also
    #have to consider not allowing low diversity in our program
    #Elitism is known to have the best fitting so we are implementing
    

def eliteSelection(population, elite_size):
    pop = PriorityQueue()
    #all values in population should be in priority queue in order of smallest MSE
    for i in range(len(population)):
        curNode = population[i]
        if curNode.fitness == None:
            curNode.findFitness()
        var = (curNode.fitness, curNode)
        pop.put(var)
    #add elite_size number of elements into values array and return a random tree in array
    parents = []
    for i in range(elite_size):
        parents.append(pop.get()[1])
    
    return parents

def new_population(crossover_pct, mutate_pct, parents, population_size, elite_size, depth, consa, consb, stale): 
    population = []
    for i in range(elite_size):
        population.append(parents[i])
        
    if stale == True:
        for i in range(elite_size, population_size):
            newtree = Tree(random.choice(operators))
            newtree.grow(depth,1, consa, consb)
            population.append(newtree)
    else:
        for i in range(elite_size, population_size):
            action = random.uniform(0,1)
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            if action < crossover_pct:
                offspring = parent1.crossover(parent2)
                if action < mutate_pct:
                    offspring = offspring.mutate()
                population.append(offspring)
            
        else:
            newtree = Tree(random.choice(operators))
            newtree.grow(depth, 1, consa, consb)
            population.append(newtree)
    return population

def printEquation(root):
    if root:
        printEquation(root.left)
        print(root.value),
        printEquation(root.right)
    

def geneticAlgorithm(population_size, generations, consa, consb, depth, mutate_pct, crossover_pct, elite, data):
    global_best = np.inf
    prev_best = np.inf
    stale = False
    elite_size = int(population_size / elite)
    
    
    if numDataset == 2:
        terminals.append("y")
        terminals.append("z")

    MSE = 0.0
    #this should initialize the initial population
    population = createPopulation(population_size, depth, consa, consb)
    for generation in range(generations):
        for i in range(len(population)):
            if population[i].fitness == None:
                population[i].findFitness(data)
            score = population[i].fitness
            
            if score < global_best:                                                                      
                global_best = score                                                                                                                                       
                best_tree = population[i]
        # print(                                                                                           
        #     "Generation: %d\n Best Score: %.2f\n Best Tree:"                   
        #     % (                                                                                          
        #         generation,                                                                                     
        #         global_best,               
        #     )                                                                                            
        # )
        
        if prev_best == global_best:
            stale_count+=1
        else:
            stale_count = 0
            
        prev_best = global_best
        
        
        if stale_count >= 5:
            stale=True
            stale_count = 0
        
        
        
        # printEquation(best_tree)
        # print("")
        #print("stale count:" + str(stale_count))
        parents = eliteSelection(population, elite_size)
        # for i in range(elite_size):
        # for i in range(population_size):

        population = new_population(crossover_pct, mutate_pct, parents, population_size, elite_size, depth, consa, consb, stale)
          

    MSE = global_best 
    return MSE
#geneticAlgorithm(population_size, generations, consa, consb, depth, mutate_pct, crossover_pct, elite, data):
#parameter 1: geneticAlgorithm(100, 20, -5, 5, 20, 0.2, 0.8, 10, train)
#parameter 6: geneticAlgorithm(50, 20, -5, 5, 50, 0.2, 0.8, 10, train)
#geneticAlgorithm(100, 1000)
runs = 20
with open("GP MSE 20 runs P6D1.csv", "w") as csvfile:
    for i in range(runs):
        print(i)
        test = geneticAlgorithm(50, 20, -5, 5, 50, 0.2, 0.8, 10, train)
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([i , test])
