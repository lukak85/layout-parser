import cv2
import numpy
import math

import matplotlib.pyplot as plt

from . import colors
from . import geometry as g
from .box import Box
from .dimension import Dimension

from scipy import spatial

import itertools
import operator

def threshold(image, threshold=colors.greyscale.MID_GREY, method=cv2.THRESH_BINARY_INV):
    retval, dst = cv2.threshold(image, threshold, colors.greyscale.WHITE, method)
    return dst

class Character:

    def __init__(self, x, y):

        self.coordinate = [x, y]
        self.x = x
        self.y = y

        self.nearestNeighbours = []
        self.parentWord = None

    def assignParentWord(self, word):

        self.parentWord = word
        self.parentWord.registerChildCharacter(self)

        for neighbour in self.nearestNeighbours:
            if neighbour.parentWord == None:
                neighbour.assignParentWord(self.parentWord)

    def toArray(self):
        return self.coordinate

    def __len__(self):
        return len(self.coordinate)

    def __getitem__(self, key):
        return self.coordinate.__getitem__(key)

    def __setitem__(self, key, value):
        self.coordinate.__setitem__(key, value)

    def __delitem__(self, key):
        self.coordinate.__delitem__(key)

    def __iter__(self):
        return self.coordinate.__iter__()

    def __contains__(self, item):
        return self.coordinate.__contains__(item)

    ''' paint '''
    ''' paint a dot on the centroid of a character '''
    def paint(self, image, color=colors.YELLOW):

        pointObj = g.Point(self.coordinate)
        image = pointObj.paint(image, color)
        return image

class CharacterSet:

    def __init__(self, sourceImage):

        self.characters = self.getCharacters(sourceImage)
        self.NNTree = spatial.KDTree([char.toArray() for char in self.characters])
        #self.angles = []
        #self.distances = []

    ''' getCharacters '''
    ''' This function (1) binarize a source image (2) get contours (characters) (3) get its centroid  '''
    def getCharacters(self, sourceImage):

        characters = []

        image = sourceImage.copy()
#        image = threshold(image, cv2.THRESH_OTSU, method=cv2.THRESH_BINARY)

        for contour in self.getContours(image):
            try:
                box = Box(contour)

                moments = cv2.moments(contour)
                centroidX = int( moments['m10'] / moments['m00'] )
                centroidY = int( moments['m01'] / moments['m00'] )
                character = Character(centroidX, centroidY)
                
            except ZeroDivisionError:
                continue
                
            #if box.area > 50:
            if box.area > 1:
            #if True:
                characters.append(character)

        print("Total ", len(characters), " characters are found.")
        return characters

    ''' getContours         '''
    ''' Input: Binary Image '''
    ''' Output: BLOBs       '''
    def getContours(self, sourceImage, threshold=-1):
        image = sourceImage.copy()
        blobs = []
        top_level_contours = []

        # cv2.findContours : It stores the (x,y) coordinates of the boundary of a shape. Here, contours are the boundaries of a shape with same intensity.
        # CHAIN_APPROX_NONE : All the boundary points are stored.
        # CHAIN_APPROX_SIMPLE : It removes all redundant points and compresses the contour, thereby saving memory.
        # hierarchy = [Next, Previous, First_Child, Parent]
        # REFERENCE : https://docs.opencv.org/3.1.0/d4/d73/tutorial_py_contours_begin.html
        contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i in range(len(hierarchy[0])):

            if len(contours[i]) > 2:    # 1- and 2-point contours have a divide-by-zero error in calculating the center of mass.

                # bind each contour with its corresponding hierarchy context description.
                obj = {'contour': contours[i], 'context': hierarchy[0][i]}
                blobs.append(obj)

        for blob in blobs:
            parent = blob['context'][3]
            if parent <= threshold: # no parent, therefore a root
                top_level_contours.append(blob['contour'])

        return top_level_contours

    ''' Find the most common element in a list '''
    ''' Input: A list '''
    ''' Output: The most common element '''    
    def most_common(self, unsorted_list):
        # Get an iterable of (item, iterable) pairs
        sorted_list = sorted((x, i) for i, x in enumerate(unsorted_list))
        # print 'sorted_list:', sorted_list
        groups = itertools.groupby(sorted_list, key=operator.itemgetter(0))
        # auxiliary function to get "quality" for an item
        def _auxfun(g):
            item, iterable = g
            count = 0
            min_index = len(unsorted_list)
            for _, where in iterable:
                count += 1
                min_index = min(min_index, where)
            return count, -min_index
        # pick the highest-count/earliest item
        return max(groups, key=_auxfun)[0]
    
    ''' getWords '''
    ''' Find nearest neighbors '''
    ''' Input: Characters '''
    ''' Output: k-nearest neighbors '''
    def getWords(self):

        words = []
        k = 5
        mode = 'horizontal' # mode = ['default','horizontal','vertical']
        #EPS = 1e-2
       
        # find the average distance between nearest neighbours
        nn_distances = []
        nn_horizontal_distances = []
        nn_vertical_distances = []
        remove_counter = 0
        for character in self.characters:
            remove_counter = remove_counter+1
            result = self.NNTree.query(character.toArray(), k=k)  # we only want nearest neighbour, but the first result will be the point matching itself.
            nearestNeighbourDistance = result[0]
            for i in range(1,k):
                #print("[%s] nearestNeighbourDistance: %s"%(remove_counter,result[0]))
                nn_distances.append(nearestNeighbourDistance[i])
        avgNNDistance = sum(nn_distances)/len(nn_distances)
        
        maxDistance = avgNNDistance*3
        #maxDistance = avgNNDistance*20000
        for character in self.characters:
            #print ("Finding a a nn of ",character.x,character.y)
            queryResult = self.NNTree.query(character.coordinate, k=k)
            distances = queryResult[0]
            neighbours = queryResult[1]
            for i in range(1,k):
                if mode == 'horizontal':
                    ###################################
                    # Transitive Closure - Horizontal #
                    ###################################
                    #if(abs(self.characters[neighbours[i]].y-character.y) < avgNNDistance/2):
                    neighbour = self.characters[neighbours[i]]
                    line = g.Line([character.coordinate, neighbour.coordinate])
                    angle = line.calculateAngle(line.start, line.end)
                    if(abs(angle.canonical) <= 0.261799 and distances[i] < maxDistance): # 15(degree) = 0.261799(rad), 30(degree) = 0.523599(rad)
                        character.nearestNeighbours.append(neighbour)
                        nn_horizontal_distances.append(distances[i])
                        #print (i,"th nn!", "dist:", distances[i], " neighbor:(",neighbour.x,",",neighbour.y,")")
                    # Below is just for calculating the most common vertical distance purpose...
                    if(1.309 <= abs(angle.canonical) <= 1.8326 and distances[i] < maxDistance): # 75(degree)=1.309(rad), 105(degree)=1.8326(rad) 60(degree)=1.0472(rad), 90(degree)=1.5708(rad), 120(degree)=2.0944(rad)
                        nn_vertical_distances.append(distances[i])
                elif mode == 'vertical': # This code might be deleted in future..?
                    ###################################
                    # Transitive Closure - Vertical   #
                    ###################################                    
                    #if(abs(self.characters[neighbours[i]].x-character.x) < avgNNDistance/2):
                    neighbour = self.characters[neighbours[i]]
                    line = g.Line([character.coordinate, neighbour.coordinate])
                    angle = line.calculateAngle(line.start, line.end)
                    if 1.309 <= abs(angle.canonical) <= 1.8326: # 75(degree)=1.309(rad), 105(degree)=1.8326(rad) 60(degree)=1.0472(rad), 90(degree)=1.5708(rad), 120(degree)=2.0944(rad)
                        character.nearestNeighbours.append(neighbour)
                        nn_vertical_distances.append(distances[i])
#                        print (i,"th nn!", "dist:", distances[i], " neighbor:(",neighbour.x,",",neighbour.y,") angle:",angle.canonical)
                else:
                    ###################################
                    # Transitive Closure - Default    #
                    ###################################
                    # Find nn in every direction within maxDistance
                    if distances[i] < maxDistance:
                        neighbour = self.characters[neighbours[i]]
                        character.nearestNeighbours.append(neighbour)
        
        num_bins = int((numpy.max(nn_distances)-numpy.min(nn_distances)+1)/10)
        n, bins, patches = plt.hist(nn_distances, num_bins, facecolor='orange', alpha=0.5)
        print("Total %d all NNs" %len(nn_distances))
        print("average NN distance: ",avgNNDistance)
        
        num_bins = int((numpy.max(nn_horizontal_distances)-numpy.min(nn_horizontal_distances)+1)/10)
        n, bins, patches = plt.hist(nn_horizontal_distances, num_bins, facecolor='orange', alpha=0.5)
        print("Total %d hor NNs" %len(nn_horizontal_distances))
        dist_peaks = []
        n_copy = n.copy()
        n_copy[::-1].sort() # sort in reverse way
        for i in range(num_bins):
            _max_idx = numpy.where(n == n_copy[i])    # Find peak
            for j in range(len(_max_idx[0])):
                dist_peaks.append(int(bins[_max_idx[0][j]]))
        print ("Distance peaks: %s" %dist_peaks)
        avg_horizontal_nn_distance = sum(nn_horizontal_distances)/(len(nn_horizontal_distances))
        print("average NN horizontal distance: %.2f\n" %avg_horizontal_nn_distance)
            
        
        num_bins = int((numpy.max(nn_vertical_distances)-numpy.min(nn_vertical_distances)+1)/10)
        n, bins, patches = plt.hist(nn_vertical_distances, num_bins, facecolor='orange', alpha=0.5)
        print("Total %d ver NNs" %len(nn_vertical_distances))
        dist_peaks = []
        n_copy = n.copy()
        n_copy[::-1].sort() # sort in reverse way
        for i in range(num_bins):
            _max_idx = numpy.where(n == n_copy[i])    # Find peak
            for j in range(len(_max_idx[0])):
                dist_peaks.append(int(bins[_max_idx[0][j]]))
        print ("Distance peaks: %s" %dist_peaks)
        avg_vertical_nn_distance = sum(nn_vertical_distances)/(len(nn_vertical_distances))
        print("average NN vertical distance: %.2f\n" %avg_vertical_nn_distance)
        
        self.characters = sorted(self.characters, key=lambda character: character.x)
        for character in self.characters:
            #print ("Deciding wordness of (",character.x,character.y,")")
            if character.parentWord is None:
                #print ("(",character.x,character.y,") is a parent!")
                if len(character.nearestNeighbours) >= 0:
                    #print ("(",character.x,character.y,") is a word!!!!")
                    word = Word([character])
                    word.findTuples()
                    words.append(word)
        '''
        print "Total ", len(words), " words are found."
        for idx, word in enumerate(words):
            print "[",idx,"] word:"
            for idx_char, character in enumerate(word.characters):
                print "**[", idx_char, "] char info.. ", "(",character.x,",",character.y,")"
        '''        
        return words

    def paint(self, image, color=colors.BLUE):

        for character in self.characters:
            image = character.paint(image, color)    # draw a dot at the word's center of mass.

        return image

class Word:

    def __init__(self, characters=[]):
        
        self.characters = set(characters)
        self.angles = []
        self.distances = []

        for character in characters:
            character.assignParentWord(self)
            
    def findTuples(self):
        # Get tuple info ... 2/21/2018
        for character in self.characters:
            for neighbour in character.nearestNeighbours:
                line = g.Line([character, neighbour])
                angle = line.calculateAngle(line.start, line.end)
                delta = line.start-line.end
                distance = math.sqrt(delta.x**2 + delta.y**2)
                #print("START: ",line.start, " END: ", line.end, " DIST: ", distance," ANGLE_degree: ", angle.degrees(), "ANGLE_canonical: ", angle.canonical)
                self.angles.append(angle.canonical)
                #self.angles.append(angle.degrees())
                self.distances.append(distance)
                            
    def registerChildCharacter(self, character):
        self.characters.add(character)

    ''' paint '''
    ''' Draw a line between characters '''
    def paint(self, image, color=colors.YELLOW):

        for character in self.characters:
            image = character.paint(image, color)
            
            for neighbour in character.nearestNeighbours:
                line = g.Line([character, neighbour])
                image = line.paint(image, color)

        return image
        
#class Line:
#    def __init__(self, words=[]):
#        self.words = set(words)
#    def update():
        
        

