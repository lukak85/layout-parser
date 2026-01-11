import itertools
import math
import operator

import cv2
import matplotlib.pyplot as plt
import numpy

from . import colors
from . import geometry
from . import text
from .dimension import Dimension


class Page:

    def __init__(self, image):
        self.lines = []
        self.orientations = []
        self.dists = []
        
        color_image = image.copy()
        greyscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        #_,binary_image = cv2.threshold(greyscale_image, cv2.THRESH_OTSU, colors.greyscale.WHITE, cv2.THRESH_BINARY)
        _, binary_image = cv2.threshold(greyscale_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        binary_image = cv2.bitwise_not(binary_image)
          
        self.characters = text.CharacterSet(binary_image)
        self.words = self.characters.getWords()

        self.build_docstrum()

        self.find_textline(color_image)
        
        self.image = color_image

    
    def build_docstrum(self):
        theta = []
        theta_hist = []
        dist_hist = []
        r = []
        sz = 1

        for word in self.words:
            for angle in word.angles:
                theta.append(1/2*numpy.pi-angle) # -pi/2 < x < pi/2 (1 and 4 quadrant)
                theta.append(3/2*numpy.pi-angle) # pi/2 < x < -pi/2 (2 and 3 quadrant)
                theta_hist.append(math.degrees(1/2*numpy.pi-angle))
            for distance in word.distances:
                r.append(distance)
                r.append(distance)
                dist_hist.append(distance)

        ax = plt.subplot(111,polar=True)
        plt.close()

        self.orientations = theta_hist
        self.dists = dist_hist

        ax.scatter(theta,r,sz)

    
    def find_textline(self, image):
        image = image.copy()
        for word in self.words:
            points = []
            for character in word.characters:
                points.append([character.x, character.y])
            points.sort(key=lambda x: x[0])
            dx, dy, x0, y0 = cv2.fitLine(numpy.array(points), cv2.DIST_L2, 0, 0.01, 0.01)
            start = (int(min(points,key=lambda x: x[0])[0]),int((dy/dx)*(min(points,key=lambda x: x[0])[0]-x0)+y0))
            end = (int(max(points,key=lambda x: x[0])[0]),int((dy/dx)*(max(points,key=lambda x: x[0])[0]-x0)+y0))
            self.lines.append(geometry.Line([start,end]))
            cv2.line(image, start, end, (0,255,255),2)
        return image


    def show(self):
        image = self.image.copy()
        image = self.paint(image)
        max_dimension = Dimension(800, 800)
        display_dimension = Dimension(image.shape[1], image.shape[0])
        display_dimension.fitInside(max_dimension)
        image = cv2.resize(image, tuple(display_dimension))
        cv2.imshow("Debug Image", image)
        cv2.waitKey(0)


    def paint(self, image):
        for word in self.words:
            image = word.paint(image, colors.RED)

        return image
