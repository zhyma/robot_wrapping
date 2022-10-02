import cv2
import numpy as np
import copy
from math import sqrt

from skimage.morphology import skeletonize
import operator

class node():
    def __init__(self, xy, parent):
        self.xy = xy
        self.parent = None
        self.len = 0
        if parent is not None:
            self.parent = parent
            self.len = parent.len + 1
            
        self.n_children = 0

def bfs(img, start, connections, skip_list):
    ## start: [x,y]
    ## connections example: [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1]]
    ## skip_list: [[x1, y1], [x2, y2], [x3, y3], ...]
    [height, width] = img.shape

    search = True
    node0 = node(start, None)
    frontier = [node0]
    visited_list = []
    ## found a starting point, search for the rope
    ## 5 connection (always search for those above or by its' side)
    while search:
        l = len(frontier)
        ## search on all frontier nodes, 
        ## move down one level (if it's child exist),
        ## or delete the frontier node (if no child, prune greadily)
        for i in range(l-1, -1, -1):
            curr_node = frontier[i]
            for next in  connections:
                x = curr_node.xy[0] + next[0]
                y = curr_node.xy[1] + next[1]
                n_children = 0
                
                visited = False
                for j in (visited_list+skip_list):
                    if [x,y] == j:
                        visited = True

                ## search for valid kids
                if visited:
                    ## skip any visited
                    continue
                if (x < 0) or (y < 0) or (x > width-1) or (y > height-1):
                    ## skip those out of the boundary
                    continue
                if img[y, x] < 100:
                    ## skip those not being marked
                    continue
                
                ## those are the children of the current node
                n_children += 1
                new_node = node([x,y], curr_node)
                frontier.append(new_node)
                visited_list.append([x,y])

                if n_children < 1:
                    ## reach the edge of the image, does not have a child  
                    curr_node.n_children = -1
                else:
                    curr_node.n_children = n_children

            if len(frontier) > 1:
                ## more than one frontier node left, the other one must has the same length
                ## (edges between nodes are equally weighted)
                frontier.pop(i)
            else:
                ## no other frontier node left, stop searching
                search = False
    
    ## rope is [[x0,y0],[x1,y1],...,[xn,yn]]
    rope = []
    i_node = frontier[0]
    while i_node.parent is not None:
        rope = [i_node.xy] + rope
        i_node = i_node.parent

    return rope, visited_list

