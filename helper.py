#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Copyright (C) 2018 FI-UNER Robotic Group

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

@mail: robotica@ingenieria.uner.edu.ar
"""
import cv2
import numpy as np


def wrap_digit(rect, padding = 3, center = True):
    x, y, w, h = rect
#    padding = 5q
    if center:
        hcenter = int(x + w/2)
        vcenter = int(y + h/2)
        if (h > w):
            w = h
            x = hcenter - int(w/2)
        else:
            h = w
            y = vcenter - int(h/2)
    return (x - padding, y - padding, w + 2*padding, h + 2*padding)

def findBiggestContour2(mask):
    temp_bigger = []
    img1, cont, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(cont) == 0:
        return False
    for cnt in cont:
        temp_bigger.append(cv2.contourArea(cnt))
    greatest = max(temp_bigger)
    index_big = temp_bigger.index(greatest)
    key = 0
    for cnt in cont:
        if key == index_big:
            return cnt
            break
        key += 1


def findBiggestContour(mask):
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    board = []
    if hierarchy is None: return False
    for cnt, hie in zip(contours, hierarchy[0,:,:]):
        if hie[3] == -1 and hie[2] != -1:
           board.append(cnt)
    if board == []:
        return False
    else:
        return board
def findContour(mask):
    img1, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def findCells(mask):
    ext, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    celdas = []
    for cnt, hie in zip(contours, hierarchy[0,:,:]):
        if hie[3] == 0:
            celdas.append(cnt)
    return celdas

def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0

	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True

	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1

	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))

	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)
