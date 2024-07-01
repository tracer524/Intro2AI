#!/usr/bin/env python3

import random, sys, math

from engine.const import Const
import graderUtil
import util
import collections
import copy

graderUtil.TOLERANCE = 1e-3
grader = graderUtil.Grader()
submission = grader.load('submission')

# General Notes:
# - Unless otherwise specified, all parts time out in 1 second.

############################################################
# Problem 1: Emission probabilities

def test1d():
    ei = submission.ExactInference(10, 10)
    ei.skipElapse = True ### ONLY FOR PROBLEM 1
    ei.observe(55, 193, 200)
    grader.require_is_equal(0.030841805296, ei.belief.getProb(0, 0))
    grader.require_is_equal(0.00073380582967, ei.belief.getProb(2, 4))
    grader.require_is_equal(0.0269846478431, ei.belief.getProb(4, 7))
    grader.require_is_equal(0.0129150762582, ei.belief.getProb(5, 9))

    ei.observe(80, 250, 150)
    grader.require_is_equal(0.00000261584106271, ei.belief.getProb(0, 0))
    grader.require_is_equal(0.000924335357194, ei.belief.getProb(2, 4))
    grader.require_is_equal(0.0295673460685, ei.belief.getProb(4, 7))
    grader.require_is_equal(0.000102360275238, ei.belief.getProb(5, 9))

grader.add_basic_part('1d-0-basic', test1d, 2, description="1d basic test for emission probabilities")

def test1d_1(): # test whether they put the pdf in the correct order
    oldpdf = util.pdf
    del util.pdf
    def pdf(a, b, c): # be super rude to them! You can't swap a and c now!
      return a + b
    util.pdf = pdf

    ei = submission.ExactInference(10, 10)
    ei.skipElapse = True ### ONLY FOR PROBLEM 1
    ei.observe(55, 193, 200)
    grader.require_is_equal(0.012231949648, ei.belief.getProb(0, 0))
    grader.require_is_equal(0.00982248065925, ei.belief.getProb(2, 4))
    grader.require_is_equal(0.0120617259453, ei.belief.getProb(4, 7))
    grader.require_is_equal(0.0152083233155, ei.belief.getProb(5, 9))

    ei.observe(80, 250, 150)
    grader.require_is_equal(0.0159738258744, ei.belief.getProb(0, 0))
    grader.require_is_equal(0.00989135100651, ei.belief.getProb(2, 4))
    grader.require_is_equal(0.0122435075636, ei.belief.getProb(4, 7))
    grader.require_is_equal(0.018212043367, ei.belief.getProb(5, 9))
    util.pdf = oldpdf # replace the old pdf

grader.add_basic_part('1d-1-basic', test1d_1, 2, description="1d test ordering of pdf")

def test1d_2():
    random.seed(10)

    ei = submission.ExactInference(10, 10)
    ei.skipElapse = True ### ONLY FOR PROBLEM 1

    N = 50
    p_values = []
    for i in range(N):
      a = int(random.random() * 300)
      b = int(random.random() * 5)
      c = int(random.random() * 300)

      ei.observe(a, b, c)

      for d in range(10):
        for e in range(10):
          p_values.append(ei.belief.getProb(d, e))

grader.add_hidden_part('1d-2-hidden', test1d_2, 3, description="1d advanced test for emission probabilities")

############################################################
# Problem 2: Transition probabilities


def test2d():
    ei = submission.ExactInference(30, 13)
    ei.elapseTime()
    grader.require_is_equal(0.0105778989624, ei.belief.getProb(16, 6))
    grader.require_is_equal(0.00250560512469, ei.belief.getProb(18, 7))
    grader.require_is_equal(0.0165024135157, ei.belief.getProb(21, 7))
    grader.require_is_equal(0.0178755550388, ei.belief.getProb(8, 4))

    ei.elapseTime()
    grader.require_is_equal(0.0138327373012, ei.belief.getProb(16, 6))
    grader.require_is_equal(0.00257237608713, ei.belief.getProb(18, 7))
    grader.require_is_equal(0.0232612833688, ei.belief.getProb(21, 7))
    grader.require_is_equal(0.0176501876956, ei.belief.getProb(8, 4))

grader.add_basic_part('2d-0-basic', test2d, 2, description="test correctness of elapseTime()")

def test2d_1i(): # stress test their elapseTime
    A = 30
    B = 30
    random.seed(15)
    ei = submission.ExactInference(A, B)

    N1 = 20
    N2 = 400
    p_values = []
    for i in range(N1):
      ei.elapseTime()
      for i in range(N2):
        d = int(random.random() * A)
        e = int(random.random() * B)
        p_values.append(ei.belief.getProb(d, e))


grader.add_hidden_part('2d-1i-hidden', test2d_1i, 2, description="advanced test for transition probabilities, strict time limit", max_seconds=5)

def test2d_1ii(): # stress test their elapseTime, making sure they didn't specifically use lombard
    random.seed(15)

    oldworld = Const.WORLD
    Const.WORLD = 'small' # well... they may have made it specific for lombard

    A = 30
    B = 30
    ei = submission.ExactInference(A, B)

    N1 = 20
    N2 = 40
    p_values = []
    for i in range(N1):
      ei.elapseTime()
      for i in range(N2):
        d = int(random.random() * A)
        e = int(random.random() * B)
        p_values.append(ei.belief.getProb(d, e))


grader.add_hidden_part('2d-1ii-hidden', test2d_1ii, 1, description="2d test for transition probabilities on other maps, loose time limit", max_seconds=20)

def test2d_2(): # let's test them together! Very important
    # This assumes the rest of the tests will be run on lombard
    Const.WORLD = 'lombard' # set it to lombard in case the previous test times out
    random.seed(20)

    A = 30
    B = 30
    ei = submission.ExactInference(A, B)

    N1 = 20
    N2 = 400
    p_values = []
    for i in range(N1):
      ei.elapseTime()

      a = int(random.random() * 5 * A)
      b = int(random.random() * 5)
      c = int(random.random() * 5 * A)

      ei.observe(a, b, c)
      for i in range(N2):
        d = int(random.random() * A)
        e = int(random.random() * B)
        p_values.append(ei.belief.getProb(d, e))



grader.add_hidden_part('2d-2-hidden', test2d_2, 2, description="advanced test for emission AND transition probabilities, strict time limit", max_seconds=5)



grader.grade()
