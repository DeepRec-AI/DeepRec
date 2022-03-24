#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc


class Request(object):
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def __str__(self):
    raise NotImplementedError('__str__() must be defined')

  @abc.abstractmethod
  def to_string(self):
    raise NotImplementedError('to_string() must be defined')

  @abc.abstractmethod
  def parse_response(self):
    raise NotImplementedError('parse_response() must be defined')


class Response(object):
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def __str__(self):
    raise NotImplementedError('__str__() must be defined')

