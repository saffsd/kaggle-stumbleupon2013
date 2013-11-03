"""
Feature extraction from raw HTML documents.

Marco Lui, September 2013
"""


import argparse, sys
import csv
import itertools

import UserDict
import os

from collections import OrderedDict, Sequence

from readability.readability import Document
from readability.htmls import build_doc
from readability.cleaners import html_cleaner

import numpy as np
from nltk.tokenize import sent_tokenize, RegexpTokenizer, word_tokenize

from mappool import MapPool

def clean_quotes(text):
 """
 Normalize unicode quotes to ASCII quotes.
 see http://www.cl.cam.ac.uk/~mgk25/ucs/quotes.html
 """
 text = text.replace(u"\u201c",u'"')
 text = text.replace(u"\u201d",u'"')
 text = text.replace(u"\u2019",u"'")
 return text

def sanitize(text):
  text = text.replace("\n"," ")
  return text

def content(html):
  doc = build_doc(html)
  text = [ clean_quotes(s.strip()) for s in doc.xpath('//text()') ]
  text = [ t.replace('\n',' ') for t in text if t ]
  text = ' '.join(text)
  return text

capword_tokenizer = RegexpTokenizer('[A-Z]\w+')
decimal_tokenizer = RegexpTokenizer('[\d\.]+')

###
# Feature Functions
###

def urlid(sudoc):
  return sudoc.ident

def len_raw(sudoc):
  return len(sudoc.raw)

def len_title(sudoc):
  return len( sudoc.doc.title() )

def str_title(sudoc):
  return sudoc.doc.title().encode('utf8')

def len_short_title(sudoc):
  return len( sudoc.doc.short_title() )

def len_content(sudoc):
  return len( sudoc.doc.content() )

def len_summary(sudoc):
  return len( sudoc.doc.summary() )

def len_paragraph(sudoc):
  paras = sudoc.html.xpath('.//p')
  retval = np.mean([len(p.text.strip()) for p in paras if p.text])
  return retval


def count_content_words(sudoc):
  words = word_tokenize(sudoc.content)
  return len(words)

def count_content_capwords(sudoc):
  capwords = capword_tokenizer.tokenize(sudoc.content)
  return len(capwords)

def count_hyperlink_abs(sudoc):
    nodes = sudoc.html.xpath('.//a[@href]')
    retval = sum(1 for n in nodes if '://' in n.attrib['href'])
    return retval

def count_hyperlink_rel(sudoc):
    nodes = sudoc.html.xpath('.//a[@href]')
    retval = sum(1 for n in nodes if '://' not in n.attrib['href'])
    return retval

from urlparse import urlparse
def str_href(sudoc):
  """
  Content of links themselves
  """
  nodes = sudoc.html.xpath('.//a[@href]')
  words = []
  for url in (n.attrib['href'] for n in nodes):
    parse = urlparse(url)
    urlwords = os.path.basename(os.path.splitext(parse.path)[0]).split('-')
    if len(urlwords) > 1:
      words.extend([w for w in urlwords if w.isalnum()])
  return ' '.join(words)

def str_sent_longest(sudoc):
  return max(sudoc.sents,key=len) if sudoc.sents else ''

def str_sent_first(sudoc):
  return sudoc.sents[0] if sudoc.sents else ''

def str_sent_firstfive(sudoc):
  return sudoc.sents[:5] if sudoc.sents else ''

def str_sent_last(sudoc):
  return sudoc.sents[-1] if sudoc.sents else ''

class NodeAttrStr(object):
  def __init__(self, nodetype, attr):
    self.nodetype = nodetype
    self.attr = attr
    self.__name__ = "str_{0}_{1}".format(nodetype, attr)

  def __call__(self, sudoc):
    nodes = sudoc.html.xpath('.//{}'.format(self.nodetype))
    text = [ n.attrib[self.attr].replace(' ','') for n in nodes if self.attr in n.attrib ]
    return clean_quotes(' '.join(text)).encode('utf8')

class NodeCount(object):
  def __init__(self, nodetype):
    self.nodetype = nodetype
    self.__name__ = "count_{}".format(nodetype)

  def __call__(self, sudoc):
    nodes = sudoc.html.xpath('.//{}'.format(self.nodetype))
    return len(nodes)

def count_sent(sudoc):
  c = sent_tokenize(sudoc.content)
  return len(c)


class Ratio(object):
  def __init__(self, numerator, denominator):
    self.numerator = numerator
    self.denominator = denominator
    self.__name__ = 'ratio_{0}_{1}'.format(numerator, denominator)

  def __call__(self, sudoc):
    try:
      val = float(sudoc[self.numerator]) / sudoc[self.denominator]
    except ZeroDivisionError:
      val = 0.0
    return val

class NodeText(object):
  def __init__(self, nodetype):
    if isinstance(nodetype, basestring):
      self.nodetype = nodetype
      self.__name__ = "str_{}".format(nodetype)
    elif isinstance(nodetype, Sequence):
      self.nodetype = "*[{}]".format(' or '.join('self::{}'.format(n) for n in nodetype))
      self.__name__ = "str_({})".format('|'.join(nodetype))
    else:
      raise ValueError

  def __call__(self, sudoc):
    nodes = sudoc.html.xpath('.//{}'.format(self.nodetype))
    text = ' '.join(n.text.strip().encode('utf8') for n in nodes if n.text)
    text = sanitize(text)
    return text

def str_content(sudoc):
  return sudoc.content

def str_content_capwords(sudoc):
  retval = ' '.join(capword_tokenizer.tokenize(sudoc.content))
  return retval

def str_content_decimal(sudoc):
  retval = ' '.join(decimal_tokenizer.tokenize(sudoc.content))
  return retval

def str_description(sudoc):
  nodes = sudoc.html.xpath('.//meta[@name="description"]')
  if len(nodes) == 0:
    return ""
  else:
    return '\n'.join(n.attrib.get('content','') for n in nodes)


###
# represent the document
###


class SUDoc(UserDict.DictMixin):
  """
  Class representing a document in the StumbleUpon challenge.
  This class is used to extract features from the document.
  """
  featfn = OrderedDict()

  @classmethod
  def register(cls, fn, name = None):
    """
    Register a feature-extracting function.
    Functions have the prototype `fn(SUDoc)` and return a value for
    that feature.
    """
    if name is None:
      name = fn.__name__
    cls.featfn[name] = fn

  @classmethod
  def from_file(cls, path):
    with open(path) as f:
      raw = f.read()
    return cls(raw, os.path.basename(path))

  def __init__(self, raw, ident):
    self.raw = raw 
    self.ident = ident 

    self.featval = {}

    self.html = build_doc(raw)
    self.doc = Document(raw)
    self.content = content(self.doc.summary())
    self.sents = sent_tokenize(self.content)

  @classmethod
  def keys(cls):
    return cls.featfn.keys()

  def __getitem__(self, key):
    try:
      return self.featval[key]
    except KeyError:
      val = self.featfn[key](self)
      self.featval[key] = val
      return val
    
def setup_extract(args):
  global __ext

  ext = SUDoc
  ext.register(urlid)

  if args.text:
    ext.register(NodeText('a[@href]'))
    ext.register(NodeText('p'))
    ext.register(NodeText('span'))
    ext.register(NodeText('div'))
    ext.register(NodeText('label'))
    ext.register(NodeText('h1'))
    ext.register(NodeText('h2'))
    ext.register(NodeText('h3'))
    ext.register(NodeText('h4'))
    ext.register(NodeText('title'))
    ext.register(NodeText('td'))
    ext.register(NodeText('li'))
    ext.register(NodeText('i'))
    ext.register(NodeText('b'))
    ext.register(NodeText('option'))
    ext.register(NodeText('strong'))
    ext.register(NodeText('em'))
    #ext.register(NodeText('script'))
    ext.register(NodeText(['h1','h2','h3','h4','h5','h6']))
    ext.register(NodeText(['b','i','u','em','strong']),'str_emph')
    ext.register(NodeText(['th','tr','td']))
    ext.register(NodeText(['ol','ul','li']))
    ext.register(str_content)
    ext.register(str_content_capwords)
    ext.register(str_content_decimal)
    ext.register(str_description)
    ext.register(str_href)
    ext.register(str_sent_longest)
    #ext.register(str_img_filename)
    #ext.register(str_sent_first)
    #ext.register(str_sent_firstfive)
    #ext.register(str_sent_last)
    #ext.register(NodeAttrStr('div','id'))
    #ext.register(NodeAttrStr('div','class'))
    #ext.register(NodeAttrStr('span','class'))
    #ext.register(NodeAttrStr('meta','content'))
    ext.register(NodeAttrStr('img','alt'))

  if args.len:
    ext.register(len_raw)
    ext.register(len_title)
    ext.register(len_short_title)
    ext.register(len_content)
    ext.register(len_summary)

  if args.count:
    ext.register(NodeCount('script'))
    ext.register(NodeCount('style'))
    ext.register(NodeCount('link'))
    ext.register(NodeCount('p'))
    ext.register(NodeCount('div'))
    ext.register(NodeCount('meta'))
    ext.register(NodeCount('table'))
    ext.register(NodeCount('ul'))
    ext.register(NodeCount('ol'))
    ext.register(NodeCount('li'))
    ext.register(NodeCount('tr'))
    ext.register(NodeCount('img'))
    ext.register(NodeCount('embed'))
    ext.register(NodeCount('a'), 'count_hyperlink')
    ext.register(count_hyperlink_abs)
    ext.register(count_hyperlink_rel)
    ext.register(count_sent)
    ext.register(count_content_words)
    ext.register(count_content_capwords)

    ext.register(Ratio('count_content_capwords', 'count_content_words'))
    ext.register(Ratio('count_content_words', 'count_sent'))
    ext.register(Ratio('count_li', 'count_ul'))
    ext.register(Ratio('count_li', 'count_ol'))
    ext.register(Ratio('count_tr', 'count_table'))

  __ext = ext

def extract(path):
  global __ext
  try:
    row = dict(__ext.from_file(path))
  except Exception as e:
    print >>sys.stderr, "path: {0} e: {1}".format(path, e)
    raise
  retval = {k:(row[k].encode('utf8') if isinstance(row[k], unicode) else row[k]) for k in row}
  return retval

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--jobs","-j", type=int)

  subsets = parser.add_argument_group("subsets", "select optional feature subsets")
  subsets.add_argument("--text", action="store_true", help="include text features")
  subsets.add_argument("--count", action="store_true", help="include count features")
  subsets.add_argument("--len", action="store_true", help="include len features")

  parser.add_argument("input", nargs="*")
  parser.add_argument("--output","-o",type=argparse.FileType('w'), default=sys.stdout)
  args = parser.parse_args()

  if args.input:
    paths = list(args.input)
  else:
    paths = map(str.strip, sys.stdin.readlines())

  def p_iter():
    for i,path in enumerate(paths):
      yield path
      if args.output and i and not(i % 1000):
        # TODO: Rate
        print "processed {}/{}".format(i, len(paths))


  setup_extract(args)
  global __ext

  writer = csv.DictWriter(args.output, __ext.keys(),delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
  writer.writeheader()
  with MapPool(args.jobs, setup_extract, (args,), chunksize=20) as p:
    for record in p(extract, p_iter()):
      try:
        writer.writerow(record)

      except UnicodeEncodeError:
        import pdb; pdb.post_mortem()





