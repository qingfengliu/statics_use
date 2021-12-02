from nltk.util import ngrams
import re
sentence="""Theoms Jefferson began building Monticello at the age of 26."""
patter=re.compile(r"([-\s.,;!?])+")
tokens=patter.split(sentence)
tokens=[x for x in tokens if x and x not in '- \t\n.,;!?']
print(tokens)
print(list(ngrams(tokens,2)))