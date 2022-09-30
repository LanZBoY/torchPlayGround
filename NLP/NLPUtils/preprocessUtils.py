import os
from typing import Iterable
def removeWord(removeWordList: list, sentence : Iterable):
    for removeWord in removeWordList:
        sentence = list(filter(removeWord.__ne__,sentence))
    return sentence

if __name__ == '__main__':
    print(removeWord(removeWordList='./NLP/NLPUtils/english.txt', sentence=['not', 'man', 'dare' , 'b']))