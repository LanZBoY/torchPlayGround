class Vocabulary():
    def __init__(self, TOKENS : dict() = {}, MAXLEN = None) -> None:
        self.num_words = len(TOKENS)
        self.word2idx = TOKENS
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.wordCounter = {}
        self.MAXLEN = MAXLEN
        
    def addWord(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.num_words
            self.idx2word[self.num_words] = word
            self.num_words += 1

    def addSentence(self, sentence):
        sent_list = sentence.split()[:self.MAXLEN]
        for word in sent_list:
            self.addWord(word)
        return sent_list
            
    def addWordList(self, wordList):
        for word in wordList[:self.MAXLEN]:
            self.addWord(word)
        return wordList[:self.MAXLEN]

    def has(self, word):
        return word in self.word2idx