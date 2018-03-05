import csv
import random

class Data :
    def __init__(self, fileName):
        self.dataFileName = fileName
        self.dataDictionary = self.processData()
        self.classIndices, self.classCount = self.getClassIndices()
        self.wordCount = self.countWordInEachClass()
        self.vocabSize = self.getVocabSize()
        self.train_test_split(0.2)

    def processData(self):
        doc_dictornary = dict()
        doc_no = 1

        with open(self.dataFileName, 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                if row[0] != '':
                    doc_dictornary[doc_no] = row[1], row[0]
                    doc_no = doc_no + 1

        return doc_dictornary

    def train_test_split(self, ratio):
        keys = list(self.dataDictionary.keys())
        split_index = int(ratio * len(self.dataDictionary))
        random.shuffle(keys)
        self.dataDictionary = {key : self.dataDictionary[key] for key in keys}
        self.trainSet = dict(self.dataDictionary.items()[0 : split_index])
        self.testSet = dict(self.dataDictionary.items()[split_index :])

    def getVocabSize(self):
        count = 0
        for keys in self.dataDictionary:
            count = count + len(self.dataDictionary[keys][0])
        return count

    def getClassIndices(self):
        classDictionary = dict()
        classCount = dict()
        elemIndex = 1

        for key in self.dataDictionary.keys():
            category = self.dataDictionary[key][1]

            if category in classDictionary:
                classDictionary[category] = classDictionary[category] + (key, )
                classCount[category] = classCount[category] + len(self.dataDictionary[key][0])
            else:
                classDictionary[category] = (key, )
                classCount[category] = len(self.dataDictionary[key][0])

            elemIndex = elemIndex + 1

        return classDictionary, classCount

    def countWordInEachClass(self):
        wordModel = dict()
        for c in self.classIndices.keys():
            for indices in self.classIndices[c]:
                for word in self.dataDictionary[indices][0]:
                    if (word, c) in wordModel:
                        wordModel[word, c] = wordModel[word, c] + 5
                    else:
                        wordModel[word, c] = 5

        return wordModel

    def getLikelihood(self, w, c):
        if (w, c) in self.wordCount:
            return (self.wordCount[w, c] + 1) * 1.0/ (self.classCount[c] + self.vocabSize)
        else:
            return 1.0/ (self.classCount[c] + self.vocabSize)

    def getClassPrior(self, c):
        return 1.0 * self.classCount[c]/self.classIndices.__len__()

    def getDocProbability(self, listofWords, c):
        p = 1.0
        for word in listofWords:
            p = p * self.getLikelihood(word, c)

        return p * self.getClassPrior(c)

    def predict(self, doc):
        p = 0.0
        pred_c = None
        for c in self.classIndices.keys():
            temp = self.getDocProbability(doc, c)
            if p < temp:
                p = temp
                pred_c = c

        return pred_c

    def getMetirces(self):
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for key in self.dataDictionary.keys():
            category = self.dataDictionary[key][1]
            pred = self.predict(self.dataDictionary[key][0])
            if category == 'spam':
                if pred == 'spam':
                    TP = TP + 1
                elif pred == 'ham':
                    FN = FN + 1
            elif pred == 'ham':
                if pred == 'spam':
                    FP = FP + 1
                elif pred == 'ham':
                    TN = TN + 1

        print(TP, TN, FP, FN)
        precision = TP * 1.0 / (TP + FP)
        recall = TP * 1.0/ (TP + FN)
        return precision, recall, 2 * precision * recall / (precision + recall)

def main():
    docs = Data('data/spam.csv')
    print docs.getMetirces()

if __name__ == '__main__':
    main()