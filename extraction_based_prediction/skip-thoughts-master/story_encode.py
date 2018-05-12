import csv
import skipthoughts
import nltk
import numpy as np


data_path = "../../../data/"


def load_story(path, type):

    story_context = []
    story_ending = []

    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)

        for row in csv_reader:
            if type == "train":
                story_context.append(row[2: 6])
                story_ending.append((row[6], True))
            else:
                story_context.append(row[1: 5])
                correct_ending = int(row[7])
                if correct_ending == 1:
                    story_ending.append(((row[5], True), (row[6], False)))
                else:
                    story_ending.append(((row[5], False), (row[6], True)))


    return story_context, story_ending

train_context, train_ending = load_story(data_path+"train.csv", "train")
dev_context, dev_ending = load_story(data_path+"dev.csv", "dev")
test_context, test_ending = load_story(data_path+"test.csv", "test")

embedding_dim = 4800
n_train = len(train_ending)

train_ending_embedding = np.zeros((n_train, embedding_dim))

def story_encode(name, story_set):
        
    for i in range(0, len(story_set)):
        print "#"+str(i)
        train_ending_embedding[i] = encoder.encode(story_set[i])
        encoder.encode(story_set[i])
        
    np.save(name, train_ending_embedding)

model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)

story_encode("train_context_embedding", train_context)
story_encode("train_ending_embedding", map(lambda x: [x[0]], train_ending))

story_encode("dev_context_embedding", dev_context)
dev_ending_sentence = [ending for ending_pair in map(lambda x: [[x[0][0]], [x[1][0]]], dev_ending) for ending in ending_pair]
story_encode("dev_ending_embedding", dev_ending_sentence)


story_encode("test_context_embedding", test_context)
dev_ending_sentence = [ending for ending_pair in map(lambda x: [[x[0][0]], [x[1][0]]], test_ending) for ending in ending_pair]
story_encode("test_ending_embedding", dev_ending_sentence)




    
    
