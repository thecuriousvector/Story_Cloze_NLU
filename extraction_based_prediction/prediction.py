import csv
import numpy as np
from scipy.spatial.distance import cosine

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

def load_sim_stories(file_name, max_k):

    sim_stories = []
    with open(file_name) as sim_context_file:
        csv_reader = csv.reader(sim_context_file, delimiter=" ")
        for row in csv_reader:
            story_id = int(row[0])
            sim_stories_list = sorted([(int(row[2 * i + 1]), float(row[2 * i + 2])) for i in range(0, max_k)],
                                  key=lambda x: x[1])
            sim_stories.append((story_id, sim_stories_list))
    return sim_stories


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def predict_ending(sim_stories, ending_embedding, k, story_ending):

    correct_cnt = 0
    for sim_pair in sim_stories:

        story_id = sim_pair[0]
        sim_story_list = sim_pair[1][0:k]

        first_ending_sim = []
        second_ending_sim = []

        for train_id in map(lambda x:x[0], sim_story_list):

            first_ending_sim.append(cosine(train_ending_embedding[train_id], ending_embedding[2 * story_id]))
            second_ending_sim.append(cosine(train_ending_embedding[train_id], ending_embedding[2 * story_id + 1]))

        context_sim = list(map(lambda x:x[1], sim_story_list))
        context_sim_weight = softmax(context_sim)

        first_ending_weighted_sim = np.dot(first_ending_sim, context_sim_weight)
        second_ending_weighted_sim = np.dot(second_ending_sim, context_sim_weight)

        if first_ending_weighted_sim < second_ending_weighted_sim:
            predicted_ending = 0
        else:
            predicted_ending = 1

        if story_ending[story_id][predicted_ending][1] == True:
            correct_cnt += 1

    return correct_cnt

dev_context, dev_ending = load_story("dev.csv", "dev")
test_context, test_ending = load_story("test.csv", "test")

dev_sim_stories = load_sim_stories("sim_k_dev.txt", 5)
test_sim_stories = load_sim_stories("sim_k_test.txt", 5)

train_ending_embedding = np.load("train_ending_embedding.npy")
dev_ending_embedding = np.load("dev_ending_embedding.npy")
test_ending_embedding = np.load("test_ending_embedding.npy")

print predict_ending(dev_sim_stories, dev_ending_embedding, 5, dev_ending)
print predict_ending(test_sim_stories, test_ending_embedding, 5, test_ending)
