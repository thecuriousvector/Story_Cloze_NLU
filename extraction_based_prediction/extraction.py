import numpy as np
from scipy.spatial.distance import cosine
import heapq

def extract_similar_story(story_context_embedding, k):

    k_similar_story = []
    heapq.heapify(k_similar_story)
    max_topk_sim = np.inf

    for train_story_id in range(0, n_train):
        sim = 0
        for i in range(0, context_len):
            sim += cosine(train_context_embedding[4 * train_story_id + i], story_context_embedding[i])
        if sim < max_topk_sim:
            if len(k_similar_story) < k:
                heapq.heappush(k_similar_story, (-sim, train_story_id))
            else:
                heapq.heapreplace(k_similar_story, (-sim, train_story_id))
            max_topk_sim = -k_similar_story[0][0]

    return [(-x[0], x[1]) for x in k_similar_story]


def story_set_extraction(embedding, file_name, k):
    sim_stories = {}

    for i in range(0, 3):
        k_sim_story = extract_similar_story(embedding[4 * i: 4 * i + 4], k)
        sim_stories.update({i: k_sim_story})

    sim_sories_file = open(file_name, "w+")
    for story_id in sim_stories:
        sim_sories_file.write(str(story_id) + " ")
        for train_id_sim in sim_stories[story_id]:
            sim_sories_file.write(str(train_id_sim[0]) + " " + str(train_id_sim[1]) + " ")
        sim_sories_file.write("\n")
    sim_sories_file.close()

train_context_embedding = np.load("train_context.npy")
dev_context_embedding = np.load("dev_context.npy")
test_context_embedding = np.load("test_context.npy")

n_train = train_context_embedding.shape[0]
context_len = 4
k = 5

story_set_extraction(dev_context_embedding, "dev_sim.txt", k)
story_set_extraction(test_context_embedding, "test_sim.txt", k)
