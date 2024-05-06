from collections import deque
from tqdm import tqdm
import numpy as np
import pandas as pd
import sys
from numpy.linalg import norm


def pref_to_rank(pref):
    return {a: {b: idx for idx, b in enumerate(a_pref)} for a, a_pref in pref.items()}


def get_difference(feature1, feature2):
    euclidean_loss = np.power(feature1 - feature2, 2)
    euclidean_loss = np.sqrt(np.sum(euclidean_loss))
    return euclidean_loss


def get_cosine_similarity(feature1, feature2):
    return np.dot(feature1, feature2) / (norm(feature1) * norm(feature2))


def gale_shapley(*, A, B, A_pref, B_pref):
    B_rank = pref_to_rank(B_pref)
    ask_list = {a: deque(bs) for a, bs in A_pref.items()}
    pair = {}
    remaining_A = set(A)
    while len(remaining_A) > 0:
        a = remaining_A.pop()
        b = ask_list[a].popleft()  # Adjusted to consider id from tuple
        if b not in pair:
            pair[b] = a
        else:
            a0 = pair[b]
            b_prefer_a0 = B_rank[b][a0] < B_rank[b][a]
            if b_prefer_a0:
                remaining_A.add(a)
            else:
                remaining_A.add(a0)
                pair[b] = a
    return [(a, b) for b, a in pair.items()]


# python stable_matching_percentile.py "C:/Users/ewang/OneDrive/Desktop/Spring_2024/COS_429/Final/light_features.npy" "C:/Users/ewang/OneDrive/Desktop/Spring_2024/COS_429/Final/dark_features.npy"
def main(args):
    df = pd.read_csv("../annotations/images_val2014.csv")
    light_id = list(df[df["bb_skin"] == "Light"]["id"])
    dark_id = list(df[df["bb_skin"] == "Dark"]["id"])

    mode = args[1]

    # l_feat = np.load(args[1], allow_pickle=True)
    # d_feat = np.load(args[2], allow_pickle=True)
    l_feat = np.load(
        "C:/Users/ewang/OneDrive/Desktop/Spring_2024/COS_429/Final/light_features.npy"
    )
    d_feat = np.load(
        "C:/Users/ewang/OneDrive/Desktop/Spring_2024/COS_429/Final/dark_features.npy"
    )

    l_pref, d_pref = {}, {}

    sim_dict = {}
    for index_i, i in tqdm(enumerate(dark_id), total=len(dark_id)):
        if mode == "cosine":
            dist = [
                (j, get_cosine_similarity(d_feat[index_i], l_feat[index_j]))
                for index_j, j in enumerate(light_id)
            ]
            dist.sort(reverse=True)  # Sort by cos sim
        elif mode == "distance":
            dist = [
                (j, get_difference(d_feat[index_i], l_feat[index_j]))
                for index_j, j in enumerate(light_id)
            ]
            dist.sort()  # Sort by distance

        sim_dict[i] = (
            dist  # dict with keys as image ids, values as a list of tuples with (light_id, diff)
        )

        d_pref[i] = [id for id, _ in dist]

    for index_i, i in tqdm(enumerate(light_id), total=len(light_id)):
        if mode == "cosine":
            dist = [
                (get_cosine_similarity(l_feat[index_i], d_feat[index_j]), j)
                for index_j, j in enumerate(dark_id)
            ]
            dist.sort(reverse=True)  # Sort by distance
        elif mode == "distance":
            dist = [
                (get_difference(l_feat[index_i], d_feat[index_j]), j)
                for index_j, j in enumerate(dark_id)
            ]
            dist.sort()  # Sort by distance

        l_pref[i] = [id for _, id in dist]

    pairs = gale_shapley(
        B=set(light_id),
        A=set(dark_id),
        B_pref=l_pref,
        A_pref=d_pref,
    )

    # Convert pairs to DataFrame
    matched_df = pd.DataFrame(pairs, columns=["dark_id", "light_id"])

    # Load similarities from preferences
    similarity_dict = {i: dict(sim_dict[i]) for i in dark_id}
    if mode == "cosine":
        matched_df["cosine_similarity"] = matched_df.apply(
            lambda x: similarity_dict[x["dark_id"]][x["light_id"]], axis=1
        )
        percentile_threshold = np.percentile(
            matched_df["cosine_similarity"], 90
        )  # Adjust percentile as needed
        filtered = matched_df[matched_df["cosine_similarity"] >= percentile_threshold]

    elif mode == "distance":
        matched_df["distance"] = matched_df.apply(
            lambda x: similarity_dict[x["dark_id"]][x["light_id"]], axis=1
        )
        percentile_threshold = np.percentile(
            matched_df["distance"], 10
        )  # Adjust percentile as needed
        filtered = matched_df[matched_df["distance"] <= percentile_threshold]

    # Filter by top similarity percentile

    matched_df.to_csv("sim_stable_percentiles.csv", index=False)

    filtered.to_csv("sim_stable_filtered.csv", index=False)


if __name__ == "__main__":
    main(sys.argv)
