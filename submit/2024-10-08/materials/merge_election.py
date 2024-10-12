import pandas as pd

E16 = pd.read_csv("./E16.csv", header=None)
election_15_answers = [[29, 1], [18, 5], [39, 1], [37, 5], [34, 1], [3, 0], [14, 2], [25, 5], [44, 1], [31, 3], [36, 4], [24, 1], [49, 0], [42, 0], [27, 0], [23, 0], [20, 0], [46, 2], [0, 0], [8, 1], [19, 5], [38, 5], [10, 3], [40, 0], [1, 2], [9, 2], [47, 0], [26, 0], [45, 0], [33, 4], [5, 2], [4, 5], [6, 2], [13, 1], [22, 1], [2, 1], [17, 5], [32, 4], [7, 0], [15, 2], [21, 2], [48, 4], [43, 4], [30, 5], [35, 0], [12, 0], [41, 4], [16, 0], [28, 3], [11, 4]]

election_15_match = [ans[0] for ans in election_15_answers]
election_15_anaume = [ans[1] for ans in election_15_answers]

maeattack_match = E16[28]
maeattack_anaume = E16[29]

def matched_count(a_list, b_list):
    return sum([1 for a, b in zip(a_list, b_list) if a == b])

print(matched_count(election_15_match, maeattack_match))
print(matched_count(election_15_anaume, maeattack_anaume))

E16[28] = election_15_match
E16[29] = election_15_anaume

E16.to_csv("./E16_merge_election.csv", header=None, index=None)