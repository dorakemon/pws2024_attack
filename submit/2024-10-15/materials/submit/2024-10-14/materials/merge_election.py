import pandas as pd

E16 = pd.read_csv("./E16.csv", header=None)

election_15 = pd.read_csv("./output_15.csv")
election_15 = election_15.sort_values(by="15")

election_19 = pd.read_csv("./output_19_11.csv")
election_19 = election_19.sort_values(by="19")

election_15_match = election_15["15"].index.values

election_19_match = election_19["19"].index.values

maeattack_15_match = E16[28]
maeattack_15_anaume = E16[29]

maeattack_19_match = E16[36]
maeattack_19_anaume = E16[37]


def matched_count(a_list, b_list):
    return sum([1 for a, b in zip(a_list, b_list) if a == b])

print(matched_count(election_15_match, maeattack_15_match))

print(matched_count(election_19_match, maeattack_19_match))

E16[28] = election_15_match
E16[36] = election_19_match

election_15 = election_15.reset_index(drop=True)
election_15_anaume = election_15["16"]
E16[29] = election_15_anaume

election_19 = election_19.reset_index(drop=True)
election_19_anaume = election_19["20"]
E16[37] = election_19_anaume

print(matched_count(election_15_anaume, maeattack_15_anaume))
print(matched_count(election_19_anaume, maeattack_19_anaume))

E16.to_csv("./E16_merge_election.csv", header=None, index=None)

# print(election_19_match)