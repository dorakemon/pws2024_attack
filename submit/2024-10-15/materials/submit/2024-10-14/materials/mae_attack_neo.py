#!/usr/bin/env python
# coding: utf-8

# In[101]:


import warnings

import numpy as np
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)


MAIN = True # True: メインの攻撃, False: 予備選の攻撃

all_answer_from_mae_attack = []

for ATTACK_TARGET in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22']:

    # if ATTACK_TARGET == "15" or ATTACK_TARGET == "19":
    #     all_answer_from_mae_attack.append(["?" for _ in range(50)])
    #     all_answer_from_mae_attack.append(["?" for _ in range(50)])
    #     continue

    if ATTACK_TARGET == "16" or ATTACK_TARGET == "13":
        all_answer_from_mae_attack.append(["*" for _ in range(50)])
        all_answer_from_mae_attack.append(["*" for _ in range(50)])
        continue


    DIR = "../../../data" if MAIN else "../../../predata/anonymization_data"

    ALL_HEADER = ['Name', 'Gender', 'Age', 'Occupation', 'ZIP-code', '2', '56', '247', '260', '653', '673', '810', '885',  # noqa: E501
                        '1009', '1073', '1097', '1126', '1525', '1654', '1702', '1750', '1881', '1920', '1967', '2017',  # noqa: E501
                        '2021', '2043', '2086', '2087', '2093', '2100', '2105', '2138', '2143', '2174', '2193', '2253',  # noqa: E501
                        '2399', '2628', '2797', '2872', '2968', '3393', '3438', '3439', '3440', '3466', '3479', '3489',  # noqa: E501
                        '3877', '3889'] # noqa

    B_HEADERS_LIST = [
    ['Gender', 'Age', 'Occupation', 'ZIP-code', '260', '653', '1525', '2105', '2193', '2253', '2628', '2872', '3438', '3439', '3440', '3877', '3889'],  # noqa: E501
    ['Gender', 'Age', 'Occupation', 'ZIP-code', '2', '56', '260', '653', '673', '1009', '1073', '1525', '1750', '1881', '1967', '2043', '2093', '2105', '2143', '2193', '2399', '2628', '2968', '3479', '3489', '3877', '3889'],  # noqa: E501
    ['Gender', 'Age', 'Occupation', 'ZIP-code', '673', '1881', '1920', '2087', '2138'],  # noqa: E501
    ['Gender', 'Age', 'Occupation', 'ZIP-code', '2', '56', '673', '810', '885', '1009', '1073', '1097', '1525', '1654', '1702', '1750', '1881', '1920', '1967', '2017', '2043', '2087', '2093', '2138', '2399', '3438', '3439', '3440'],  # noqa: E501
    ['Gender', 'Age', 'Occupation', 'ZIP-code', '673', '810', '1073', '1126', '1702', '2100', '2174', '2253', '2797', '3393', '3466'],  # noqa: E501
    ['Gender', 'Age', 'Occupation', 'ZIP-code', '247', '885', '1097', '1654', '2086', '2138', '2872'],  # noqa: E501
    ['Gender', 'Age', 'Occupation', 'ZIP-code', '247', '2100', '2143', '2872', '3479'],  # noqa: E501
    ['Gender', 'Age', 'Occupation', 'ZIP-code', '260', '1097', '1750', '2021', '2093', '2105', '2628', '2968'],  # noqa: E501
    ['Gender', 'Age', 'Occupation', 'ZIP-code', '247', '1920', '2017', '2087'],  # noqa: E501
    ['Gender', 'Age', 'Occupation', 'ZIP-code', '260', '1097', '2628', '2174', '2797', '1073', '2100', '2968', '2105', '2193'],  # noqa: E501
    ]

    B_USER_ATTRIBUTE_HEADERS = ['Gender', 'Age', 'Occupation', 'ZIP-code']

    B_REVIEW_HEADERS_LIST = [
    ['260', '653', '1525', '2105', '2193', '2253', '2628', '2872', '3438', '3439', '3440', '3877', '3889'],  # noqa: E501
    ['2', '56', '260', '653', '673', '1009', '1073', '1525', '1750', '1881', '1967', '2043', '2093', '2105', '2143', '2193', '2399', '2628', '2968', '3479', '3489', '3877', '3889'],  # noqa: E501
    ['673', '1881', '1920', '2087', '2138'],  # noqa: E501
    ['2', '56', '673', '810', '885', '1009', '1073', '1097', '1525', '1654', '1702', '1750', '1881', '1920', '1967', '2017', '2043', '2087', '2093', '2138', '2399', '3438', '3439', '3440'],  # noqa: E501
    ['673', '810', '1073', '1126', '1702', '2100', '2174', '2253', '2797', '3393', '3466'],  # noqa: E501
    ['247', '885', '1097', '1654', '2086', '2138', '2872'],  # noqa: E501
    ['247', '2100', '2143', '2872', '3479'],  # noqa: E501
    ['260', '1097', '1750', '2021', '2093', '2105', '2628', '2968'],  # noqa: E501
    ['247', '1920', '2017', '2087'],  # noqa: E501
    ['260', '1097', '2628', '2174', '2797', '1073', '2100', '2968', '2105', '2193'],  # noqa: E501
    ]

    MOVIE_IDS = ['2', '56', '247', '260', '653', '673', '810', '885', '1009', '1073', '1097', '1126', '1525', '1654', '1702', '1750', '1881', '1920', '1967', '2017', '2021', '2043', '2086', '2087', '2093', '2100', '2105', '2138', '2143', '2174', '2193', '2253', '2399', '2628', '2797', '2872', '2968', '3393', '3438', '3439', '3440', '3466', '3479', '3489', '3877', '3889']
    # fmt: on


    ### Cデータの読み取り
    from os import listdir

    #c0~c9に対するdfのリスト
    c_data_list = []

    files = listdir(DIR)

    for i in range(10):
        file = None
        for f in files:
            if f.startswith(f"C{ATTACK_TARGET}_{i}"):
                file = f
                break
        if file is None:
            print(f)
            raise("File not found")
        c_data = pd.read_csv(f"{DIR}/C{ATTACK_TARGET}_{i}.csv")
        c_data_list.append(c_data)

    c_data_list[0]

    Ba = pd.read_csv(f"{DIR}/B{ATTACK_TARGET}a.csv")
    Bb = pd.read_csv(f"{DIR}/B{ATTACK_TARGET}b.csv")

    cross_tab_pairs = []

    for gaoz_header in B_USER_ATTRIBUTE_HEADERS:
        for movie_id in MOVIE_IDS:
            cross_tab_pairs.append((gaoz_header, movie_id))

    # 1. c0からc9までのデータを結合
    combined_data = pd.concat(c_data_list, ignore_index=True)
    combined_data.astype("category")
    for col in MOVIE_IDS:
        combined_data[col] = pd.Categorical(combined_data[col], categories=[0, 1, 2, 3, 4, 5], ordered=True)


    # クロス集計表を作成し、P(評価 | GAOZ属性)を計算
    cross_tabs = {}
    for gaoz_header, movie_id in cross_tab_pairs:
        cross_tab = pd.crosstab(combined_data[gaoz_header], combined_data[movie_id], normalize='index')
        cross_tabs[(gaoz_header, movie_id)] = cross_tab

    # 各GAOZ属性の事前確率を計算
    gaoz_priors = {}
    for gaoz_header in B_USER_ATTRIBUTE_HEADERS:
        gaoz_priors[gaoz_header] = combined_data[gaoz_header].value_counts(normalize=True)

    # Bbの各ユーザーに対して、対数尤度を計算
    results = []
    top8_results = []
    for target_Bb_row_index in range(len(Bb)):
        target_Bb_row = Bb.iloc[target_Bb_row_index]
        gaoz_posteriors = {}
        for target_gaoz in B_USER_ATTRIBUTE_HEADERS:
            log_posterior = np.log(gaoz_priors[target_gaoz].copy())
            for movie_id in MOVIE_IDS:
                bb_review_value = target_Bb_row[movie_id]
                if bb_review_value == "*":
                    continue
                observed_rating = int(bb_review_value)
                cross_tab = cross_tabs[(target_gaoz, movie_id)]
                for gaoz_category in cross_tab.index:
                    P_rating_given_gaoz = cross_tab.loc[gaoz_category].get(observed_rating, 0)
                    if P_rating_given_gaoz == 0:
                        P_rating_given_gaoz = 1e-6  # 小さな値を代入
                    log_posterior[gaoz_category] += np.log(P_rating_given_gaoz)
            # 対数尤度を指数関数で戻し、正規化
            log_posterior -= log_posterior.max()
            posterior = np.exp(log_posterior)
            posterior /= posterior.sum()
            gaoz_posteriors[target_gaoz] = posterior
        # Baのユーザーと比較して、確率が最大のものを選択
        Ba_probabilities = []
        for idx, ba_row in Ba.iterrows():
            probability = 1.0
            for target_gaoz in B_USER_ATTRIBUTE_HEADERS:
                ba_gaoz_value = ba_row[target_gaoz]
                probability *= gaoz_posteriors[target_gaoz].get(ba_gaoz_value, 0)
            Ba_probabilities.append((idx, probability))
        Ba_probabilities.sort(key=lambda x: x[1], reverse=True)
        top_idx, top_prob = Ba_probabilities[0]
        top_ba_row = Ba.iloc[top_idx]
        results.append({
            "Bb_Index": target_Bb_row_index,
            "Ba_Index": top_idx,
            "Probability": top_prob,
            "Ba_Gender": top_ba_row["Gender"],
            "Ba_Age": top_ba_row["Age"],
            "Ba_Occupation": top_ba_row["Occupation"],
            "Ba_ZIP-code": top_ba_row["ZIP-code"]
        })

        # 上位8件の結果を保存
        for i in range(8):
            idx, prob = Ba_probabilities[i]
            ba_row = Ba.iloc[idx]
            top8_results.append({
                "Bb_Index": target_Bb_row_index,
                "Ba_Index": idx,
                "Probability": prob,
                "Ba_Gender": ba_row["Gender"],
                "Ba_Age": ba_row["Age"],
                "Ba_Occupation": ba_row["Occupation"],
                "Ba_ZIP-code": ba_row["ZIP-code"]
            })

    # 結果をデータフレームに変換
    results_df = pd.DataFrame(results)
    top8_results_df = pd.DataFrame(top8_results)


    from collections import defaultdict

    import numpy as np
    import pandas as pd

    def improved_diverse_matching(top8_results_df, diversity_penalty=0.5):
        results = []
        used_ba_indices = defaultdict(int)
        
        for bb_index in top8_results_df['Bb_Index'].unique():
            bb_matches = top8_results_df[top8_results_df['Bb_Index'] == bb_index].sort_values('Probability', ascending=False)
            
            selected = []
            for _, match in bb_matches.iterrows():
                ba_index = match['Ba_Index']
                probability = match['Probability']
                
                prob_score = np.log(probability)
                usage_penalty = -used_ba_indices[ba_index] * diversity_penalty
                score = prob_score + usage_penalty
                
                selected.append((ba_index, probability, score, match))
            
            selected.sort(key=lambda x: x[2], reverse=True)
            
            # 最高スコアのマッチングのみを選択
            ba_index, probability, _, match_data = selected[0]
            results.append({
                "Bb_Index": bb_index,
                "Ba_Index": ba_index,
                "Probability": probability,
                "Ba_Gender": match_data['Ba_Gender'],
                "Ba_Age": match_data['Ba_Age'],
                "Ba_Occupation": match_data['Ba_Occupation'],
                "Ba_ZIP-code": match_data['Ba_ZIP-code']
            })
            used_ba_indices[ba_index] += 1
        
        return results

    # top8_results_dfを使用して改善されたダイバーシティマッチングを適用
    diversity_penalty = 1.2 # この値を調整してバランスを変更できます
    improved_results_list = improved_diverse_matching(top8_results_df, diversity_penalty)

    # リスト形式の結果を表示する関数
    def display_results_list(results_list):
        for match in results_list:
            print(f"Bb_Index: {match['Bb_Index']}, Ba_Index: {match['Ba_Index']}, "
                f"Probability: {match['Probability']:.6f}, "
                f"Gender: {match['Ba_Gender']}, Age: {match['Ba_Age']}, "
                f"Occupation: {match['Ba_Occupation']}, ZIP: {match['Ba_ZIP-code']}")
        print("-" * 80)

    # マッチングの多様性を評価する関数
    def evaluate_diversity(results_list):
        unique_ba_indices = len(set(match['Ba_Index'] for match in results_list))
        total_matches = len(results_list)
        diversity_ratio = unique_ba_indices / total_matches
        print(f"\nDiversity Evaluation:")
        print(f"Total matches: {total_matches}")
        print(f"Unique Ba_Index matches: {unique_ba_indices}")
        print(f"Diversity ratio: {diversity_ratio:.2f}")

    # 結果を表示
    print("Top 1 Matches for each Bb_Index:")
    display_results_list(improved_results_list)

    # 多様性を評価
    evaluate_diversity(improved_results_list)

    # リスト形式の結果をDataFrameに変換
    improved_results_df = pd.DataFrame(improved_results_list)

    # 結果を表示（最初の10行）
    print("\nResults DataFrame (first 10 rows):")
    print(improved_results_df.head(10))

    # Ba_Indexのリストを作成
    my_answer = improved_results_df['Ba_Index'].tolist()

    print(my_answer)

    all_answer_from_mae_attack.append(my_answer)


    import itertools

    cross_tab_pairs = []

    pairs_set = set()
    for b_review_headers in B_REVIEW_HEADERS_LIST:
        for pair in itertools.combinations(b_review_headers, 2):
            pairs_set.add(pair)

    for pair in itertools.combinations(MOVIE_IDS, 2):
        if pair in pairs_set:
            cross_tab_pairs.append(pair)

    print(f"ペアの数: {len(cross_tab_pairs)}")
    cross_tab_pairs[:5]

    # 1. c0からc9までのデータを結合
    combined_data = pd.concat(c_data_list, ignore_index=True)
    combined_data.astype("category")
    for col in MOVIE_IDS:
        combined_data[col] = pd.Categorical(combined_data[col], categories=[0, 1, 2, 3, 4, 5], ordered=True)


    # 2. cross_tab_pairsごとにクロス集計を行う
    cross_tabs = {}
    for movie_id_i, movie_id_j in cross_tab_pairs:
        cross_tab = pd.crosstab(combined_data[movie_id_i], combined_data[movie_id_j], normalize='all')
        cross_tabs[(movie_id_i, movie_id_j)] = cross_tab


    MAX_0_COUNT = 22

    answer_df = pd.DataFrame(index=range(50), columns=["Answer", "0_Prob", "NoCondition0", "MaxOther0"])

    # 0を出力する確率
    prob_0 = []
    # 0以外の中で最も確率が高いもの
    max_other_0 = []

    for target_Bb_row_index in range(50):
        target_Bb_row = Bb.iloc[target_Bb_row_index]

        hidden_movie_id = None
        for movie_id in MOVIE_IDS:
            if target_Bb_row[movie_id] == "*":
                hidden_movie_id = movie_id
                break

        # TODO:　ここの実装が複雑なので後ほど問題ないか確認する
        # 各映画IDに対する target_cross を格納するリスト
        cross_tabs_with_hidden_movie = []

        for movie_id_i, movie_id_j in cross_tab_pairs:
            if hidden_movie_id not in (movie_id_i, movie_id_j):
                continue
            cross_tab = cross_tabs[(movie_id_i, movie_id_j)]
            paired_movie_id = None
            if hidden_movie_id == movie_id_i:
                paired_movie_id = movie_id_j
                # 転置する
                cross_tab = cross_tab.T
            else:
                paired_movie_id = movie_id_i
            # 2143       0       1       2       3       4       5
            # 2                                                   
            # 0     0.1678  0.0290  0.0415  0.0272  0.0413  0.0268
            # 1     0.0406  0.0176  0.0188  0.0116  0.0204  0.0140
            # 2     0.0215  0.0102  0.0126  0.0108  0.0154  0.0095
            # 3     0.0546  0.0173  0.0233  0.0163  0.0236  0.0183
            # 4     0.0477  0.0163  0.0201  0.0161  0.0201  0.0164
            # 5     0.0696  0.0184  0.0243  0.0195  0.0240  0.0175
            review_value = target_Bb_row[paired_movie_id]

            column_sums = cross_tab.sum()
            # 正規化係数を計算（目標値 1/6 を各列の合計で割る）
            normalization_factors = (1/6) / column_sums
            # データフレームの各値に正規化係数を適用
            normalized_cross_tab = cross_tab * normalization_factors
            # print(cross_tab)
            # print(review_value)
            target_cross_tab = normalized_cross_tab.loc[int(review_value)]
            # print(target_cross_tab)
            cross_tabs_with_hidden_movie.append(target_cross_tab)

        cross_tabs_with_hidden_movie = pd.concat(cross_tabs_with_hidden_movie, axis=1)

        # 各カラムごとに総和が1になるように確率を正規化
        def normalize_columns(df):
            return df.div(df.sum(axis=0), axis=1)

        # クロス集計表を列ごとに正規化
        normalized_cross_tabs_with_hidden_movie = normalize_columns(cross_tabs_with_hidden_movie)
        normalized_cross_tabs_with_hidden_movie

        hidden_movie_probabilities = normalized_cross_tabs_with_hidden_movie.mean(axis=1)

        prob_0.append(hidden_movie_probabilities[0])

        max_other_index = hidden_movie_probabilities.iloc[1:].idxmax()
        max_other_0.append(max_other_index)

    answer_df["0_Prob"] = prob_0
    answer_df["MaxOther0"] = max_other_0
    # 0_Probの上位MAX_0_COUNTを求める
    top_0_probs = answer_df.nlargest(MAX_0_COUNT, "0_Prob")
    # NoCondition0を設定
    answer_df["NoCondition0"] = False
    answer_df.loc[top_0_probs.index, "NoCondition0"] = True

    # Answerを決定
    answer_df["Answer"] = np.where(answer_df["NoCondition0"], 0, answer_df["MaxOther0"])


    result_list = answer_df["Answer"].values
    
    all_answer_from_mae_attack.append(result_list)
    print(result_list)

def transpose_2d_list(list_2d):
    return list(map(list, zip(*list_2d)))
all_answer_from_mae_attack = transpose_2d_list(all_answer_from_mae_attack)
all_answer_df = pd.DataFrame(all_answer_from_mae_attack)
all_answer_df.to_csv("./E16.csv", index=False, header=False)
all_answer_df.to_csv("./E16.csv", index=False, header=False)
