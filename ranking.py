import japanize_matplotlib
import matplotlib.pyplot as plt

# チームごとのUtility Score
utility_scores = [
    91.02, 83.69, 76.76, 69.1, 72.36, 74.55, 72.4, 63.3, 72.2, 86.1,
    76.89, 80.17, 50.63, 70.27, 82.02, 67, 29.38, 69.56, 72.08, 72.01, 70.38
]

# 匿名スコア
anonymity_scores = [68,51,58,34,43,41,39,31,28,43,43,51,16,88,0,34,11,83,34,33,48]
anonymity_scores = [100 - score for score in anonymity_scores]

# チーム名
team_names = [
    "宮地研.exe", "私達日本語本当下手", "ポップコーン", "Hots", "SHA-NES", "神ぼ大νττ", "たけのこ映画守り隊", 
    "0xA", "ステテコ泥棒", "動的計画法", "Gunmataro117", "HAL", "privocy", "ES5", "佐古研究室", 
    "こそっとアタック、しれっとブロック", "匿名アノニマス", "RITCHEY", "KAT-TUNE", "PR.AVATECT", "春日部防衛隊（かすかべ防衛隊）"
]

# 散布図の作成
plt.figure(figsize=(12, 8))
plt.scatter(utility_scores, anonymity_scores, color='b', marker='o')

# チーム名を各ポイントに表示
for i, team in enumerate(team_names):
    plt.text(utility_scores[i], anonymity_scores[i], team, fontsize=9, ha='right', va='bottom')

# ラベルの設定
plt.title("Utility Score vs Anonymity Score")
plt.xlabel("Utility Score")
plt.ylabel("Anonymity Score")

# グラフの表示
plt.grid(True)
plt.show()
