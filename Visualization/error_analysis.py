import re
import pandas as pd
from collections import Counter

FIND_MOST_FREQUENT_EXAMPLES = True

if FIND_MOST_FREQUENT_EXAMPLES:
    # Load Txt File: Sample ID
    file_path = "../OutrageousExamples_FineTune.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    sample_ids = []
    for line in lines:
        match = re.search(r"Sample ID: (\d+)", line)
        if match:
            sample_ids.append(match.group(1))

    top_100 = Counter(sample_ids).most_common(100)

    df_top_100 = pd.DataFrame(top_100, columns=["Sample ID", "Count"])
    df_top_100.head()
    print(df_top_100)

# Print the text
df_imdb_te = pd.read_csv("../raw_data/splits/imdb_test.csv")
df_douban_te = pd.read_csv("../raw_data/splits/douban_test.csv")
df_test = pd.concat([df_imdb_te, df_douban_te], ignore_index=True)

# top sample_ids
sample_ids = [88, 2918, 18964, 18967, 19004]

for sid in sample_ids:
    row = df_test.iloc[sid]
    print(f"Sample ID: {sid}")
    print(f"Label: {row['label']}")
    print(f"Review: {row['review']}")
    print("-" * 60)

# Output for "OutrageousExamples_Basic.txt"
# """
#      Sample ID  Count
# 0        635      5
# 1       1281      5
# 2       1351      5
# 3       1604      5
# 4       1700      5
# 5       3574      5
# 6       3735      5
# 7       4759      5
# 8       7178      5
# 9       7976      5
# 10      9306      5
# """

# Output for "OutrageousExamples_FineTune.txt"
# """
#       Sample ID  Count
# 0         39      3
# 1         63      3
# 2         88      3
# 3        144      3
# 4        152      3
# ..       ...    ...
# 95      2893      3
# 96      2913      3
# 97      2918      3
# 98      2923      3
# 99      2924      3
# ..        ...    ...
# 495     14090      3
# 496     14199      3
# 497     14247      3
# 498     14328      3
# 499     14338      3
# ..        ...    ...
# 695     18839      3
# 696     18885      3
# 697     18964      3
# 698     18967      3
# 699     19004      3
# ...
# """