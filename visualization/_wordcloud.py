from nltk.corpus import stopwords 
from wordcloud import WordCloud, STOPWORDS 
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize 
import matplotlib.pyplot as plt

df = pd.read_csv("codeforces_question_v3.csv")
print(df.columns)

# tags = df["solution"]

# tags_list = []
# print(tags)
# for row in tags:
#     tokens = word_tokenize(row)
#     for t in tokens:
#         tags_list.append(t)

# print(tags_list)


# unique_string=(" ").join(tags_list)
# wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)
# plt.figure(figsize=(15,8))
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.savefig("tags"+".png", bbox_inches='tight')
# plt.show()
# plt.close()





