import matplotlib.pyplot as plt
from wordcloud import WordCloud
import jieba
text='你要的全部语句'
#结巴分词
wordlist = jieba.cut(text,cut_all=True)
wordlist=[x for x in wordlist if len(x)>=2]
wl = " ".join(wordlist)
wc = WordCloud(background_color = "black", #设置背景颜色
               #mask = "图片",  #设置背景图片
               max_words = 2000, #设置最大显示的字数
               #stopwords = "", #设置停用词
               font_path='C:/Windows/Fonts/simkai.ttf',
        #设置中文字体，使得词云可以显示（词云默认字体是“DroidSansMono.ttf字体库”，不支持中文）
               max_font_size = 50,  #设置字体最大值
               random_state = 30, #设置有多少种随机生成状态，即有多少种配色方案
                width=600,  # 指定宽度
        height=600
    )
myword = wc.generate(wl)#生成词云
 
#展示词云图
plt.subplots(figsize=(20,20),dpi=50)
plt.imshow(myword)
plt.axis("off")
plt.savefig('词云.png')