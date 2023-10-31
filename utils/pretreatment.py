
# 预处理 数据集
# 剔除标点符号,\xa0 空格
def pretreatment(comments):
    result_comments = []
    punctuation = '。，？！：%&~（）、；“”&|,.?!:%&~();""'
    for comment in comments:
        comment = ''.join([c for c in comment if c not in punctuation])
        comment = ''.join(comment.split())  # \xa0
        result_comments.append(comment)

    return result_comments
