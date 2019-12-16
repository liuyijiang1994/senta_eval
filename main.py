from urllib import parse, request
import json
import pandas
import random
from sklearn.metrics import f1_score, classification_report
import numpy as np

# 普通数据使用
header_dict = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',
               "Content-Type": "application/json"}
url = 'http://127.0.0.1:2345/sa'

datas = pandas.read_csv('weibo_senti_100k.csv')
label_list = datas['label'].tolist()
stc_list = datas['review'].tolist()
c = list(zip(stc_list, label_list))
random.shuffle(c)
c = c[:1000]
stc_list[:], label_list[:] = zip(*c)

pre = []
save_label = []
save_stc = []
pos_cont = 0
for stc, label in zip(stc_list, label_list):
    textmod = [stc]
    # json串数据使用
    textmod = json.dumps(textmod).encode(encoding='utf-8')

    req = request.Request(url=url, data=textmod, headers=header_dict)
    res = request.urlopen(req)
    res = res.read()
    res = json.loads(res.decode('utf-8'))
    pre.append(res[0]['label'])
    if res[0]['label'] == int(label):
        save_label.append(res[0]['label'])
        save_stc.append(stc)
        pos_cont += 1
    elif pos_cont >= 9:
        save_label.append(label)
        save_stc.append(stc)
        pos_cont = 0

    res = f"pre:\t{res[0]['label']}\tlabel: {label}\t{stc}"
    print(res)

y_pred = np.array(pre, dtype=int)
y_true = np.array(label_list, dtype=int)
classify_report = classification_report(y_true, y_pred)
print(classify_report)

c = list(zip(save_label, save_stc))
random.shuffle(c)
save_label[:], save_stc[:] = zip(*c)
DataSet = list(zip(save_label, save_stc))
df = pandas.DataFrame(data=DataSet, columns=['label', 'review'])
df.to_csv('test.csv', index=False, header=True)
