from urllib import parse, request
import json
import pandas
import random
from sklearn.metrics import f1_score, classification_report
import numpy as np
import datetime

# long running


# 普通数据使用
header_dict = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',
               "Content-Type": "application/json"}
url = 'http://127.0.0.1:2345/sa'

datas = pandas.read_csv('test.csv')
label_list = datas['label'].tolist()
stc_list = datas['review'].tolist()
c = list(zip(stc_list, label_list))
c = c[:len(c) // 3]
random.shuffle(c)
stc_list[:], label_list[:] = zip(*c)

pre = []
tar = []
starttime = datetime.datetime.now()

batch_num = len(stc_list) // 32
for i in range(batch_num):
    to_text = stc_list[i:i + 32]
    to_label = label_list[i:i + 32]
    textmod = json.dumps(to_text).encode(encoding='utf-8')

    req = request.Request(url=url, data=textmod, headers=header_dict)
    res = request.urlopen(req)
    res = res.read()
    res = json.loads(res.decode('utf-8'))
    for i in range(32):
        print_line = f"pre:\t{res[i]['label']}\tlabel: {to_label[i]}\t{to_text[i]}"
        pre.append(res[i]['label'])
        tar.append(to_label[i])
pre = pre * 3
tar = tar * 3
endtime = datetime.datetime.now()
print('用时：', (endtime - starttime).seconds)
print('总数据量', len(pre))
y_pred = np.array(pre, dtype=int)
y_true = np.array(tar, dtype=int)
classify_report = classification_report(y_true, y_pred)
print(classify_report)
