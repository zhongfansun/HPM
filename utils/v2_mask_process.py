#删除除第一个外的其他[mask]
import os
import re
import json
with open('/home/szf/EJSCM/temp/okvqa/cache/okvqa_trunc_v2_train_mavexdecla.json', 'r') as fd:
    declas = json.load(fd)

for i in declas:
    if len(re.findall("\[MASK\]", declas[i])) > 1:
        declas[i] = declas[i].replace('[MASK]', '[MASK1]', 1).replace(' [MASK]', '').replace('[MASK1]', '[MASK]')
        # declas[i] = declas[i].split('<-->')[0] + '<-->' + declas[i].split('<-->')[1]
        # print(declas[i])

with open('/home/szf/EJSCM/temp/okvqa/cache/okvqa_final_v2_train_mavexdecla.json','w') as k:
    json.dump(declas, k)


# #[mask]在超过20个tokens的位置，在句首添加[mask]
# import os
# import json
# with open('/home/szf/EJSCM/temp/okvqa/cache/okvqa_plus_v2_train_mavexdecla.json', 'r') as fd:
#     declas = json.load(fd)
#
# for i in declas:
#     if '[MASK]' not in ' '.join(declas[i].split(' ')[:20]):
#         declas[i] ='[MASK]' + ' ' + declas[i]
#         print(declas[i])
#
# with open('/home/szf/EJSCM/temp/okvqa/cache/okvqa_trunc_v2_train_mavexdecla.json','w') as k:
#     json.dump(declas, k)


#加[mask],对于没有mask的句子在末尾加[mask]
# import os
# import json
# with open('/home/szf/EJSCM/temp/okvqa/cache/train2014_declarative.json', 'r') as fd:
#     declas = json.load(fd)
#
# for i in declas:
#     if '[MASK]' not in declas[i]:
#         declas[i] = declas[i] + ' ' + '[MASK]'
#         print(declas[i])
#
# with open('/home/szf/EJSCM/temp/okvqa/cache/okvqa_plus_v2_train_mavexdecla.json','w') as k:
#     json.dump(declas, k)