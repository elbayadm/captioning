import json

Res = json.load(open('bw3.json', 'r'))
for item in Res:
    item['image_id'] = item['image_path']
    del item['image_path']
json.dump(Res, open('bw3_formatted.json', 'w'))
