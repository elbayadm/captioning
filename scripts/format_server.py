import json


Res = json.load(open('ensemble_val2014.json', 'r'))
for item in Res:
    item['image_id'] = int(item['image_path'].split('.')[0].split('_')[-1])
    print('parsed id:', item['image_id'])
    del item['image_path']
json.dump(Res, open('ensemble_val2014_formatted.json', 'w'))
