import json


Flickr = json.load(open('data/flickr30k/dataset_flickr30k.json', 'r'))
Out = {'info': 'formatted flickr30k captions o match coco eval code',
       'type': "captions",
       'annotations': [],
       'images': []
       }
Flickr = Flickr['images']
for flick in Flickr:
    if flick['split'] == 'val':
        Out['images'].append({'file_name': flick['filename'], 'id': flick['imgid']})
        for sent in flick['sentences']:
            Out['annotations'].append({'image_id': sent['imgid'],
                                       'id': sent['sentid'],
                                       'caption': sent['raw']})
json.dump(Out, open('data/flickr30k/flickr30k_val_annotations.json', 'w'))

