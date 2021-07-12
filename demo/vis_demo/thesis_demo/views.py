import json
import os
import time

from django.core.files.storage import FileSystemStorage
from django.shortcuts import render


from . import vrd_demo
detector = vrd_demo.VRDPredictor("thesis_demo")


def show(request, img_dir):
    print(img_dir)
    start = time.time()
    det_res = detector.inference(img_dir)
    print("cost time: %f" % (time.time() - start))
    context = {
        "det_res":  json.dumps([det_res])
    }
    return render(request, 'res_vis.html', context=context)


def upload(request):
    context = {}
    if request.method == 'POST':

        uploaded_file = request.FILES.get('image')
        if uploaded_file is None:
            return render(request, 'upload.html', context)
        fs = FileSystemStorage()
        if uploaded_file.name.split('.')[-1] not in ["jpg", 'png', 'jpeg']:
            return render(request, 'upload.html', context)
        f_dir = os.path.join('thesis_demo/static/img_cache', uploaded_file.name)
        fs.save(f_dir, uploaded_file)
        return show(request, f_dir)

    return render(request, 'upload.html', context)


