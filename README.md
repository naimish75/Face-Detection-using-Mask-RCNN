Instance segmenting more than 2 classes in an image,where image dataset is from publicly available data, for annotation we are using VGG annotator latest version

  Here we defined 1 classes :<ul>
  <li>sunglasses</li>
</ul>
  
  


 <h2># Make sure you make a new folder with name as "main" and move "dataset, logs, Mask_RCNN, mask_rcnn_coco.h5" in the folder to avoid path changes</h2>
<h3># Train a new model starting from pre-trained COCO weights</h3>
        python final.py train --dataset=/path/to/datasetfolder --weights=coco

<h3># Resume training a model that you had trained earlier</h3>
        python final.py train --dataset=/path/to/datasetfolder --weights=last

