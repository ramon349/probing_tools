
# Setup 
- Create an environment  that has python 3.10.13  using conda or whichever means you prefer 
- install the packages in the requirements file or use the conda yml file 
- Install this code as an editable pacakge by doing 
```bash 
    python3 -m pip install -e .
``` 

# Running Extraction of feature embeddings  
- you run extraction by doing 
```bash 
python3 -m a3i_general_probe.extract_embeds --config_path ./Path2Config.json
```
- The parameters expected by the config file are
- Extraction expects a few things  as seen in the csv_path
- csv_path: should be absolute Path to a csv file that contains a column where you specify the path to png/jpeg/dicom files we can load for extraction 
- num_workers: number of workers used by our dataloaders during extraction 
- device: GPU id to use. Right now we only support using one gpu 
- model: what model will we use for feature extraction. See models folder for available models
- extractor:  Class that specifies the extraction logic only one available 
- batchsize: dataloader parameter 
- img_func: takes "dcm" or "png" . Use "png" if loading jpegs or anything supported by pillow 
- output_dir: this is where the embeddings are stored 
- mapping_path: we make a mapping file that says  feat_path --> original_img_path 
- test_transforms: these are the transforms needed to massage the data into the format expected by the model. VALIDATE FOR YOUR OWN USE 
- col_info: parameters needed by the dataloader. Right now only  specify the column  in your csv that specifies the path to your images 
- transform_conf: parameters to be used by the transforms you've specified 
- model_parameters: anything the model needs during initialization. right now it's only the path to the pre-trained weights 
```json 
my_config = {
"csv_path":"PathToCSVFile",
"num_workers":16,
"device":["cuda:3"],
"extractor":"MammoClip",
"model":"mammo_clip_vision",
"batch_size":16,
"dataset":"MammoFeatExtract",
"image_func":"dcm",
"output_dir":"",
"mapping_path":"",
"test_transforms":["tensor","make8bit","crop","resize","makeZeroOne","grey2rgb","norm"],
"col_info":{
	"img_path":"file"
},
"transform_conf":{
	"img_shape":[1520,912],
	"norm_mu":0.2197,
	"norm_std":0.2029
},
"model_parameters":{
	"w_path":"PathToMammoclipweights"

}
}
``` 





