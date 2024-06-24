# Adaptive-Multi-Resolution-Feature-Fusion
Code release for “[Adaptive Multi-Resolution Feature Fusion for Fine-Grained Visual Classification]"



## Changelog
- 2024/06/21 upload the code.


## Requirements

- python 3.8
- PyTorch 1.13.1+cu117
- torchvision  0.14.1+cu117
- learn2learn 0.2.0

## Data
- Download datasets
- Extract them to `data/cars/`, `data/birds/` and `data/airs/`, respectively.
- Split the dataset into train and test folder, the index of each class should follow the Birds.xls, Air.xls, and Cars.xls

* e.g., CUB-200-2011 dataset
```
  -/birds/train
	         └─── 001.Black_footed_Albatross
	                   └─── Black_Footed_Albatross_0001_796111.jpg
	                   └─── ...
	         └─── 002.Laysan_Albatross
	         └─── 003.Sooty_Albatross
	         └─── ...
   -/birds/test	
             └─── ...         
```



## Training
- `CUDA_VISIBLE_DEVICES=X python main.py`



## Contact
Thanks for your attention!
If you have any suggestion or question, you can leave a message here or contact us directly:
- yangyuqi@bupt.edu.cn
- mazhanyu@bupt.edu.cn
