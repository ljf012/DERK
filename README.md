# DERK
This code is for the paper "Diversity-Enhanced Recommendation with Knowledge-Aware Devoted and Diverse Interest Learning".
All experiments are run on a machine containing 256GB of RAM, and 4 NVIDIA 3090 or 4 NVIDIA A40 graphics cards in Ubuntu 20.04.

Due to the limitation of upload file size, this zip only contains the preprocessed dataset and training weights of MovieLens. 

## Requirements
- Python 3.8
- CUDA 11.4
- pytorch 1.11
- numpy 1.20.3
- pandas 1.1.4
- torch-scatter 2.0.9
- sklearn 0.24.2
- networkx 2.6.3

## Get Started
1. Install all the requirements.

2. Train and evaluate the DERK using the Python script [main.py](main.py).

    To demonstrate the reproducibility of the best performance reported in our paper and faciliate researchers to track whether the model status is consistent with ours, we provide the best parameter settings (see [utils/parser.py](utils/parser.py)) and the [log](training_log/) for our trainings.

    For efficiency, we use multiple GPUs for training via [DistributedDataParallel](https://pytorch.org/docs/1.11/notes/ddp.html?highlight=distributeddataparallel). To train our method with multiple GPUs on MovieLens-1M dataset, you can run
    ```bash
    python main.py --dataset ml-1m --devices 0,1,2,3
    ```
    If you want to specify only one GPU for training, you can run
    ```bash
    python main.py --dataset ml-1m --devices 0
    ```
    
    To fast reproduce the results on MovieLens-1M in our paper, you can run
    ```bash
    python main.py --dataset ml-1m --test
    ```


## Datasets
We provide three processed datasets: MovieLens-1M, Alibaba-iFashion, and Amazon-book.

- You can find the full version of recommendation datasets via [MovieLens](https://files.grouplens.org/datasets/movielens/), [Alibaba-iFashion](https://github.com/wenyuer/POG), and [Amazon](http://jmcauley.ucsd.edu/data/amazon/).
- We use the preprocessed Alibaba-iFashion and Amazon-Book datasets released by [KGIN](https://github.com/huangtinglin/Knowledge_Graph_based_Intent_Network).
- On MovieLens-1M dataset, We follow [KB4Rec](https://github.com/RUCDM/KB4Rec) to map items into Freebase entities via title matching if there is a mapping available. Then we can obtain `relation_list.txt`, `entity_list.txt` and `kg_final.txt` with remap ID (see [preposess.py](data/ml-1m/preposess.py)).
- We counted the categories of items in each dataset and assigned IDs to them (see example of MovieLens-1M [category_map.ipynb](data/ml-1m/category_map.ipynb)), and then corresponded the items to the categories to get `item_cate.txt` (see example of MovieLens-1M [item_cate.ipynb](data/ml-1m/item_cate.ipynb)).


|                       |               | MovieLens-1M | Alibaba-iFashion | Amazon-book |
| :-------------------: | :------------ | ----------: | --------: | ---------------: |
| User-Item Interaction | #Users        |      6,040 |    114,737 |          70,679 |
|                       | #Items        |      3,706 |    30,040 |           24,915 |
|                       | #Interactions |     734,564 | 1,380,510 |        652,514 |
|                       | #Categories |     18 | 75 |        598 |
|    Knowledge Graph    | #Entities     |      42,097 |    89,196 |           113,487 |
|                       | #Relations    |          52 |         51 |               39 |
|                       | #Triplets     |   603,188 |   279,155 |          2,557,746 |

- `train.txt`
  - Train file.
  - Each line is a user with her/his positive interactions with items: (`userID` and `a list of itemID`).
- `test.txt`
  - Test file (positive instances).
  - Each line is a user with her/his positive interactions with items: (`userID` and `a list of itemID`).
  - Note that here we treat all unobserved interactions as the negative instances when reporting performance.
- `user_list.txt`
  - User file.
  - Each line is a triplet (`org_id`, `remap_id`) for one user, where `org_id` and `remap_id` represent the ID of such user in the original and our datasets, respectively.
- `item_list.txt`
  - Item file.
  - Each line is a triplet (`org_id`, `remap_id`, `freebase_id`) for one item, where `org_id`, `remap_id`, and `freebase_id` represent the ID of such item in the original, our datasets, and freebase, respectively.
- `entity_list.txt`
  - Entity file.
  - Each line is a triplet (`freebase_id`, `remap_id`) for one entity in knowledge graph, where `freebase_id` and `remap_id` represent the ID of such entity in freebase and our datasets, respectively.
- `relation_list.txt`
  - Relation file.
  - Each line is a triplet (`freebase_id`, `remap_id`) for one relation in knowledge graph, where `freebase_id` and `remap_id` represent the ID of such relation in freebase and our datasets, respectively.
- `kg_final.txt`
  - KG triplets file.
  - Each line is a triplet (`Entity remap_id`, `Relation remap_id`, `Entity remap_id`) for one head entity from knowledge graph.
- `category_list`
  - Categoty file.
  - Each line is a triplet (`category`, `cate_count`,	`cate_id`) for one category in dataset, where `category` , `cate_count` and	`cate_id` is the original category, the number of this category in dataset and the ID of this category, respectively.
- `item_cate.txt`
  - Item's category file
  - Each line is a triplet (`item_id`,	`cate_id`) for one item, where `item_id` and `cate_id` represent the ID of such item in our datasets and category, respectively.