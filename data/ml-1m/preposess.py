import pandas as pd
from tqdm import tqdm

'''Build relation_list with remap ID'''
df = pd.read_csv('./ml-1m/graph_movie2.txt', sep='\t', names=['item', 'relation', 'entity', '.'], error_bad_lines=False)
# graph_movie2.txt mapping items into Freebase entities is built from KB4Rec (https://github.com/RUCDM/KB4Rec).

# Cleaning data with a number less than 10
d = pd.DataFrame(df.relation.value_counts())
d.columns = ['nums']
d = d[d['nums'] < 10]
delindexs = d.index
df = df[~df['relation'].isin(delindexs)]

df.relation = df.relation.str.replace('<','')
df.relation = df.relation.str.replace('>','')
df.item = df.item.str.replace('<http://rdf.freebase.com/ns/','')
df.item = df.item.str.replace('>','')
df.entity = df.entity.str.replace('<http://rdf.freebase.com/ns/','')
df.entity = df.entity.str.replace('>','')
df = df.drop('.', axis=1)

d = pd.DataFrame(df.entity.value_counts())
d.columns = ['nums']
d = d[d['nums'] < 5]
delindexs = d.index
df = df[~df['entity'].isin(delindexs)]

df.to_csv('./ml-1m/graph_movie.txt', sep=' ', index=False, header=False)

relation_a = df['relation'].value_counts()
relation = list(relation_a.index)

num = list(relation_a.values)

id = list(range(len(relation)))
remap = {
    "org_id" : relation,
    "remap_id" : id
}
re_remap = pd.DataFrame(remap)
re_remap.to_csv('./relation_list.txt', sep=' ', index=False)


'''Build entity_list with remap ID'''
relation = pd.read_csv('./relation_list.txt', sep=' ')
kg = pd.read_csv('./ml-1m/graph_movie.txt', sep=' ', names=['item', 'relation', 'entity'])
item_list = pd.read_csv('./item_list.txt', sep=' ')

kg_item = list(kg.item.value_counts().index)
kg_entity = list(kg.entity.value_counts().index)

entity = list(set(kg_item + kg_entity))

item = list(item_list.freebase_id)
for i in range(len(entity)):
    if not entity[i] in item:
        item.append(entity[i])

id = list(range(len(item)))
entity_remap = {
    "org_id" : item,
    "remap_id" : id
}
re_remap = pd.DataFrame(entity_remap)

re_remap.to_csv('./entity_list.txt', sep=' ', index=False)


'''Build KG triplet with remap ID'''
relation = pd.read_csv('./relation_list.txt', sep=' ')
kg = pd.read_csv('./ml-1m/graph_movie.txt', sep=' ', names=['item', 'relation', 'entity'])
entity = pd.read_csv('./entity_list.txt', sep=' ')


filename = 'kg_final.txt'
file = open('./' + filename, 'w+')
for i in tqdm(range(len(kg))):
    s = str(entity[entity.org_id == kg.item[i]].remap_id.values[0]) + ' '               # entity remap_id
    s = s + str(relation[relation.org_id == kg.relation[i]].remap_id.values[0]) + ' '   # relation remap_id
    s = s + str(entity[entity.org_id == kg.entity[i]].remap_id.values[0]) + '\n'        # entity remap_id
    file.write(s)
file.close()