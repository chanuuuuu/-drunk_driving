
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.cluster.vq import vq, kmeans, kmeans2, whiten
import pandas as pd
# 위도, 경도로 구성된 dataframe과 method를 입력받는다.
# single    - 군집간의 거리를 최소거리를 사용 
# complete  - 군집간의 거리를 최대거리를 사용 -> 군집들의 응집을 중요시
# average   - 군집간의 거리를 원소들의 평균거리를 사용 -> 군집의 규모, 이상치에 영향이 큼
# centroid  - 군집간의 거리를 원소들의 중앙값을 사용 -> 정보의 손실
# ward      - 군집들의 mse를 최소화함 -> 자주 사용함.
def heir_cluster(df, method = 'ward', n_cluster = 10):
    mycood = df.loc[:,['위도','경도']] if '위도' in df.columns else df.loc[:,['x','y']]
    cluster = linkage(mycood, method=method)
    cutree  = cut_tree(cluster, n_clusters = n_cluster)
    label = cutree.reshape(-1)
    mydf = pd.concat([df, pd.Series(label, name= 'label')], axis = 1)
    center = mydf.groupby('label').mean()
    return label, center


# scipy의 vector quantization(벡터 양자화)를 사용할 것.
# 벡터양자화 - 벡터를 clustering 하여 code - codebook으로 나눈다는 의미. -> image compress에 사용됨.
# whiten  - 각 feature의 basis(기저)에 따라 정규화하여줌 (각 feature들 간의 unrelated, one variance가 보장) **반드시 선수행
# vq      - codebook을 통해 새로운 vector를 cluster로 할당시켜준다. -> label, distortion으로 출력 
# kmeans  - k-means 알고리즘을 통해 codebook을 구성, clustering의 개념보다는 군집의 centroid를 distrortion을 최소화하는 방향
# kmeans2 - k-means 알고리즘을 통해 codebook과 label을 출력
def kmeans_cluster(df, k = 5, minit = 'random'):
    mycood = df.loc[:,['위도','경도']] if '위도' in df.columns else df.loc[:,['x','y']]
    #mycood = whiten(mycood)
    codebook, _ = kmeans(mycood, k)
    label, _ = vq(mycood, codebook)
    mydf = pd.concat([df, pd.Series(label, name= 'label')], axis = 1)
    center = mydf.groupby('label').mean()
    return label, center
