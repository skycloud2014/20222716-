# 20222716-
# **개요**

**함수의 목적** 
 파이썬에서 지원하는 패키지인 skitlearn 을 활용한  Brain Tumor Classification 모델 구성 

**활동의 목적** 
그중에서도 최적의 모델과 이에 맞는 HyperParameter을 찾기 위한 활동 

 

# 모델

사용모델은 RandomForest 모델과 KNN ( K - 최근접 이웃) 모델 두개를 골라 각각 정확도를 시험해 본 뒤 더 높게 결과가 나온 KNN으로 선택하였다 

**KNN**
+ 지도 학습 알고리즘중, 분류쪽으로 사용 되는 알고리즘 
+ 데이터가 주어질때 유클리드 거리를 통해 인접한 데이터의 개수를 비교하여 데이터를 분류하는 방식  
```python
sklearn.neighbors.KNeighborsClassifier
```
을 통해 사용
## HyperParameter
KNN은 다음과 같은 HyperParameter가 존재 
+ **n_neighbors**
+ **weights**
+ **algorithm** 
--> (‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’)
+ **leaf_size**
+ **p**
+ **metric**
+ **metric_params**

**알고리즘** 
--> 의 경우 k차원의 점으로 트리를 저장하는 KD트리를 
범위의 기준으로 차원을 내림차순로 정럴하고 Subtree에 속한 범위의 절반을 기준으로 KD트리를 적용시키는 Ball 트리를 선택하였다 

**metric** 
종류로는 minkowski, euclidean, mahalanobis 등이 있으며 일일히 적용시켜보았을때 그중 수치가 잘 나온 manhattan을 사용했다 

n_neighbors 와 leaf size의 경우 int형식으로 입력을 받기에 
어느 값에서 가장 정확도가 높은지 알아보기 위해 다음과 같은 코드를 구성했다 
```python
score_list = []
for each in range(1,60):
    knn2 =sklearn.neighbors.KNeighborsClassifier(n_neighbors=1,algorithm='ball_tree',metric='manhattan',leaf_size=1)
    knn2.fit(X_train, y_train)
    score_list.append(knn2.score(X_test, y_test))

plt.plot(range(1,60), score_list)
plt.xlabel("leaf size")
plt.ylabel("accuracy")
plt.title("몫에 따른 정확도")
plt.show()
```
```python
score_list = []
for each in range(1,30):
    knn2 =sklearn.neighbors.KNeighborsClassifier(n_neighbors=each,algorithm='ball_tree',metric='manhattan')
    knn2.fit(X_train, y_train)
    score_list.append(knn2.score(X_test, y_test))

plt.plot(range(1,30), score_list)
plt.xlabel("n_neighbors")
plt.ylabel("accuracy")
plt.title("몫에 따른 정확도")
plt.show()
```
결과는 코드안에 있으며, leaf size는 거의 영향을 안끼치기에 default로 두고 
n_neighbor의 경우 0에 가까이 갈수록 정확도가 높아졌으므로 1로 설정하였다 
