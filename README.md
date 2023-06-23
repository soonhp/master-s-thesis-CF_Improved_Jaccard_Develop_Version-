# master_thesis-CF_Improved_Jaccard_Develop_Version(Paper_Code)

# 석사학위논문
## 연구주제 : 평점을 고려한 개선된 자카드 유사도 기반 사용자 협업필터링 추천시스템

## 연구동기
#### Rating_Jaccard

- Rating_Jaccard는 두 사용자 간의 평점이 같은 항목의 개수를 두 사용자가 같이 평가한 항목의 비율로 구하는데 이로 인해 한계점들이 발생.

  1) 일반적으로 두 사용자 유사도는 co-rated item에 비례. But, Rating_Jaccard는 반비례

  2) 평점이 같은 경우 만을 고려하기 때문에 유사도가 0이 되는 경우가 다수 발생.

  3) 각 사용자별 평점의 편향을 반영하지 못함.

- 본 연구에서는 이러한 한계점들을 해결하기 위한 새로운 유사도 제안. 더 나아가 제안 방법에서 발생한 한계에 대해서도 공통항목의 수를 고려하는 확장된 방법을 제안하여 해결하고자 한다.  
- Park, S. H., & Kim, K. (2023). Collaborative filtering recommendation system based on improved Jaccard similarity. Journal of Ambient Intelligence and Humanized Computing, 1-18. -> 본 논문을 develop한 연구.

## 제안방법(기존 논문에서 추가된 내용)
#### RJAC_DZ(Rating_Jaccard의 세번째 한계점)
- RJAC_DUB에서 확장하여 사용자별 표준편차를 반영하는 유사도 제안.
  - 각 사용자의 평점과 평균평점의 차이를 표준편차로 나눠주면 Z-SCORE가 됨.
  - Z-SCORE 간의 차이에 절대값을 씌운 값이 𝑻𝑯 이하인 항목의 수를 두 사용자가 평가한 모든 항목의 수로 나눠줌.
  - RJAC_DUB와 달리 사용자별 평점의 편향과 분포를 동시에 반영.
- 본 연구에서는 이것을 RJAC_DZ(Rating_Jaccard allowing a small difference in ratings’ z_score considering user biases)로 명명.
  
![image](https://github.com/soonhp/master_thesis-CF_Improved_Jaccard_Develop_Version/assets/73877159/27463a5f-b866-43dd-a761-18805d39cbbd)

#### RSC(Extended Version)
- 이전의 제안 방법론들은 공통항목의 수를 고려하지 않는 한계가 있음.
- 공통항목 중에 유사한 항목이 어느 정도인지 비율을 반영하는 개선된 유사도 제안
- 제안 유사도별로 각각 가중치를 부여 → RJAC_DUB 예시 :  (|𝑵_𝑻𝑫𝑼𝑩 (𝒂,𝒃)|/|𝑰_𝒂∩𝑰_𝒃 | )^(𝟏/𝟐)
- 가중치 term에 𝟏/𝟐 제곱을 하는 이유 : 사용자 간 공통항목이 매우 많은 경우 가중치가 지나치게 감소할 수 있으므로 보정을 해주기 위함.
- 공통항목의 수와 유사하다고 판단되는 항목의 수를 모두 반영하는 유사도
- 본 연구에서는 이것을 RSC(ratio of similar items among co-rated items) 로 명명.
- RJAC_DUB 예시)
![image](https://github.com/soonhp/master_thesis-CF_Improved_Jaccard_Develop_Version/assets/73877159/fc4bc1dd-294a-4f08-a39a-7b1e3d8f6262)

#### RCT(Extended Version)
- 이전의 제안 방법론들은 공통항목의 수를 고려하지 않는 한계가 있음.
- 전체 아이템 중에 공통항목이 어느 정도인지 비율을 반영하는 개선된 유사도 제안
- 제안 유사도별로 가중치를 부여 → |𝑰_𝒂∩𝑰_𝒃 |/|𝑰|
- 공통항목을 그대로 반영하기 보다는 전체 아이템의 수(|𝑰|)로 normalize.
- 공통항목의 수와 유사하다고 판단되는 항목의 수를 모두 반영하는 유사도
- 본 연구에서는 이것을 RCT(ratio of co-rated items among total items) 로 명명.
- RJAC_DUB 예시)
![image](https://github.com/soonhp/master_thesis-CF_Improved_Jaccard_Develop_Version/assets/73877159/f19fc53c-801f-4265-b65b-afc7985495e5)

## 실험설계 및 결과

#### 데이터
- 본 연구에서는 추천시스템 영역에서 자주 쓰이는 데이터셋인 CiaoDVD, Filmtrust, MovieLens100k, MovieLens1M, Amazon, Netflix의 총 6가지의 공개 데이터셋에 대해 성능을 비교 검증.
- 최소 20개 이상 아이템을 평가한 사용자들을 대상으로 데이터셋을 구성하는 것이 일반적이기 때문에 데이터셋 재구성함.
- 모든 데이터셋의 평점 범위를 1~5점으로 설정하기 위해 범위가 다른 데이터셋의 경우 scaling 진행.
- 데이터셋에 대해 학습데이터 셋 80%, 검증 데이터셋 20%로 stratified five fold cross−validation 를 수행하여 실험을 진행하고 평균 결과를 제시

#### 평가지표
![image](https://github.com/soonhp/master_thesis-CF_Improved_Jaccard_Develop_Version/assets/73877159/7cc11118-4155-4c29-8af9-08abf6b5736a)

#### 실험 결과
1) 제안 유사도 별 최적의 파라미터
   
   ![image](https://github.com/soonhp/master_thesis-CF_Improved_Jaccard_Develop_Version/assets/73877159/0c288efb-2c38-4a8d-8c12-eca907144c43)

2) 제안 유사도 별 사전 실험
   - RJAC_DUB와 RJAC_DZ의 S점수
     ![image](https://github.com/soonhp/master_thesis-CF_Improved_Jaccard_Develop_Version/assets/73877159/7902376c-a95e-4ba7-912f-cd14a98e9434)
     - Rating_Jaccard의 세 번째 한계점을 해결하기 위해 RJAC_DUB, RJAC_DZ 제안.
     - 두 방법론 간의 성능 비교 결과 모든 데이터셋에서 RJAC_DUB의 성능이 좋음.
     - 사용자별 평점의 편향과 분포를 같이 반영하는 것보다는 평점의 편향만을 반영하는 것이 성능향상에 더 큰 요인임. -> 따라서 Rating_Jaccard의 세 번째 한계점을 해결하는 방법론으로 RJAC_DUB 선정
   - 확장된 제안 방법론 : RSC, RCT
     - RSC와 RCT는 사전 실험을 통해 제안 유사도 방법론들 중에서 RJAC_DUB와 결합하는 것이 성능이 가장 좋다는 것을 확인함.
     - RJAC_DUB_RSC와 RJAC_DUB_RCT를 확장된 유사도 방법론으로 제안.
    
3) 제안 방법론 별 성능 비교
   
   ![image](https://github.com/soonhp/master_thesis-CF_Improved_Jaccard_Develop_Version/assets/73877159/3758870d-2c95-4811-bb91-12c5fdfe367b)

- 본 연구에서 제안한 모든 방법론들이 기존 Rating_Jaccard 보다 훨씬 성능이 좋음.
    -> Rating_Jaccard에서 분모에 두 사용자가 평가한 모든 항목의 수로 변경하는 것이 성능이 더 좋다는 것을 실험적으로 증명.
- 대체적으로 RJAC_D 보다 RJAC_DUB가 𝑺 점수에서 좋은 성능을 나타내고 있음.
   -> 단순히 평점의 차이를 반영하기 보다는 그 평점에 평균평점을 빼주어 차이를 반영하는 것이 더 효과적이라는 사실을 알 수 있음.
- 데이터셋 별로 제안 유사도들 중에서 RJAC_DUB_RSC와 RJAC_DUB_RCT가 좋은 성능을 나타내고 있음.
- 보편적으로 성능이 가장 좋은 유사도는 RJAC_DUB_RSC.


4) 타 유사도와 성능 비교
   - MAE
     
   ![image](https://github.com/soonhp/master_thesis-CF_Improved_Jaccard_Develop_Version/assets/73877159/0b1fff98-b2f4-4fd8-b976-60bf74e3a608)

   - F1
     
   ![image](https://github.com/soonhp/master_thesis-CF_Improved_Jaccard_Develop_Version/assets/73877159/2acd9bf8-96ee-40fc-a085-2678a035e245)

   - 타 유사도와 비교 결과
     - RJAC_DUB_RSC는 모든 데이터셋에 대해 가장 낮거나 두 번째로 낮은 MAE 값을 나타냄.
     - F1-score 측면에서 CiaoDVD 데이터셋을 제외한 나머지 데이터셋에서 다른 유사도에 비해 RJAC_DUB_RSC가 월등한 성능을 보임.
     - CiaoDVD에서는 COS, MSD, Rjaccard보다 낮은 F1-score를 보임.
     - 하지만 CiaoDVD 데이터셋에서 유사도 방법론들 간의 성능 차이는 MAE와 달리 F1-score에서는 미미함.
     - 최근접이웃 𝐾의 개수가 100인 경우)
         - 가장 높은 F1-score(MSD) : 0.819
         - 가장 낮은 F1-score(PCC) : 0.801
     - 최고와 최악의 F1-score 차이는 0.018에 불과하다. -> 무시할만한 차이
     - 따라서 RJAC_DUB_RSC는 일반적으로 MAE 및 F1-score 측면에서 다른 유사도 방법론보다 우수함.

## 결론

#### 연구 요약 및 의의
- Rating_Jaccard의 세 가지 한계점을 제기하고 이를 해결하기 위한 전략을 제안.
- 더 나아가 한계점들을 해결하는 과정에서 발생하는 문제 또한 해결하는 전략 제안.
- 공통으로 평가한 항목 수에 반비례하는 첫 번째 한계점을 고려하는 것이 Rating_Jaccard의 추천 성능을 가장 크게 향상시키는 것으로 나타남.
- 두 번째, 세 번째 한계점을 해결하는 전략 또한 Rating_Jaccard의 성능을 상당히 향상 시킴.
- RJAC_DUB에서 공통 평가 항목을 반영하지 못하는 한계를 극복하기 위해 제안한 RSC라는 요인을 결합한 RJAC_DUB_RSC가 기존 RJAC_DUB보다 성능을 향상시킴.
- RJAC_DUB_RSC를 제안된 후보 방법론들 중에서 가장 성능이 좋은 방법론으로 결정.
- 타 유사도와 비교했을 때에도 RJAC_DUB_RSC가 일반적으로 더 우수한 성능을 나타냄.

#### 연구 한계점 및 추후 연구
- 제안된 유사도 방법론이 다른 방법론과 결합되었을 때의 성능은 평가되지 않았다.
    - 이전의 여러 연구에서 자카드 유사도가 유사도 성능을 향상시키는 데 효과적이라는 것이 입증되었으므로 추후 연구에서 다른 방법론과 함께 제안된 유사도 방법론의 성능을 평가할 수 있음.

- RJAC_DZ 는 사용자별 평점의 편향과 분포를 함께 반영하였지만 사용자별 평점 편향만을 반영한 RJAC_DUB보다 성능이 좋지 않았다.
    - 사용자별 평점의 분포를 반영하는 다양한 방법론에 대한 연구가 필요함.

- Rating_Jaccard에서 RJAC_U로의 성능향상의 정도를 기준으로 RJAC_U에서 RJAC_D, RJAC_DUB, RJAC_DZ, RJAC_DUB_RSC와 RJAC_DUB_RCT로 수식을 개선할 때 주목할만한 성능개선이 이루어지지 않았다.
    - 실험 데이터셋이 평가할 수 있는 평점의 선택지가 많은 데이터셋에서 추후 연구가 필요함.





