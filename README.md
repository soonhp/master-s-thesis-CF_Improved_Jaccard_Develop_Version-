# master_thesis-CF_Improved_Jaccard_Develop_Version(Paper_Code)

## 석사학위논문
### 연구주제 : 평점을 고려한 개선된 자카드 유사도 기반 사용자 협업필터링 추천시스템

### 연구동기
#### Rating_Jaccard

- Rating_Jaccard는 두 사용자 간의 평점이 같은 항목의 개수를 두 사용자가 같이 평가한 항목의 비율로 구하는데 이로 인해 한계점들이 발생.

  1) 일반적으로 두 사용자 유사도는 co-rated item에 비례. But, Rating_Jaccard는 반비례

  2) 평점이 같은 경우 만을 고려하기 때문에 유사도가 0이 되는 경우가 다수 발생.

  3) 각 사용자별 평점의 편향을 반영하지 못함.

- 본 연구에서는 이러한 한계점들을 해결하기 위한 새로운 유사도 제안. 더 나아가 제안 방법에서 발생한 한계에 대해서도 공통항목의 수를 고려하는 확장된 방법을 제안하여 해결하고자 한다.  
- Park, S. H., & Kim, K. (2023). Collaborative filtering recommendation system based on improved Jaccard similarity. Journal of Ambient Intelligence and Humanized Computing, 1-18. -> 본 논문을 develop한 연구.

### 제안방법(기존 논문에서 추가된 내용)
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
- 제안 유사도별로 각각 가중치를 부여 → RJAC_DUB 예시 :  (〖|𝑵_𝑻𝑫𝑼𝑩 (𝒂,𝒃)|/|𝑰_𝒂∩𝑰_𝒃 | )〗^(𝟏/𝟐)
- 가중치 term에 𝟏/𝟐 제곱을 하는 이유 : 사용자 간 공통항목이 매우 많은 경우 가중치가 지나치게 감소할 수 있으므로 보정을 해주기 위함.
- 공통항목의 수와 유사하다고 판단되는 항목의 수를 모두 반영하는 유사도
- 본 연구에서는 이것을 RSC(ratio of similar items among co-rated items) 로 명명.
- RJAC_DUB 예시)



