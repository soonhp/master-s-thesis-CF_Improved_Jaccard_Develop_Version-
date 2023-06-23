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

### 제안방법
#### RJAC_U(Rating_Jaccard의 첫번째 한계점)
- 자카드 유사도의 성능은 두 사용자가 같이 평가한 항목과 비례관계인데 Rating_Jaccard의 수식에서는 반비례관계가 된다. → 사용자가 평가한 모든 아이템을 반영하는 개선된 유사도를 제안.
- Rating_jaccard 수식 내에서 분모에 두 사용자가 공통으로 평가한 항목의 수(교집합) 대신 두 사용자가 평가한 모든 항목의 수(합집합)로 변경.
- 본 연구에서는 이것을 RJAC_U(Rating jaccard with Union)로 명명.
![image](https://github.com/soonhp/master_thesis-CF_Improved_Jaccard_Develop_Version/assets/73877159/da3e8199-8672-4991-80e5-381d52735889)




#### RJAC_DZ(Rating_Jaccard의 세번째 한계점)
- RJAC_DUB에서 확장하여 사용자별 표준편차를 반영하는 유사도 제안.
  - 각 사용자의 평점과 평균평점의 차이를 표준편차로 나눠주면 Z-SCORE가 됨.
  - Z-SCORE 간의 차이에 절대값을 씌운 값이 𝑻𝑯 이하인 항목의 수를 두 사용자가 평가한 모든 항목의 수로 나눠줌.
  - RJAC_DUB와 달리 사용자별 평점의 편향과 분포를 동시에 반영.
- 본 연구에서는 이것을 RJAC_DZ(Rating_Jaccard allowing a small difference in ratings’ z_score considering user biases)로 명명.
  
![image](https://github.com/soonhp/master_thesis-CF_Improved_Jaccard_Develop_Version/assets/73877159/27463a5f-b866-43dd-a761-18805d39cbbd)




