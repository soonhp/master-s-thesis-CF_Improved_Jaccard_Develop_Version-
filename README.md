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
