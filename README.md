KAGGLE MCTS challenge!

**1. EDA**

   **1.1 problem with challenge**
   
    MCTS의 여러가지 세부 방식으로 만든 모델(agent)들이 있고, 그들이 다른 agent에 비해 각각 어느 정도의 좋은 성능을 보이는 지 예측하는 문제
       
    성능은 각각의 agent들이 보드게임을 플레이하도록 해서 얼마나 상대적으로 더 많이 이겼는지를 기준으로 판단(target value)
       
    MCTS: 몬테카를로 트리 샘플링 / RL에서 중요한 갈래인 몬테카를로 방식으로 현재 game에서 가장 유망한 방법론


    어려운 점: 1. many features, 약 800개의 해당하는 features가 존재
               2. dealing with String, agent가 플레이하는 보드게임을 간략하게 ludii 형식(보드게임을 플레이하도록 고안된 언어 방식)으로 표현되어 있는데, type이 String이다. 구체적으로 어떻게 vectorization할 것인가?

   **1.2 several information from EDA**
   1. identity(constant) data, NULL data 존재: 모든 sample에서 값이 null이거나 constant인 data가 존재한다.
   2. 상관관게 분석: 
