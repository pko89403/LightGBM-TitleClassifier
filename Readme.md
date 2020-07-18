# Title Classifier Using LightGBM with Tokenizer
아래의 사용 데이터의 형태를 학습 데이터셋으로 사용해 영어가 포함된 제목으로 해당 컨텐츠가 어떤 유형 인지 분류하는 분류기를 LightGBM(왜냐면 핫하니깐)을 사용해서 진행 해본다.
## 사용 데이터
~~~csv
titles	labels
뭔가의제목	분류
~~~
## 과정 정리 
특정 키워드의 존재 여부가 클래스로 분류 되는데 영향을 많이 작용할 것이라고 예상 했습니다.   

Bag Of Words 형태로 입력 데이터를 토큰화하고    
 의사 결정 트리(lightGBM)를 사용한 분류가 성능이 좋을 것이라고 판단 했습니다.     

분류 결과의 전체 Accuracy는 학습 데이터 셋에서 87%, 테스트 데이터 셋에서 86%로 나쁘지 않았지만,     
학습 데이터 셋에서 데이터가 적은 클래스의 경우 클래스로 분류가 되지 않았습니다.     
( 해당 클래스는 일반화하는 과정에서 제외 되었다고 생각해 특별 취급을 해주었습니다. )    

각 클래스 토큰 등장 횟수 상위( TF 변형 ) + 토큰 별 클래스에 속할 확률( DF 변형 )을 사용해서     
각 클래스 별 토큰 등장 횟수 상위 20개에 각 타이틀이 속한다면,     
해당 토큰이 특정 클래스로 속할 확률들을 평균의 형태로 결합해 분류를 했습니다.

전체 성능은 좋은 편이 아니었지만, 데이터가 적은 클래스의 분류가 가능해졌기 떄문에      
lightGBM을 사용한 결과와 SUM을 해서 분류 결과를 생성했습니다.

SUM을 한 결과 전체 Accuracy는 변화가 적었지만, F1 Score의 경우 학습 데이터 셋에서 5% 테스트 데이터 셋에서 8%의 성능 개선이 있었습니다.

성능 평가 metric으로 top1, 3, 5의 각 클래스 별 Accuracy, 각 클래스 별 f1 Score를 사용했습니다.


## 실행 방법 
~~~sh 
conda env create -f environment.yml # conda 가상 환경 생성
conda activate BankSalad            # conda 가상 환경 활성화 
python main.py                      # 학습 및 테스트 실행
~~~ 

## 실행 과정
1. 전체 타이틀을 토크나이징 합니다.( Tokenizer.py의 Tokenizer 클래스 )
   - 전처리로 특수문자 및 숫자를 제거합니다.
   - okt (한국어)
   - mecab (한국어)
   - soynlp (학습 데이터 학습 결과)
   - spacy (영어)
2. sklearn.TfidfVectorizer를 사용해 학습에 사용할 token 들로 feature로 변형하는 학습 및 저장합니다. ( Vectorizer.py 의 Vectorizer 클래스 )
3. lightGBM로 feature 들로 17개로 분류할 모델을 학습 및 저장합니다. ( Classifier.py 의 LGB_Classifer 클래스 )
4. Count 기반 클래스 분류는 feature가 아닌 token을 사용해 학습 및 저장합니다. ( Classifier.py 의 CountBased_Classifier 클래스 )
5. 학습 데이터를 기반으로 성능 평가를 진행합니다. ( Metrics.py 의 Report 함수 )
6. 테스트 데이터를 기반으로 성능 평가를 진행합니다.( Metrics.py 의 Report 함수 )
7. 성능 평가 파일은 Report.txt 파일을 생성 합니다.

## 실행 결과 파일
각 클래스 별 Token의 등장 횟수 Top 20 들 각각이 어떤 Class에 속한 빈도를 퍼센트로 계산한 목록은 Report.txt에 있습니다.      

latency를 제외한 각 클래스 별 Accuracy와 F1 Score의 경우에는    
 train과 test 데이터 셋에 대해 각각 성능 측정한 파일이 있습니다.
- train_ml_pred.report
- train_stat_pred.report
- train_ensamble_pred.report
- test_ml_pred.report
- test_stat_pred.report
- test_ensamble_pred.report

학습 데이터 셋에 대해 전체 과정과 성능 측정 결과 까지 걸리는 시간은 아래와 같습니다.    
> Training Done at 600181.477305 ms seconds


테스트 데이터 셋에 대해 각각 기능의 latency를 측정한 결과는 아래와 같습니다. 마찬가지로 Report.txt 에 있습니다.    
>Testing samples : 4748
Testing Done at 42587.17548800007 ms seconds
Tokenize Time : 34205.46124600003 ms seconds
Vectorized Time : 437.13029799994274 ms seconds
ML Inference Time : 4617.86647100007 ms seconds
STAT Inference Time : 1607.5382579999768 ms seconds


## 추가 자료
### LightGBM
Gradient Boosting 프레임워크로 Tree 기반 학습 알고리즘      
LightGBM은 Tree를 수직으로 확장 시킨다(leaf-wise)    
보통 수평으로 확장한다.(level-wise)    
gpu를 지원합니다
### Tokenizer
okt 하나만 사용하는 경우 시간이 단축이 가능하나 성능이 accuracy 차가 8%가 발생했습니다.

### 맥북 에어에서 진행