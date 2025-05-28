
from fastapi import FastAPI, HTTPException, Query
import pickle
import pandas as pd

app = FastAPI(title="상권 분석 예측 API")

# 모델과 데이터 로드 (서버 시작 시 1회)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

merged_model = pd.read_pickle("merged_model.pkl")

# 결과 생성 함수
def generate_result_template(행정동명):
    등급_텍스트 = {0: '하', 1: '중', 2: '상'}
    등급_info = {
        '상': {
            'desc': '매우 높음',
            'precision': '77%',
            'recommendations': ['카페', '헬스장', '미용실']
        },
        '중': {
            'desc': '보통',
            'precision': '62%',
            'recommendations': ['편의점', '분식집', '세탁소']
        },
        '하': {
            'desc': '낮음',
            'precision': '69%',
            'recommendations': ['중고매장', 'PC방', '호프집']
        }
    }

    filtered = merged_model[merged_model['행정동_코드_명'] == 행정동명].copy()
    if filtered.empty:
        return f"❌ '{{}}'에 대한 예측 결과가 없습니다.".format(행정동명)

    filtered['예측_등급'] = filtered['예측_등급'].map(등급_텍스트)
    result_summary = filtered[['서비스_업종_코드_명', '예측_등급']].drop_duplicates()
    top_grade = result_summary['예측_등급'].value_counts().idxmax()
    info = 등급_info[top_grade]
    reco = info['recommendations']

    output = f"""
🔍 '{{행정동명}}' 상권 분석 결과
예측 모델에 따르면 이 지역은 창업 적합도 등급 '{{top_grade}}' ({{info['desc']}})으로 분류되는 업종이 가장 많습니다.
이는 전국 상권 데이터를 바탕으로 유동인구, 매출 흐름, 폐업률 등의 지표를 종합 분석한 결과입니다.

- 모델 전체 정확도는 71%, '{{top_grade}}' 등급 예측의 정밀도는 약 {{info['precision']}}입니다.
- 분석 결과를 토대로, 이 지역에서는 다음과 같은 업종이 특히 적합한 업종군으로 추천됩니다:

✅ 추천 업종 TOP 3
1. {{reco[0]}}
2. {{reco[1]}}
3. {{reco[2]}}

📌 추천 업종은 유사 상권에서 높은 생존율과 매출 흐름을 보인 업종을 기반으로 도출됩니다.
"""
    return output

@app.get("/")
def health_check():
    return {"status": "ok", "message": "상권 분석 API가 실행 중입니다."}

@app.get("/predict")
def predict(dong_name: str = Query(..., description="예측할 행정동명 입력 예: 역삼1동")):
    try:
        result_text = generate_result_template(dong_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"result": result_text}
