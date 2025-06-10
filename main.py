from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd
import numpy as np

app = FastAPI(title="상권 분석 & 추천 API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

merged_model = pd.read_pickle("merged_model.pkl")
merged_model['행정동_코드_명'] = merged_model['행정동_코드_명'].str.replace('?', '.', regex=False)

age_pop_cols = [
    '연령대_10_유동인구_수', '연령대_20_유동인구_수',
    '연령대_30_유동인구_수', '연령대_40_유동인구_수',
    '연령대_50_유동인구_수', '연령대_60_이상_유동인구_수'
]

def format_ratio(x):
    return f"{x:.5f}"

def format_money_mw(x):
    """
    금액(x)을 만원 단위로만 반환
    ex) 36000000 → '3600'
    """
    return str(int(round(x / 10000)))

def format_income_mw(x):
    """
    월 평균 소득(x)을 만원 단위로 반환
    ex) 3600000 → '360'
    """
    return str(int(round(x / 10000)))

# =========================
# 16) 리팩토링된 결과 출력 함수
# =========================

def generate_result_template(merged_model, 행정동명):
    등급_텍스트 = {0: '하', 1: '중', 2: '상'}
    등급_info = {
        '상': {'desc': '매우 높음', 'precision': '77%', 'recommendations': ['카페', '헬스장', '미용실']},
        '중': {'desc': '보통', 'precision': '62%', 'recommendations': ['편의점', '분식집', '세탁소']},
        '하': {'desc': '낮음', 'precision': '69%', 'recommendations': ['중고매장', 'PC방', '호프집']}
    }

    filtered = merged_model[merged_model['행정동_코드_명'] == 행정동명].copy()
    if filtered.empty:
        return f"❌ '{행정동명}'에 대한 예측 결과가 없습니다."

    filtered['예측_등급_텍스트'] = filtered['예측_등급'].map(등급_텍스트)
    summary = filtered[['서비스_업종_코드_명', '예측_등급_텍스트']].drop_duplicates()
    top_grade = summary['예측_등급_텍스트'].value_counts().idxmax()
    info = 등급_info[top_grade]

    서울시 = merged_model
    서울시_avg_flow = 서울시['유동인구_변화율'].mean()
    서울시_avg_yoy_pop = 서울시['전년동기_유동인구_변화율'].mean()
    서울시_avg_sales = 서울시['당월_매출_금액'].mean()
    서울시_avg_yoy_sales = 서울시['전년동기_매출_변화율'].mean()
    서울시_avg_income = 서울시['월_평균_소득_금액'].mean()

    지역_avg_flow = filtered['유동인구_변화율'].mean()
    지역_avg_yoy_pop = filtered['전년동기_유동인구_변화율'].mean()
    지역_avg_sales = filtered['당월_매출_금액'].mean()
    지역_avg_yoy_sales = filtered['전년동기_매출_변화율'].mean()
    지역_avg_income = filtered['월_평균_소득_금액'].mean()
    지역_count = filtered.shape[0]

    남_ratio = filtered['남성_유동인구_비율'].mean()
    여_ratio = filtered['여성_유동인구_비율'].mean()
    주성별 = '남성' if 남_ratio >= 여_ratio else '여성'

    연령대별비율_cols = [f"{col}_비율" for col in age_pop_cols]
    avg_age_ratios = filtered[연령대별비율_cols].mean()
    주연령대 = avg_age_ratios.idxmax().replace('_비율', '')

    peak_time = filtered['최대_시간대_이름'].mode().iloc[0]

    reasons = []
    # 1) 유동인구 변화율 비교
    if 지역_avg_flow > 서울시_avg_flow:
        reasons.append(
            f"유동인구 변화율이 서울시 평균보다 높습니다 ({format_ratio(지역_avg_flow)} > {format_ratio(서울시_avg_flow)})."
        )
    else:
        reasons.append(
            f"유동인구 변화율이 서울시 평균보다 낮습니다 ({format_ratio(지역_avg_flow)} < {format_ratio(서울시_avg_flow)})."
        )
    # 2) 전년 동기 대비 유동인구 변화율 “추세”
    if np.isclose(지역_avg_yoy_pop, 서울시_avg_yoy_pop):
        reasons.append("전년 동기 대비 유동인구 변화가 서울시 평균과 유사한 수준입니다.")
    elif 지역_avg_yoy_pop > 서울시_avg_yoy_pop:
        reasons.append("전년 동기 대비 유동인구가 서울시 평균보다 증가 추세를 보입니다.")
    else:
        reasons.append("전년 동기 대비 유동인구가 서울시 평균보다 감소 추세를 보입니다.")
    # 3) 현재 매출 비교
    if 지역_avg_sales > 서울시_avg_sales:
        reasons.append(
            f"현재(당월) 평균 매출이 서울시 평균보다 높습니다 ({format_money_mw(지역_avg_sales)}만 원 > {format_money_mw(서울시_avg_sales)}만 원)."
        )
    else:
        reasons.append(
            f"현재(당월) 평균 매출이 서울시 평균보다 낮습니다 ({format_money_mw(지역_avg_sales)}만 원 < {format_money_mw(서울시_avg_sales)}만 원)."
        )
    # 4) 전년 동기 대비 매출 변화율 “추세”
    if np.isclose(지역_avg_yoy_sales, 서울시_avg_yoy_sales):
        reasons.append("전년 동기 대비 매출 변화가 서울시 평균과 유사한 수준입니다.")
    elif 지역_avg_yoy_sales > 서울시_avg_yoy_sales:
        reasons.append("전년 동기 대비 매출이 서울시 평균보다 상승 추세를 보입니다.")
    else:
        reasons.append("전년 동기 대비 매출이 서울시 평균보다 하락 추세를 보입니다.")
    # 5) 평균 소득 비교 (단위: 만원)
    if 지역_avg_income > 서울시_avg_income:
        reasons.append(
            f"월 평균 소득이 서울시 평균보다 높습니다 ({format_income_mw(지역_avg_income)}만 원 > {format_income_mw(서울시_avg_income)}만 원)."
        )
    else:
        reasons.append(
            f"월 평균 소득이 서울시 평균보다 낮습니다 ({format_income_mw(지역_avg_income)}만 원 < {format_income_mw(서울시_avg_income)}만 원)."
        )
    # 6) 데이터 건수 비교
    서울시_avg_count = 서울시.groupby('행정동_코드_명').size().mean()
    if 지역_count > 서울시_avg_count:
        reasons.append(
            f"이 지역 데이터 건수가 평균보다 많아 예측 신뢰도가 높습니다 ({지역_count}개 > {서울시_avg_count:.1f}개)."
        )
    else:
        reasons.append(
            f"이 지역 데이터 건수가 평균보다 적어 데이터가 제한적일 수 있습니다 ({지역_count}개 < {서울시_avg_count:.1f}개)."
        )

    # 동적 Top3 업종 추천
    업종별_avg_future = (
        filtered
        .groupby('서비스_업종_코드_명')['향후_평균_매출']
        .mean()
        .sort_values(ascending=False)
    )
    top3_업종 = 업종별_avg_future.head(3).index.tolist()

    fixed_reco = 등급_info[top_grade]['recommendations']

    output = f"""
🔍 '{행정동명}' 상권 분석 결과
예측 모델에 따르면 이 지역은 창업 적합도 등급 '{top_grade}' ({info['desc']})으로 분류되는 업종이 가장 많습니다.
이는 서울시 상권 데이터를 바탕으로 유동인구 변화, 매출 흐름, 소득 수준 등 다양한 지표를 종합 분석한 결과입니다.

- 모델 전체 정확도는 71%이며, '{top_grade}' 등급 예측 정밀도는 약 {info['precision']}입니다.

📌 분류된 주요 이유:
"""
    for reason in reasons:
        output += f"- {reason}\n"

    output += f"""
🔹 주요 고객층 및 피크 시간대:
- 주요 고객 성별: {주성별}
- 주요 고객 연령대: {주연령대.replace('연령대_', '').replace('_유동인구_수', '')}대
- 피크 매출 시간대: {peak_time}

✅ '{행정동명}' 지역 동적 추천 업종 Top 3:
1. {top3_업종[0]}
2. {top3_업종[1]}
3. {top3_업종[2]}

🔹 '{top_grade}' 등급에 속할 때 추천 업종:
"""
    for idx, 업종 in enumerate(fixed_reco, start=1):
        output += f"{idx}. {업종}\n"

    return output.strip()

def format_money_eokman(x):
    """
    원 단위 금액(x)을 '000억 0000만' 형태로 반환
    내부적으로 1만 원 단위 과도 적용 시 보정합니다.
    ex) 113018800000 → '1130억 1883만'
    """
    # 만원 단위 과도 적용 보정
    val_corrected = x // 10000
    val = int(round(val_corrected))
    # 억 단위(100,000,000원)
    eok = val // 100_000_000
    man = (val % 100_000_000) // 10_000
    return f"{eok}억 {man:04d}만"

# =========================
# 상등급 추천 함수
# =========================
def recommend_for_top_grade_v3_updated(merged_model, 행정동명, top_n=5):
    # 대상 행정동 필터링
    filtered = merged_model[merged_model['행정동_코드_명'] == 행정동명]
    if filtered.empty:
        raise ValueError(f"'{행정동명}'에 해당하는 데이터가 없습니다.")
    # Top1 업종
    top1 = (filtered.groupby('서비스_업종_코드_명')['향후_평균_매출']
            .mean().sort_values(ascending=False).index[0])
    results = {'Top1_업종': top1}

    # 2(a) 해당 업종 상등급 매출 상위
    df_top = merged_model[(merged_model['서비스_업종_코드_명'] == top1) &
                          (merged_model['예측_등급'] == 2)]
    top_by_sales = pd.DataFrame(columns=['행정동','평균 향후 매출'])
    if not df_top.empty:
        tmp = (df_top.groupby('행정동_코드_명')['향후_평균_매출']
               .mean().sort_values(ascending=False)
               .drop(index=행정동명, errors='ignore')
               .head(top_n)
               .reset_index().rename(columns={'행정동_코드_명':'행정동','향후_평균_매출':'평균 향후 매출(Raw)'})
        )
        tmp['평균 향후 매출'] = tmp['평균 향후 매출(Raw)'].map(format_money_eokman)
        top_by_sales = tmp[['행정동','평균 향후 매출']]
        top_by_sales.index = range(1, len(top_by_sales)+1)
    results['상등급_향후매출_추천'] = top_by_sales

    # 2(b) 피크 시간대 유동인구 상위
    df_peak = merged_model[(merged_model['예측_등급']==2)&
                           (merged_model['최대_시간대_이름']=='시간대_17~21')]
    peak_top = pd.DataFrame(columns=['행정동','건수'])
    if not df_peak.empty:
        tmp = (df_peak.groupby('행정동_코드_명').size()
               .sort_values(ascending=False)
               .drop(index=행정동명, errors='ignore')
               .head(top_n)
               .reset_index(name='건수').rename(columns={'행정동_코드_명':'행정동'})
        )
        tmp.index = range(1, len(tmp)+1)
        peak_top = tmp
    results['상등급_피크유동인구_추천'] = peak_top

    # 3) 전체 상등급 비율 추천
    grade_counts = (merged_model.groupby(['행정동_코드_명','예측_등급'])['예측_등급']
                   .count().unstack(fill_value=0))
    grade_counts['데이터 건수'] = grade_counts.sum(axis=1)
    grade_counts['비율(%)'] = grade_counts.get(2,0)/grade_counts['데이터 건수']*100
    overall = (grade_counts.sort_values('비율(%)',ascending=False)
               .drop(index=행정동명,errors='ignore')
               .head(top_n)
               .reset_index().rename(columns={'행정동_코드_명':'행정동'})
    )
    overall = overall[['행정동','데이터 건수','비율(%)']]
    overall['비율(%)'] = overall['비율(%)'].map(lambda x:f"{x:.2f}%")
    overall.index = range(1, len(overall)+1)
    results['추가_상등급_비율_추천'] = overall

    return results

# =========================
# 중등급 추천 함수
# =========================
def recommend_for_mid_grade_growth(merged_model, 행정동명, top_n=5):
    filtered = merged_model[merged_model['행정동_코드_명']==행정동명]
    if filtered.empty:
        raise ValueError(f"'{행정동명}'에 해당하는 데이터가 없습니다.")
    df_mid = merged_model[merged_model['예측_등급']==1]

    sales_growth = (df_mid.groupby('행정동_코드_명')['매출_변화율']
                    .mean().sort_values(ascending=False)
                    .drop(index=행정동명,errors='ignore')
                    .head(top_n)
                    .reset_index().rename(columns={'행정동_코드_명':'행정동','매출_변화율':'평균 매출 증가율'})
    )
    sales_growth.index = range(1,len(sales_growth)+1)

    pop_growth = (df_mid.groupby('행정동_코드_명')['전년동기_유동인구_변화율']
                  .mean().sort_values(ascending=False)
                  .drop(index=행정동명,errors='ignore')
                  .head(top_n)
                  .reset_index().rename(columns={'행정동_코드_명':'행정동','전년동기_유동인구_변화율':'평균 유동인구 증가율'})
    )
    pop_growth.index = range(1,len(pop_growth)+1)

    return {'중등급_매출증가_추천':sales_growth,'중등급_유동인구증가_추천':pop_growth}

# =========================
# 하등급 추천 함수
# =========================
def recommend_for_low_grade_risk(merged_model, 행정동명, top_n=5):
    filtered = merged_model[merged_model['행정동_코드_명']==행정동명]
    if filtered.empty:
        raise ValueError(f"'{행정동명}'에 해당하는 데이터가 없습니다.")
    df_low = merged_model[merged_model['예측_등급']==0]

    # 1) 폐업 위험
    closure_risk = pd.DataFrame(columns=['행정동','평균 영업 개월'])
    if '폐업_영업_개월_평균' in df_low and not df_low.empty:
        tmp = (df_low.groupby('행정동_코드_명')['폐업_영업_개월_평균']
               .mean().round(1).sort_values()
               .drop(index=행정동명,errors='ignore').head(top_n)
               .reset_index().rename(columns={'행정동_코드_명':'행정동','폐업_영업_개월_평균':'평균 영업 개월'})
        )
        tmp.index = range(1,len(tmp)+1)
        closure_risk = tmp

    # 2) 소득 낮음
    income_low = pd.DataFrame(columns=['행정동','평균 소득'])
    if '월_평균_소득_금액' in df_low and not df_low.empty:
        tmp2 = (df_low.groupby('행정동_코드_명')['월_평균_소득_금액']
                .mean().sort_values()
                .drop(index=행정동명,errors='ignore').head(top_n)
                .reset_index().rename(columns={'행정동_코드_명':'행정동','월_평균_소득_금액':'평균 소득(Raw)'})
        )
        tmp2['평균 소득'] = tmp2['평균 소득(Raw)'].floordiv(10000).astype(int).astype(str)+'만 원'
        income_low = tmp2[['행정동','평균 소득']]
        income_low.index = range(1,len(income_low)+1)

    # 3) 매출 낮음
    sales_low = pd.DataFrame(columns=['행정동','평균 매출'])
    if '당월_매출_금액' in df_low and not df_low.empty:
        tmp3 = (df_low.groupby('행정동_코드_명')['당월_매출_금액']
                .mean().sort_values()
                .drop(index=행정동명,errors='ignore').head(top_n)
                .reset_index().rename(columns={'행정동_코드_명':'행정동','당월_매출_금액':'평균 매출(Raw)'})
        )
        tmp3['평균 매출'] = tmp3['평균 매출(Raw)'].map(format_money_eokman)
        sales_low = tmp3[['행정동','평균 매출']]
        sales_low.index = range(1,len(sales_low)+1)

    return {'하등급_폐업위험_추천':closure_risk,
            '하등급_소득낮은_추천':income_low,
            '하등급_매출낮은_추천':sales_low}

# =========================
# 전체 리포트 출력 함수
# =========================
def print_recommendation_report(merged_model, 행정동명, top_n=5):
    filtered = merged_model[merged_model['행정동_코드_명']==행정동명]
    if filtered.empty:
        raise ValueError(f"'{행정동명}'에 해당하는 데이터가 없습니다.")
    local_grade = int(filtered['예측_등급'].mode().iloc[0])

    if local_grade==2:
        top = recommend_for_top_grade_v3_updated(merged_model,행정동명,top_n)
        print(f"▶ Top1 업종: {top['Top1_업종']}\n")
        print(f"▶ 상등급 매출 상위 {top_n}개:")
        print(top['상등급_향후매출_추천'],"\n")
        print(f"▶ 피크 시간대 유동인구 상위 {top_n}개:")
        print(top['상등급_피크유동인구_추천'],"\n")
        print(f"▶ 상등급 비율 상위 {top_n}개(중복제외):")
        print(top['추가_상등급_비율_추천'],"\n")
    elif local_grade==1:
        mid = recommend_for_mid_grade_growth(merged_model,행정동명,top_n)
        print(f"▶ 중등급 매출 증가율 상위 {top_n}개 (자기 제외):")
        print(mid['중등급_매출증가_추천'],"\n")
        print(f"▶ 중등급 유동인구 증가율 상위 {top_n}개 (자기 제외):")
        print(mid['중등급_유동인구증가_추천'],"\n")
    else:
        low = recommend_for_low_grade_risk(merged_model,행정동명,top_n)
        print(f"▶ 하등급 폐업 위험 상위 {top_n}개 (영업 개월 짧은 순):")
        print(low['하등급_폐업위험_추천'],"\n")
        print(f"▶ 하등급 소득 낮은 순 상위 {top_n}개:")
        print(low['하등급_소득낮은_추천'],"\n")
        print(f"▶ 하등급 매출 낮은 순 상위 {top_n}개:")
        print(low['하등급_매출낮은_추천'],"\n")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "상권 분석 API가 실행 중입니다."}

@app.get("/predict")
def predict(dong_name: str = Query(..., description="예측할 행정동명 입력 예: 역삼1동")):
    try:
        result_text = generate_result_template(merged_model, dong_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"result": result_text} 

@app.get("/recommend/all")
def recommend_all(
    dong_name: str = Query(..., description="추천 기준 행정동명"),
    top_n:    int   = Query(5,   description="추천 상위 N개")
):
    base = merged_model[merged_model['행정동_코드_명'] == dong_name]
    if base.empty:
        raise HTTPException(status_code=404, detail=f"'{dong_name}' 데이터가 없습니다.")
    
    try:
        grade = int(base['예측_등급'].mode().iloc[0])
    except:
        raise HTTPException(status_code=500, detail="예측 등급 추출 중 오류 발생")

    response = {"등급": grade}

    try:
        if grade == 2:
            result = recommend_for_top_grade_v3_updated(merged_model, dong_name, top_n)
            response.update({
                "추천_기준": "상등급",
                "Top1_업종": result['Top1_업종'],
                "향후매출_상위": result['상등급_향후매출_추천'].reset_index().to_dict(orient="records"),
                "피크유동인구_상위": result['상등급_피크유동인구_추천'].reset_index().to_dict(orient="records"),
                "상등급_비율_상위": result['추가_상등급_비율_추천'].reset_index().to_dict(orient="records"),
            })
        elif grade == 1:
            result = recommend_for_mid_grade_growth(merged_model, dong_name, top_n)
            response.update({
                "추천_기준": "중등급",
                "매출증가_상위": result['중등급_매출증가_추천'].reset_index().to_dict(orient="records"),
                "유동인구증가_상위": result['중등급_유동인구증가_추천'].reset_index().to_dict(orient="records"),
            })
        else:
            result = recommend_for_low_grade_risk(merged_model, dong_name, top_n)
            response.update({
                "추천_기준": "하등급",
                "폐업위험_상위": result['하등급_폐업위험_추천'].reset_index().to_dict(orient="records"),
                "소득낮은_상위": result['하등급_소득낮은_추천'].reset_index().to_dict(orient="records"),
                "매출낮은_상위": result['하등급_매출낮은_추천'].reset_index().to_dict(orient="records"),
            })
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추천 처리 중 오류: {str(e)}")

    return response
