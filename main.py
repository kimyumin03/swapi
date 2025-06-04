from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd
import numpy as np

app = FastAPI(title="상권 분석 예측 API")

# =========================
# 1) CORS 설정
# =========================
origins = ["*"]  # 필요에 따라 도메인 제한 가능
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 2) 모델 및 데이터 로드
# =========================
# 모델 파일(model.pkl)과 merged_model 데이터(병합된 DataFrame)를 로드합니다.
try:
    # 예: 피클로 저장된 학습된 모델 (향후 사용할 수도 있지만, 지금은 사용되지 않으므로 미리 로드만)
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    # 모델 파일이 없을 경우, 추후 디버깅용 메시지 출력
    print("Warning: model.pkl 파일을 찾을 수 없습니다.")

try:
    # merged_model.pkl에는 행정동별 모든 피처와 예측 결과가 포함되어 있어야 합니다.
    merged_model = pd.read_pickle("merged_model.pkl")
    # '?' 대신 '.' 로 변환
    merged_model['행정동_코드_명'] = merged_model['행정동_코드_명'].str.replace('?', '.', regex=False)
except FileNotFoundError:
    raise RuntimeError("merged_model.pkl 파일을 찾을 수 없습니다. 해당 파일을 준비해 주세요.")

# =========================
# 3) 공통 헬퍼 함수들
# =========================
# (a) 비율 포맷팅: 소수점 다섯 자리 형식
def format_ratio(x):
    return f"{x:.5f}"

# (b) 만원 단위로 반환 (예: 36000000 → '3600')
def format_money_mw(x):
    return str(int(round(x / 10000)))

# (c) 월 평균 소득을 만원 단위로 반환 (예: 3600000 → '360')
def format_income_mw(x):
    return str(int(round(x / 10000)))

# (d) 텍스트 기반 결과 생성 함수
def generate_result_template(merged_model: pd.DataFrame, 행정동명: str) -> str:
    """
    행정동명에 대한 분석 결과를 자연어 템플릿으로 반환합니다.
    """
    # 예측 등급 텍스트 매핑
    등급_텍스트 = {0: '하', 1: '중', 2: '상'}
    등급_info = {
        '상': {'desc': '매우 높음', 'precision': '77%', 'recommendations': ['카페', '헬스장', '미용실']},
        '중': {'desc': '보통',   'precision': '62%', 'recommendations': ['편의점', '분식집', '세탁소']},
        '하': {'desc': '낮음',   'precision': '69%', 'recommendations': ['중고매장', 'PC방', '호프집']}
    }

    # 해당 행정동 필터링
    filtered = merged_model[merged_model['행정동_코드_명'] == 행정동명].copy()
    if filtered.empty:
        return f"❌ '{행정동명}'에 대한 예측 결과가 없습니다."

    # 예측_등급을 텍스트로 매핑
    filtered['예측_등급_텍스트'] = filtered['예측_등급'].map(등급_텍스트)
    # 중복 제거 후, 가장 빈도가 높은 등급(Top1)을 선택
    summary = filtered[['서비스_업종_코드_명', '예측_등급_텍스트']].drop_duplicates()
    top_grade = summary['예측_등급_텍스트'].value_counts().idxmax()
    info = 등급_info[top_grade]

    # 서울시 전체 평균 값들 계산
    서울시 = merged_model
    서울시_avg_flow       = 서울시['유동인구_변화율'].mean()
    서울시_avg_yoy_pop    = 서울시['전년동기_유동인구_변화율'].mean()
    서울시_avg_sales      = 서울시['당월_매출_금액'].mean()
    서울시_avg_yoy_sales  = 서울시['전년동기_매출_변화율'].mean()
    서울시_avg_income     = 서울시['월_평균_소득_금액'].mean()
    서울시_avg_count      = 서울시.groupby('행정동_코드_명').size().mean()

    # 해당 행정동 평균 값들 계산
    지역_avg_flow   = filtered['유동인구_변화율'].mean()
    지역_avg_yoy_pop  = filtered['전년동기_유동인구_변화율'].mean()
    지역_avg_sales    = filtered['당월_매출_금액'].mean()
    지역_avg_yoy_sales = filtered['전년동기_매출_변화율'].mean()
    지역_avg_income   = filtered['월_평균_소득_금액'].mean()
    지역_count       = filtered.shape[0]

    # 주요 고객 성별 비율
    남_ratio = filtered['남성_유동인구_비율'].mean()
    여_ratio = filtered['여성_유동인구_비율'].mean()
    주성별 = '남성' if 남_ratio >= 여_ratio else '여성'

    # 연령대별 비율 컬럼 목록
    age_pop_cols = [
        '연령대_10_유동인구_수', '연령대_20_유동인구_수',
        '연령대_30_유동인구_수', '연령대_40_유동인구_수',
        '연령대_50_유동인구_수', '연령대_60_이상_유동인구_수'
    ]
    연령대별비율_cols = [f"{col}_비율" for col in age_pop_cols]
    avg_age_ratios = filtered[연령대별비율_cols].mean()
    # 가장 비율이 큰 연령대(예: '연령대_30_유동인구_수_비율' → '연령대_30_유동인구_수' → '30대')
    주연령대 = avg_age_ratios.idxmax().replace('_비율', '')
    주연령대 = 주연령대.replace('연령대_', '').replace('_유동인구_수', '') + "대"

    # 피크 시간대 (mode)
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
    if 지역_count > 서울시_avg_count:
        reasons.append(
            f"이 지역 데이터 건수가 평균보다 많아 예측 신뢰도가 높습니다 ({지역_count}개 > {서울시_avg_count:.1f}개)."
        )
    else:
        reasons.append(
            f"이 지역 데이터 건수가 평균보다 적어 데이터가 제한적일 수 있습니다 ({지역_count}개 < {서울시_avg_count:.1f}개)."
        )

    # 동적 Top3 업종 추천 (예시: 향후 평균 매출 기준)
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

- 모델 전체 정확도는 71%이며, '{info['precision']}' 예측 정밀도는 약 {info['precision']}입니다.

📌 분류된 주요 이유:
"""
    for reason in reasons:
        output += f"- {reason}\n"

    output += f"""
🔹 주요 고객층 및 피크 시간대:
- 주요 고객 성별: {주성별}
- 주요 고객 연령대: {주연령대}
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

# =========================
# 4) 새로 추가할 함수: format_money_eokman, recommend_for_top_grade_v3_updated
# =========================
def format_money_eokman(x):
    """
    금액(x)을 '00억 0000만원' 형태로 반환
    ex) 113018800000 → '1130억 1883만원'
    """
    val = int(round(x))
    eok = val // 100000000
    man = (val % 100000000) // 10000
    return f"{eok}억 {man:04d}만원"

def recommend_for_top_grade_v3_updated(merged_model: pd.DataFrame, 행정동명: str, top_n: int = 5) -> dict:
    """
    • Top1 업종의 '상' 등급인 행정동 중
      2(a) 향후 평균 매출 상위 Top N
      2(b) 피크 시간대 '시간대_17~21'인 Top N (건수 기준)
    • 추가 추천: 전체 행정동 중 ‘상’ 등급 비율 상위 Top N
      (단, '행정동', '총 건수', '상 등급 비율'만 출력)
    """

    # --- 1) 해당 행정동의 Top3 업종 → Top1 선정 ---
    filtered_region = merged_model[merged_model['행정동_코드_명'] == 행정동명].copy()
    if filtered_region.empty:
        raise ValueError(f"'{행정동명}'에 해당하는 데이터가 없습니다.")

    업종별_avg_future = (
        filtered_region
        .groupby('서비스_업종_코드_명')['향후_평균_매출']
        .mean()
        .sort_values(ascending=False)
    )
    if 업종별_avg_future.empty:
        raise ValueError(f"'{행정동명}'의 업종별 향후 평균 매출 데이터가 없습니다.")

    top3_업종 = 업종별_avg_future.head(3).index.tolist()
    top1_업종 = top3_업종[0]

    # --- 2(a) Top1 업종의 '상' 등급 → 향후 평균 매출 Top N ---
    df_industry = merged_model[merged_model['서비스_업종_코드_명'] == top1_업종].copy()
    df_industry_top = df_industry[df_industry['예측_등급'] == 2].copy()  # 예측_등급이 2인, 즉 '상' 등급

    if df_industry_top.empty:
        top_by_sales = pd.DataFrame(columns=['행정동', '평균 향후 매출'])
    else:
        avg_future = (
            df_industry_top
            .groupby('행정동_코드_명')['향후_평균_매출']
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
            .rename(columns={
                '행정동_코드_명': '행정동',
                '향후_평균_매출': '평균 향후 매출(Raw)'
            })
        )
        # 포맷된 매출만 남기기
        avg_future['평균 향후 매출'] = avg_future['평균 향후 매출(Raw)'].map(format_money_eokman)
        top_by_sales = avg_future[['행정동', '평균 향후 매출']]
        top_by_sales.index = range(1, len(top_by_sales) + 1)

    # --- 2(b) '상' 등급 & 피크 시간대 '시간대_17~21' Top N (건수) ---
    mask_peak = (merged_model['예측_등급'] == 2) & (merged_model['최대_시간대_이름'] == '시간대_17~21')
    df_peak = merged_model[mask_peak].copy()
    if df_peak.empty:
        peak_top = pd.DataFrame(columns=['행정동', '건수'])
    else:
        peak_top = (
            df_peak
            .groupby('행정동_코드_명')['최대_시간대_이름']
            .count()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
            .rename(columns={
                '행정동_코드_명': '행정동',
                '최대_시간대_이름': '건수'
            })
        )
        peak_top.index = range(1, len(peak_top) + 1)

    # --- 3) 추가 추천: 전체 행정동 중 ‘상’ 등급 비율 Top N (행정동, 총 건수, 상 등급 비율) ---
    grade_counts = (
        merged_model
        .groupby(['행정동_코드_명', '예측_등급'])['예측_등급']
        .count()
        .unstack(fill_value=0)
    )
    grade_counts['총 건수'] = grade_counts.sum(axis=1)
    grade_counts['상 등급 비율'] = grade_counts.get(2, 0) / grade_counts['총 건수']

    # 2(a) 추천 리스트(행정동) 제외
    exclude = set(top_by_sales['행정동']) if not top_by_sales.empty else set()

    overall_top = (
        grade_counts
        .loc[~grade_counts.index.isin(exclude)]
        .sort_values('상 등급 비율', ascending=False)
        .head(top_n)
        .reset_index()
        .rename(columns={'행정동_코드_명': '행정동'})
    )
    # 필요한 칼럼만 남기기
    overall_top = overall_top[['행정동', '총 건수', '상 등급 비율']]
    overall_top['상 등급 비율'] = overall_top['상 등급 비율'].map(lambda x: f"{x:.5f}")
    overall_top.index = range(1, len(overall_top) + 1)

    return {
        'Top1_업종'               : top1_업종,
        '상등급_향후매출_추천'      : top_by_sales,
        '상등급_피크17~21_추천'   : peak_top,
        '추가_상등급_비율_추천'    : overall_top
    }

# =========================
# 5) 헬스체크 엔드포인트
# =========================
@app.get("/")
def health_check():
    return {"status": "ok", "message": "상권 분석 API가 실행 중입니다."}

# =========================
# 6) 행정동 분석 텍스트 반환 엔드포인트
# =========================
@app.get("/predict")
def predict(dong_name: str = Query(..., description="예측할 행정동명 입력 예: 역삼1동")):
    """
    • dong_name: 분석할 행정동명
    • 반환: 자연어 형태의 분석 결과 텍스트
    """
    try:
        result_text = generate_result_template(merged_model, dong_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"result": result_text}


# =========================
# 7) 세 가지 개별 추천 엔드포인트
#   (1) 향후매출 기준 Top-N만 반환
#   (2) 피크시간대 건수 기준 Top-N만 반환
#   (3) 전체 상등급 비율 기준 Top-N만 반환
# =========================

# (1) 향후매출 기준 Top N만 반환
@app.get("/recommend_sales")
def recommend_sales(
    dong_name: str = Query(..., description="기준 행정동명"),
    top_n: int = Query(5, ge=1, description="상위 몇 개까지")
):
    try:
        results = recommend_for_top_grade_v3_updated(merged_model, dong_name, top_n=top_n)
        sales_df = results['상등급_향후매출_추천']
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "Top1_업종": results['Top1_업종'],
        "상등급_향후매출_추천": sales_df.to_dict(orient="records")
    }

# (2) 피크시간대 건수 기준 Top N만 반환
@app.get("/recommend_peak")
def recommend_peak(
    dong_name: str = Query(..., description="기준 행정동명"),
    top_n: int = Query(5, ge=1, description="상위 몇 개까지")
):
    try:
        results = recommend_for_top_grade_v3_updated(merged_model, dong_name, top_n=top_n)
        peak_df = results['상등급_피크17~21_추천']
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "Top1_업종": results['Top1_업종'],
        "상등급_피크17~21_추천": peak_df.to_dict(orient="records")
    }

# (3) 전체 상등급 비율 기준 Top N만 반환
@app.get("/recommend_overall")
def recommend_overall(
    dong_name: str = Query(..., description="기준 행정동명"),
    top_n: int = Query(5, ge=1, description="상위 몇 개까지")
):
    try:
        results = recommend_for_top_grade_v3_updated(merged_model, dong_name, top_n=top_n)
        overall_df = results['추가_상등급_비율_추천']
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "Top1_업종": results['Top1_업종'],
        "추가_상등급_비율_추천": overall_df.to_dict(orient="records")
    }


# =========================
# 8) 통합 추천 엔드포인트 (/recommend_top)
# =========================
@app.get("/recommend_top")
def recommend_top(
    dong_name: str = Query(..., description="기준 행정동명"),
    top_n: int = Query(5, ge=1, description="상위 몇 개까지")
):
    """
    • dong_name: 기준이 되는 행정동명
    • top_n: 상위 몇 개 행정동까지 추천할지
    • 반환: Top1 업종, 세 가지(향후매출 기준, 피크시간대 기준, 전체 비율 기준) 추천 리스트
    """
    try:
        results = recommend_for_top_grade_v3_updated(merged_model, dong_name, top_n=top_n)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    sales_df   = results['상등급_향후매출_추천']
    peak_df    = results['상등급_피크17~21_추천']
    overall_df = results['추가_상등급_비율_추천']

    return {
        "Top1_업종": results['Top1_업종'],
        "상등급_향후매출_추천": sales_df.to_dict(orient="records"),
        "상등급_피크17~21_추천": peak_df.to_dict(orient="records"),
        "추가_상등급_비율_추천": overall_df.to_dict(orient="records"),
    }
