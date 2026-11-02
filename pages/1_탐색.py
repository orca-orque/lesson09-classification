import pandas as pd
import plotly.express as px
import streamlit as st

APP_TITLE = "뇌졸중 예측 실습실"
DATA_URL = "https://raw.githubusercontent.com/orca-orque/data/main/stroke.csv"

st.set_page_config(page_title=f"탐색 - {APP_TITLE}", page_icon="🧠", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def stroke_rate(group: pd.DataFrame) -> float:
    """뇌졸중 비율(%). 사람이 없으면 0을 돌려준다."""
    return float((group["stroke"] == 1).mean() * 100) if len(group) else 0.0


st.title("📊 탐색")
st.caption("데이터를 여러 방향에서 살펴봐요.")

try:
    df = load_data()
except Exception as e:  # 인터넷 연결이나 주소 문제
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n{e}")
    st.stop()

GROUP_NAMES = {0: "뇌졸중 겪지 않음", 1: "뇌졸중 겪음"}
GROUP_ORDER = ["뇌졸중 겪지 않음", "뇌졸중 겪음"]
df = df.assign(뇌졸중_여부=df["stroke"].map(GROUP_NAMES))

# ------------------------------------------------------------ 1) 히스토그램
st.header("1. 나이와 평균 혈당의 분포")
col1, col2 = st.columns(2)
with col1:
    fig = px.histogram(df, x="age", nbins=30, title="나이 분포")
    fig.update_layout(xaxis_title="나이(세)", yaxis_title="사람 수")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.histogram(df, x="avg_glucose_level", nbins=40, title="평균 혈당 분포")
    fig.update_layout(xaxis_title="평균 혈당(avg_glucose_level)", yaxis_title="사람 수")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------ 2) 상자그림
st.header("2. 뇌졸중을 겪은 사람과 겪지 않은 사람 비교")
col1, col2 = st.columns(2)
with col1:
    fig = px.box(
        df,
        x="뇌졸중_여부",
        y="age",
        category_orders={"뇌졸중_여부": GROUP_ORDER},
        title="나이 비교",
    )
    fig.update_layout(xaxis_title="", yaxis_title="나이(세)")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.box(
        df,
        x="뇌졸중_여부",
        y="avg_glucose_level",
        category_orders={"뇌졸중_여부": GROUP_ORDER},
        title="평균 혈당 비교",
    )
    fig.update_layout(xaxis_title="", yaxis_title="평균 혈당(avg_glucose_level)")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("두 그룹의 평균값")
means = (
    df.groupby("뇌졸중_여부")
    .agg(사람_수=("id", "count"), 평균_나이=("age", "mean"), 평균_혈당=("avg_glucose_level", "mean"))
    .reindex(GROUP_ORDER)
    .reset_index()
)
means.columns = ["그룹", "사람 수", "나이 평균", "평균 혈당의 평균"]
st.dataframe(
    means,
    hide_index=True,
    use_container_width=True,
    column_config={
        "사람 수": st.column_config.NumberColumn("사람 수", format="%d명"),
        "나이 평균": st.column_config.NumberColumn("나이 평균", format="%.2f세"),
        "평균 혈당의 평균": st.column_config.NumberColumn("평균 혈당의 평균", format="%.2f"),
    },
)

# ------------------------------------------------------------ 3) 막대그래프
st.header("3. 고혈압·심장병과 뇌졸중 비율")
LABELS = {0: "없음", 1: "있음"}
col1, col2 = st.columns(2)
for col, column_name, title in [
    (col1, "hypertension", "고혈압 여부별 뇌졸중 비율"),
    (col2, "heart_disease", "심장병 여부별 뇌졸중 비율"),
]:
    rates = pd.DataFrame(
        [
            {
                "구분": LABELS[value],
                "뇌졸중 비율(%)": stroke_rate(df[df[column_name] == value]),
                "사람 수": int((df[column_name] == value).sum()),
            }
            for value in (0, 1)
        ]
    )
    fig = px.bar(
        rates,
        x="구분",
        y="뇌졸중 비율(%)",
        text=rates["뇌졸중 비율(%)"].map(lambda v: f"{v:.2f}%"),
        hover_data=["사람 수"],
        title=title,
    )
    fig.update_layout(xaxis_title=title.split(" ")[0], yaxis_title="뇌졸중 비율(%)")
    fig.update_traces(textposition="outside")
    with col:
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------ 4) bmi 빈 값
st.header("4. bmi가 비어 있는 사람들")
bmi_missing = df[df["bmi"].isna()]
compare = pd.DataFrame(
    {
        "구분": ["bmi가 빈 사람", "전체"],
        "사람 수": [len(bmi_missing), len(df)],
        "뇌졸중 사람 수": [int((bmi_missing["stroke"] == 1).sum()), int((df["stroke"] == 1).sum())],
        "뇌졸중 비율(%)": [stroke_rate(bmi_missing), stroke_rate(df)],
    }
)
st.dataframe(
    compare,
    hide_index=True,
    use_container_width=True,
    column_config={
        "사람 수": st.column_config.NumberColumn("사람 수", format="%d명"),
        "뇌졸중 사람 수": st.column_config.NumberColumn("뇌졸중 사람 수", format="%d명"),
        "뇌졸중 비율(%)": st.column_config.NumberColumn("뇌졸중 비율(%)", format="%.2f%%"),
    },
)

# ------------------------------------------------------------ 5) 흡연 상태
st.header("5. 흡연 상태별 사람 수")
smoking = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)
smoking["흡연 상태"] = smoking["흡연 상태"].fillna("(빈 값)")
st.dataframe(
    smoking,
    hide_index=True,
    use_container_width=True,
    column_config={"사람 수": st.column_config.NumberColumn("사람 수", format="%d명")},
)
