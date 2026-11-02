import json
from pathlib import Path

import pandas as pd
import streamlit as st

APP_TITLE = "뇌졸중 예측 실습실"
APP_ICON = "🧠"
DATA_URL = "https://raw.githubusercontent.com/orca-orque/data/main/stroke.csv"

# 내_기록 폴더(이 파일과 같은 위치)와 저장 파일
RECORD_DIR = Path(__file__).parent / "내_기록"
RECORD_FILE = RECORD_DIR / "기록.json"

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

# 숫자 카드의 숫자를 크게 보이게 하는 스타일
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 3rem; font-weight: 700; }
    [data-testid="stMetricLabel"] p { font-size: 1.1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def load_record() -> dict:
    """내_기록 폴더의 json 파일을 읽어 온다. 없거나 깨졌으면 빈 기록을 돌려준다."""
    empty = {"meanings": {}, "source": ""}
    if not RECORD_FILE.exists():
        return empty
    try:
        with open(RECORD_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "meanings": dict(data.get("meanings", {})),
            "source": str(data.get("source", "")),
        }
    except (json.JSONDecodeError, OSError, AttributeError, TypeError, ValueError):
        return empty


def save_record(record: dict) -> bool:
    try:
        RECORD_DIR.mkdir(parents=True, exist_ok=True)
        with open(RECORD_FILE, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        return True
    except OSError:
        return False


def describe_values(series: pd.Series) -> str:
    """열의 값이 어떤 종류인지 한 줄로 설명한다."""
    values = series.dropna()
    if values.empty:
        return "값 없음"
    if series.dtype == "object" or values.nunique() <= 2:
        kinds = sorted(values.unique().tolist(), key=str)
        return ", ".join(str(k) for k in kinds)
    return f"숫자 ({values.min():g} ~ {values.max():g})"


# ---------------------------------------------------------------- 화면 시작
st.title(f"{APP_ICON} {APP_TITLE}")
st.caption("뇌졸중 환자 데이터가 어떤 모습인지 먼저 살펴봐요.")

try:
    df = load_data()
except Exception as e:  # 인터넷 연결이나 주소 문제
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n{e}")
    st.stop()

record = load_record()

# 1) 숫자 카드 네 개
total = len(df)
stroke_count = int((df["stroke"] == 1).sum())
ratio = stroke_count / total * 100 if total else 0

c1, c2, c3, c4 = st.columns(4)
with c1:
    with st.container(border=True):
        st.metric("전체 사람 수", f"{total:,}명")
with c2:
    with st.container(border=True):
        st.metric("열 개수", f"{df.shape[1]}개")
with c3:
    with st.container(border=True):
        st.metric("stroke가 1인 사람 수", f"{stroke_count:,}명")
with c4:
    with st.container(border=True):
        st.metric("stroke가 1인 비율", f"{ratio:.2f}%")

# 2) 열 설명 표 (우리말 뜻은 직접 입력)
st.subheader("열 설명 표")
st.caption("‘우리말 뜻’ 칸을 눌러 교재를 보고 직접 적어 보세요. 적은 내용은 자동으로 저장돼요.")

info = pd.DataFrame(
    {
        "열 이름": df.columns,
        "우리말 뜻": [record["meanings"].get(col, "") for col in df.columns],
        "값의 종류": [describe_values(df[col]) for col in df.columns],
        "빈 값 개수": [int(df[col].isna().sum()) for col in df.columns],
    }
)

edited = st.data_editor(
    info,
    key="info_editor",
    hide_index=True,
    use_container_width=True,
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "우리말 뜻": st.column_config.TextColumn("우리말 뜻", help="교재를 보고 직접 적어 보세요."),
        "빈 값 개수": st.column_config.NumberColumn("빈 값 개수", format="%d개"),
    },
)

new_meanings = {
    row["열 이름"]: ("" if pd.isna(row["우리말 뜻"]) else str(row["우리말 뜻"]))
    for _, row in edited.iterrows()
}

# 3) 처음 다섯 줄
st.subheader("데이터 처음 다섯 줄")
st.dataframe(df.head(5), use_container_width=True)

# 4) 데이터 출처
st.subheader("데이터 출처")
new_source = st.text_area(
    "출처",
    value=record["source"],
    height=100,
    placeholder="교재에 있는 출처 글을 여기에 적어 주세요. (적은 뒤 칸 밖을 누르면 저장돼요.)",
    label_visibility="collapsed",
)

# 바뀐 내용이 있으면 내_기록 폴더에 저장
new_record = {"meanings": new_meanings, "source": new_source}
if new_record != record:
    if save_record(new_record):
        st.toast("내_기록에 저장했어요.", icon="💾")
    else:
        st.warning("내_기록 폴더에 저장하지 못했어요. 폴더 쓰기 권한을 확인해 주세요.")
