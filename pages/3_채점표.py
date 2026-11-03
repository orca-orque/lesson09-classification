import pandas as pd
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

APP_TITLE = "뇌졸중 예측 실습실"
DATA_URL = "https://raw.githubusercontent.com/orca-orque/data/main/stroke.csv"

st.set_page_config(page_title=f"채점표 - {APP_TITLE}", page_icon="🧠", layout="wide")

# ---------------------------------------------------------------- 상수
# '분류 모델' 페이지에서 처음 골라져 있던 넷(체질량지수를 뺀 나머지)을 그대로 쓴다.
FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]

MODEL_LOGIT = "로지스틱 회귀(확률로 답하는 모델)"
MODEL_TREE = "의사결정트리(질문으로 답하는 모델)"
MODEL_BASE = "기준 모델(입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델)"

NO_VALUE = "계산할 수 없음"
MAX_ROWS = 20


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def count_cells(y_true, y_pred) -> dict:
    """뇌졸중(1)을 양성으로 두고 TP·FP·FN·TN 사람 수를 센다."""
    pairs = list(zip(list(y_true), list(y_pred)))
    return {
        "TP": sum(1 for a, p in pairs if a == 1 and p == 1),
        "FN": sum(1 for a, p in pairs if a == 1 and p == 0),
        "FP": sum(1 for a, p in pairs if a == 0 and p == 1),
        "TN": sum(1 for a, p in pairs if a == 0 and p == 0),
    }


def four_scores(c: dict) -> dict:
    """네 칸에서 정확도·재현율·정밀도·F1을 구한다. 나눌 수 없으면 None."""
    tp, fn, fp, tn = c["TP"], c["FN"], c["FP"], c["TN"]
    total = tp + fn + fp + tn
    accuracy = (tp + tn) / total if total else None
    recall = tp / (tp + fn) if (tp + fn) else None  # 분모: 실제 뇌졸중인 사람 수
    precision = tp / (tp + fp) if (tp + fp) else None  # 분모: 뇌졸중이라 예측한 사람 수
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {"정확도": accuracy, "재현율": recall, "정밀도": precision, "F1": f1}


def percent(value) -> str:
    return NO_VALUE if value is None else f"{value * 100:.2f}%"


def confusion_table(c: dict) -> pd.DataFrame:
    """행은 실제, 열은 예측. 칸마다 약자와 사람 수, 그리고 행·열 합계."""
    tp, fn, fp, tn = c["TP"], c["FN"], c["FP"], c["TN"]
    total = tp + fn + fp + tn
    table = pd.DataFrame(
        {
            "예측: 뇌졸중": [f"TP · {tp:,}명", f"FP · {fp:,}명", f"{tp + fp:,}명"],
            "예측: 아님": [f"FN · {fn:,}명", f"TN · {tn:,}명", f"{fn + tn:,}명"],
            "합계": [f"{tp + fn:,}명", f"{fp + tn:,}명", f"{total:,}명"],
        },
        index=["실제: 뇌졸중", "실제: 아님", "합계"],
    )
    return table


def person_table(frame: pd.DataFrame) -> pd.DataFrame:
    """나이가 많은 순으로 정렬하고 앞 스무 줄만 남긴다."""
    shown = frame.sort_values("age", ascending=False, kind="stable").head(MAX_ROWS)
    return pd.DataFrame(
        {
            "번호": shown["id"].tolist(),
            "나이": [f"{v:g}" for v in shown["age"]],
            "평균 혈당": [f"{v:g}" for v in shown["avg_glucose_level"]],
            "고혈압 여부": ["있음" if v == 1 else "없음" for v in shown["hypertension"]],
            "심장병 여부": ["있음" if v == 1 else "없음" for v in shown["heart_disease"]],
        }
    )


def list_title(label: str, detail: str, n: int) -> str:
    title = f"#### {label} — 전체 {n:,}명"
    if n > MAX_ROWS:
        title += f" (나이 많은 앞 {MAX_ROWS}명만 표시)"
    return title + f"\n\n{detail}"


# ---------------------------------------------------------------- 화면 시작
st.title("📝 채점표")
st.caption(
    "‘분류 모델’ 페이지와 같은 방법으로 데이터를 나누고 두 모델을 학습한 뒤, 테스트 데이터로 채점해요. "
    "입력 속성은 나이·평균 혈당·고혈압·심장병이고, 뇌졸중(stroke가 1)을 양성으로 둬요."
)

try:
    raw = load_data()
except Exception as e:  # 인터넷 연결이나 주소 문제
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n{e}")
    st.stop()

# 번호 순으로 정렬 → 열 명씩 묶기 → 각 묶음의 앞 세 명이 테스트용
df = raw.sort_values("id", kind="stable").reset_index(drop=True)
is_test = (df.index % 10) < 3
test_df = df[is_test]
train_df = df[~is_test]

X_train = train_df[FEATURES]
X_test = test_df[FEATURES]
y_train = train_df["stroke"]
y_test = test_df["stroke"]

# 크기 맞추기는 훈련용으로만 계산
scaler = StandardScaler().fit(X_train)
Xs_train = scaler.transform(X_train)
Xs_test = scaler.transform(X_test)

# 모델 학습 (로지스틱 회귀와 기준 모델은 크기를 맞춘 값, 의사결정트리는 원래 값)
logit = LogisticRegression(max_iter=1000).fit(Xs_train, y_train)
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(X_train, y_train)
baseline = DummyClassifier(strategy="most_frequent").fit(Xs_train, y_train)

pred_logit = logit.predict(Xs_test)
pred_tree = tree.predict(X_test)
pred_base = baseline.predict(Xs_test)

cells_logit = count_cells(y_test, pred_logit)
cells_tree = count_cells(y_test, pred_tree)
cells_base = count_cells(y_test, pred_base)

# ---------------------------------------------------------------- 1) 혼동행렬
st.header("1. 혼동행렬")
st.caption(
    "행은 실제, 열은 예측이에요. "
    "TP: 뇌졸중인데 뇌졸중이라 예측 · FN: 뇌졸중인데 아니라고 예측 · "
    "FP: 아닌데 뇌졸중이라 예측 · TN: 아닌데 아니라고 예측"
)
left, right = st.columns(2)
with left:
    st.subheader(MODEL_LOGIT)
    st.dataframe(confusion_table(cells_logit), use_container_width=True)
with right:
    st.subheader(MODEL_TREE)
    st.dataframe(confusion_table(cells_tree), use_container_width=True)

# ---------------------------------------------------------------- 2) 네 지표
st.header("2. 정확도·재현율·정밀도·F1")
n_test = len(test_df)
n_test_stroke = int((y_test == 1).sum())
st.write(f"테스트용 사람은 {n_test:,}명이고, 그중 뇌졸중인 사람은 {n_test_stroke:,}명이에요.")

rows = []
for name, cells in [(MODEL_LOGIT, cells_logit), (MODEL_TREE, cells_tree), (MODEL_BASE, cells_base)]:
    scores = four_scores(cells)
    rows.append({"모델": name, **{k: percent(v) for k, v in scores.items()}})
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
st.caption(
    "재현율 = TP ÷ 실제 뇌졸중인 사람 수, 정밀도 = TP ÷ 뇌졸중이라 예측한 사람 수, "
    "F1 = 2 × 정밀도 × 재현율 ÷ (정밀도 + 재현율). "
    f"뇌졸중이라 예측한 사람이 0명이면 정밀도와 F1을 ‘{NO_VALUE}’으로 적었어요. "
    f"정밀도와 재현율이 모두 0이어서 F1의 분모가 0이 될 때도 ‘{NO_VALUE}’으로 적었어요."
)

# ---------------------------------------------------------------- 3) 틀린 사람 목록
st.header("3. 틀린 사람 살펴보기")
choice = st.radio("어느 모델의 목록을 볼까요?", [MODEL_LOGIT, MODEL_TREE], horizontal=True)
pred = pred_logit if choice == MODEL_LOGIT else pred_tree
cells = cells_logit if choice == MODEL_LOGIT else cells_tree

missed = test_df[(y_test == 1) & (pred == 0)]  # 놓친 환자: 실제 뇌졸중인데 아니라고 예측
wrong = test_df[(y_test == 0) & (pred == 1)]  # 헛짚은 사람: 아닌데 뇌졸중이라고 예측

left, right = st.columns(2)
with left:
    st.markdown(list_title("놓친 환자", "실제 뇌졸중인데 아니라고 예측한 사람", len(missed)))
    if len(missed):
        st.dataframe(person_table(missed), hide_index=True, use_container_width=True)
    else:
        st.write("해당하는 사람이 없어요.")
with right:
    st.markdown(list_title("헛짚은 사람", "뇌졸중이 아닌데 뇌졸중이라고 예측한 사람", len(wrong)))
    if len(wrong):
        st.dataframe(person_table(wrong), hide_index=True, use_container_width=True)
    else:
        st.write("해당하는 사람이 없어요.")

st.write(
    f"왼쪽 목록의 사람 수 {len(missed):,}명은 혼동행렬의 FN 칸({cells['FN']:,}명)과, "
    f"오른쪽 목록의 사람 수 {len(wrong):,}명은 FP 칸({cells['FP']:,}명)과 같아요."
)
